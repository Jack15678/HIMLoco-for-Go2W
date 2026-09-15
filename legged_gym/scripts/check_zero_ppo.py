"""GPU environment/inference diagnostic; never calls learn() or optimizer.step()."""
import json
import argparse
from pathlib import Path
import sys
import tempfile

import isaacgym  # Must precede torch.
import torch
import legged_gym
import rsl_rl
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, export_policy_as_jit
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules import HIMActorCritic


def check_s10(env, policy, exported, metadata):
    import mujoco
    import numpy as np
    import xml.etree.ElementTree as ET
    from isaacgym import gymtorch
    from deploy.s10_mujoco import observation, targets, load_model

    properties = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    assert not properties['stiffness'].any() and not properties['damping'].any()
    assert not properties['hasLimits'][env.wheel_indices.cpu().numpy()].any()

    def nominal_gym_state():
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        env.dof_pos[:] = env.default_dof_pos
        env.dof_vel[:] = 0.
        env.root_states[:] = env.base_init_state
        env.root_states[:, :3] += env.env_origins
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.commands[:] = 0.

    for i in range(16):
        nominal_gym_state()
        # Check the initial response, before PD braking can reverse velocity.
        pulse = torch.zeros_like(env.torques)
        pulse[:, i] = 1.  # One Nm, one physics step.
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(pulse))
        env.gym.simulate(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        assert env.dof_vel[0, i] > .001, (env.dof_names[i], env.dof_vel[0, i].item())
    nominal_gym_state()
    settled_max_velocity = 0.
    for _ in range(round(5. / env.dt)):
        out = env.step(torch.zeros_like(env.actions))
        assert len(out[5]) == 0, 'Nominal Gym stance terminated'
        if _ >= round(4. / env.dt):
            settled_max_velocity = max(settled_max_velocity, env.dof_vel.abs().max().item())
    gym_height = (env.root_states[:, 2] - env.env_origins[:, 2]).tolist()
    assert all(.38 < h < .46 for h in gym_height), gym_height
    assert settled_max_velocity < 1., settled_max_velocity

    env.dof_pos[:] = env.default_dof_pos + torch.linspace(-.1, .1, 16, device=env.device)
    env.dof_vel[:] = torch.linspace(-4., 4., 16, device=env.device)
    env.commands[:, :3] = torch.tensor([.7, -.2, .4], device=env.device)
    env.base_ang_vel[:] = torch.tensor([.12, -.21, .32], device=env.device)
    env.projected_gravity[:] = torch.tensor([0., 0., -1.], device=env.device)
    env.actions[:] = torch.linspace(-.5, .5, 16, device=env.device)
    cpu = lambda value: value[0].detach().cpu().numpy()
    frame = observation(metadata, cpu(env.base_ang_vel), cpu(env.projected_gravity),
                        cpu(env.commands)[:3], cpu(env.dof_pos), cpu(env.dof_vel), cpu(env.actions))
    env.compute_observations()
    obs_error = float(np.max(np.abs(frame - cpu(env.obs_buf)[:57])))
    assert obs_error < 1e-6
    with torch.no_grad():
        raw = policy.act_inference(env.obs_buf)
        raw_error = (raw[0] - exported(env.obs_buf[0])).abs().max().item()
    assert raw_error < 1e-5
    action = np.linspace(-2., 2., 16).astype(np.float32)
    q_target, v_target = targets(metadata, action)
    expected = np.clip(np.asarray(metadata['p_gains']) * (q_target - cpu(env.dof_pos)) +
                       np.asarray(metadata['d_gains']) * (v_target - cpu(env.dof_vel)),
                       -np.asarray(metadata['torque_limits']), metadata['torque_limits'])
    actual = env._compute_torques(torch.tensor(action, device=env.device).repeat(2, 1))
    torque_error = float(np.max(np.abs(expected - cpu(actual))))
    assert torque_error < 2e-5
    assert max(abs(v_target)) > 5.  # Scale 5 is not a wheel-speed cap.

    model, data, joints, motors = load_model(metadata)
    q_ids, v_ids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    initial = data.qpos.copy()
    gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.
    for j, motor in zip(joints, motors):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial
        data.ctrl[motor] = 1.
        mujoco.mj_step(model, data)
        assert data.qvel[model.jnt_dofadr[j]] > 0., model.joint(j).name
    model.opt.gravity[:] = gravity
    mujoco.mj_resetData(model, data)
    data.qpos[:] = initial
    kp, kd, limit = map(np.asarray, (metadata['p_gains'], metadata['d_gains'], metadata['torque_limits']))
    for _ in range(round(5. / model.opt.timestep)):
        data.ctrl[motors] = np.clip(kp * (np.asarray(metadata['default_dof_pos']) - data.qpos[q_ids]) -
                                    kd * data.qvel[v_ids], -limit, limit)
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and .38 < data.qpos[2] < .46
    assert not data.warning.number.any()
    assert abs(data.time - 5.) < 1e-6  # Gym exposes dt as float32.
    mujoco_height = float(data.qpos[2])

    root = Path(legged_gym.LEGGED_GYM_ROOT_DIR) / 'resources/robots/s10'
    urdf = ET.parse(root / 'urdf/S10.urdf').getroot()
    for mesh in urdf.findall('.//mesh'):
        assert (root / 'urdf' / mesh.attrib['filename']).is_file()
    for link in urdf.findall('link'):
        inertia = link.find('inertial/inertia')
        if inertia is None:
            continue
        a = {k: float(v) for k, v in inertia.attrib.items()}
        eig = np.linalg.eigvalsh([[a['ixx'], a['ixy'], a['ixz']],
                                [a['ixy'], a['iyy'], a['iyz']], [a['ixz'], a['iyz'], a['izz']]])
        assert eig[0] > 0 and eig[2] <= eig[0] + eig[1]
    urdf_model = mujoco.MjModel.from_xml_path(str(root / 'urdf/S10.urdf'))
    urdf_data = mujoco.MjData(urdf_model)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = 0.
    for i, name in enumerate(env.dof_names):
        urdf_data.qpos[urdf_model.joint(name).qposadr[0]] = .01 * i
        data.qpos[model.joint(name).qposadr[0]] = .01 * i
    mujoco.mj_forward(urdf_model, urdf_data)
    mujoco.mj_forward(model, data)
    fk_error = max(float(np.abs(urdf_data.body(urdf_model.body(i).name).xpos -
                                data.body(urdf_model.body(i).name).xpos).max())
                   for i in range(1, urdf_model.nbody))
    assert fk_error < 1e-6
    assert abs(model.body_mass.sum() - 18.987425) < 1e-6
    official = [f'{leg}_{joint}_joint' for leg in ['fl', 'fr', 'hl', 'hr']
                for joint in ['hipx', 'hipy', 'knee']] + [f'{leg}_wheel_joint' for leg in ['fl', 'fr', 'hl', 'hr']]
    return dict(observation_max_error=obs_error, raw_action_max_error=raw_error,
                torque_max_error=torque_error, urdf_mjcf_fk_max_error=fk_error,
                policy_to_official=[env.dof_names.index(name) for name in official],
                mujoco_version=mujoco.__version__, static_pd_seconds=5., direction_checks=16,
                gym_direction_checks=16, gym_static_height=gym_height, mujoco_static_height=mujoco_height,
                gym_settled_max_velocity=settled_max_velocity,
                pd_dt=model.opt.timestep, policy_dt=model.opt.timestep * metadata['decimation'])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--export-dir', type=Path)
    check_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.headless = True
    args.num_envs = args.num_envs or 2
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    # Two-environment interface checks use a plane; capacity checks keep terrain.
    if args.num_envs == 2:
        cfg.terrain.mesh_type = 'plane'
        cfg.commands.curriculum = False
        cfg.noise.add_noise = False
        cfg.noise.noise_level = 0.
        for key in dir(cfg.domain_rand):
            if isinstance(getattr(cfg.domain_rand, key), bool):
                setattr(cfg.domain_rand, key, False)
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    env.reset()
    assert env.device.startswith('cuda') and env.dof_pos.is_cuda
    assert env.sim_params.use_gpu_pipeline and env.sim_params.physx.use_gpu
    policy = HIMActorCritic(env.num_obs, env.num_privileged_obs,
                           env.num_one_step_obs, env.num_actions,
                           **class_to_dict(train_cfg.policy)).to(env.device).eval()
    if env.num_envs > 2:
        from rsl_rl.algorithms import HIMPPO
        storage_owner = HIMPPO(policy, device=env.device, **class_to_dict(train_cfg.algorithm))
        storage_owner.init_storage(env.num_envs, 48, [342], [262], [16])
    with torch.no_grad(), tempfile.TemporaryDirectory() as tmp:
        tmp = str(check_args.export_dir or tmp)
        if check_args.strict:
            export_policy_as_jit(policy, tmp, env=env)
        else:
            export_policy_as_jit(policy, tmp)
        exported = torch.jit.load(str(Path(tmp) / 'policy.pt')).to(env.device)
        obs = env.get_observations()
        raw = policy.act_inference(obs)
        jit_error = (raw[0] - exported(obs[0])).abs().max().item()
        assert jit_error < 1e-5
        for _ in range(48):
            if env.num_envs > 2:
                raw = storage_owner.act(env.get_observations(), env.get_privileged_observations())
            else:
                raw = policy.act_inference(env.get_observations())
            out = env.step(raw)
            assert torch.isfinite(out[0]).all() and torch.isfinite(out[2]).all()
            assert torch.isfinite(out[1]).all()
            if env.num_envs > 2:
                next_critic = out[1].clone()
                next_critic[out[5]] = out[6]
                storage_owner.process_env_step(out[2], out[3], out[4], next_critic)
        if env.num_envs > 2:
            storage_owner.compute_returns(env.get_privileged_observations())
            assert torch.isfinite(storage_owner.storage.returns).all()
            assert not storage_owner.optimizer.state and not policy.estimator.optimizer.state
        s10_report = None
        if check_args.strict and args.task == 's10' and env.num_envs == 2:
            s10_report = check_s10(env, policy, exported, json.loads((Path(tmp) / 'policy.json').read_text()))
    free_memory, total_memory = torch.cuda.mem_get_info()
    report = dict(task=args.task, num_envs=env.num_envs, ppo_updates=0,
                  legged_gym=legged_gym.__file__, rsl_rl=rsl_rl.__file__,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu_pipeline=env.sim_params.use_gpu_pipeline, gpu_physics=env.sim_params.physx.use_gpu,
                  dof_names=env.dof_names, wheel_indices=env.wheel_indices.tolist(),
                  body_names=env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]),
                  feet_indices=env.feet_indices.tolist(),
                  penalty_indices=env.penalised_contact_indices.tolist(),
                  termination_indices=env.termination_contact_indices.tolist(),
                  obs_shape=list(env.obs_buf.shape), critic_shape=list(env.privileged_obs_buf.shape),
                  policy_dt=env.dt, jit_max_error=jit_error,
                  gpu_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                  gpu_used_mb=(total_memory - free_memory) / 2**20,
                  gpu_free_mb=free_memory / 2**20,
                  rollout_storage_steps=48 if env.num_envs > 2 else 0)
    report['s10_contract'] = s10_report
    if env.num_envs == 2:
        env.reset()
        env.dof_vel[:, env.wheel_indices] = 3.
        before = env.dof_vel.clone()
        env._reward_dof_vel()
        report['reward_mutates_velocity'] = not torch.equal(before, env.dof_vel)
        env.dof_pos[:, env.wheel_indices] = 1.7
        before = env.dof_pos.clone()
        env.compute_observations()
        report['observation_mutates_position'] = not torch.equal(before, env.dof_pos)
        env.reset()
        env.obs_buf[:] = 7.
        env.episode_length_buf[0] = env.max_episode_length
        env.episode_length_buf[1] = 0
        out = env.step(torch.full_like(env.actions, .1))
        report['reset_ids'] = out[5].tolist()
        report['reset_history_nonzero'] = torch.count_nonzero(env.obs_buf[0, 57:]).item()
        report['continuous_history_preserved'] = bool((env.obs_buf[1, 57:] == 7.).all())
        report['reset_previous_action_nonzero'] = torch.count_nonzero(env.obs_buf[0, 41:57]).item()
        report['terminal_shape'] = list(out[6].shape)
        report['terminal_action_preserved'] = bool(torch.allclose(out[6][0, 41:57], torch.full((16,), .1, device=env.device)))
        if check_args.strict:
            assert not report['reward_mutates_velocity'] and not report['observation_mutates_position']
            assert report['reset_ids'] == [0] and report['reset_history_nonzero'] == 0
            assert report['continuous_history_preserved'] and report['reset_previous_action_nonzero'] == 0
            assert report['terminal_action_preserved'] and report['terminal_shape'] == [1, 262]
            assert torch.equal(env.last_dof_vel, env.dof_vel)
            if env.cfg.commands.heading_command:
                q = env.base_quat[0]
                yaw = torch.atan2(2 * (q[3]*q[2] + q[0]*q[1]), 1 - 2 * (q[1]**2 + q[2]**2))
                error = (env.commands[0, 3] - yaw + torch.pi) % (2 * torch.pi) - torch.pi
                assert torch.allclose(env.commands[0, 2], torch.clamp(.5 * error, -2., 2.))
            # Exercise the other two observation paths with nonzero wheel angles.
            env.dof_pos[:, env.wheel_indices] = 2.
            before = env.dof_pos.clone()
            current = env.get_current_obs()
            terminal = env.compute_termination_observations(torch.tensor([0], device=env.device))
            assert torch.equal(before, env.dof_pos)
            assert not torch.count_nonzero(current[:, 9 + env.wheel_indices])
            assert not torch.count_nonzero(terminal[:, 9 + env.wheel_indices])
            assert all('hip' in env.dof_names[i] for i in [0, 4, 8, 12])
        # Stop at the target-network input, before any estimator/PPO optimizer step.
        class Captured(Exception):
            pass
        target_input = []
        def capture(module, inputs):
            target_input.append(inputs[0].detach().clone())
            raise Captured()
        hook = policy.estimator.target.register_forward_pre_hook(capture)
        try:
            policy.estimator.update(torch.zeros((2, 342), device=env.device),
                                    torch.arange(262., device=env.device).repeat(2, 1))
        except Captured:
            pass
        finally:
            hook.remove()
        report['estimator_target_columns'] = target_input[0][0].tolist()
        assert not policy.estimator.optimizer.state
        if check_args.strict:
            assert report['estimator_target_columns'] == list(range(6)) + list(range(9, 60))
            runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root=None)
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                checkpoint = str(Path(checkpoint_dir) / 'roundtrip.pt')
                runner.current_learning_iteration = 7  # Synthetic counter, no learning loop.
                runner.tot_timesteps, runner.tot_time = 123, 4.5
                saved_std = runner.alg.actor_critic.std.detach().clone()
                saved_lr = runner.alg.actor_critic.estimator.optimizer.param_groups[0]['lr']
                runner.alg.optimizer.param_groups[0]['lr'] = .0002
                runner.save(checkpoint)
                with torch.no_grad():
                    runner.alg.actor_critic.std.zero_()
                runner.current_learning_iteration = 0
                runner.alg.actor_critic.estimator.optimizer.param_groups[0]['lr'] = .123
                runner.load(checkpoint)
                assert runner.current_learning_iteration == 7 and runner.tot_timesteps == 123 and runner.tot_time == 4.5
                assert torch.equal(runner.alg.actor_critic.std, saved_std)
                assert runner.alg.actor_critic.estimator.optimizer.param_groups[0]['lr'] == saved_lr
                assert runner.alg.learning_rate == .0002
                assert not runner.alg.optimizer.state and not runner.alg.actor_critic.estimator.optimizer.state
            report['checkpoint_roundtrip_without_updates'] = True
    print('ZERO_PPO_RESULT ' + json.dumps(report, sort_keys=True))
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
