"""Ten-second, zero-command acceptance: nominal + 16 fixed perturbations; no resets."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse
import numpy as np
import torch
from legged_gym.envs import S10RoughCfg
from legged_gym.utils import get_args, task_registry, export_policy_as_jit
from legged_gym.utils.helpers import class_to_dict
from legged_gym.scripts.evaluate_s10 import restore_config
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.resume_s10_candidate import write_json
from rsl_rl.modules import HIMActorCritic


class StandingSpawn:
    """Keep the chassis above the plane before prepare_sim builds contact state."""
    def __init__(self, gym, height):
        self.gym, self.height = gym, height
        self.created_heights = []

    def __getattr__(self, name):
        return getattr(self.gym, name)

    def create_actor(self, env, asset, pose, *args):
        pose.p.z += self.height
        self.created_heights.append(pose.p.z)
        return self.gym.create_actor(env, asset, pose, *args)


def stance_summary(rows, initial_root, dt):
    height = rows['root'][:, :, 2]
    contact = np.linalg.norm(rows['base_contact_force'], axis=-1).max(-1) > 1.
    tail = slice(round(5./dt), None)
    result = []
    for i in range(height.shape[1]):
        hits = np.flatnonzero(contact[:, i])
        height_ok = bool(((height[tail, i] >= .38) & (height[tail, i] <= .46)).all())
        tilt_ok = bool((rows['tilt'][tail, i] < np.deg2rad(15.)).all())
        result.append(dict(trial=i, seed=None if i == 0 else i-1,
            passed=bool(not len(hits) and height_ok and tilt_ok),
            first_base_contact_seconds=float((hits[0]+1)*dt) if len(hits) else None,
            tail_height_min=float(height[tail, i].min()), tail_height_max=float(height[tail, i].max()),
            tail_tilt_max_degrees=float(np.rad2deg(rows['tilt'][tail, i]).max()),
            displacement_xy=(rows['root'][-1, i, :2]-initial_root[i, :2]).tolist(),
            tail_velocity_mean=rows['velocity'][tail, i].mean(0).tolist(),
            tail_velocity_rms=np.sqrt(np.square(rows['velocity'][tail, i]).mean(0)).tolist(),
            tail_wheel_velocity_rms=np.sqrt(np.square(rows['dof_velocity'][tail, i][:, [3, 7, 11, 15]]).mean(0)).tolist()))
    return dict(passed=bool(result[0]['passed'] and sum(r['passed'] for r in result[1:]) >= 15),
        nominal_passed=result[0]['passed'], perturbed_passed=sum(r['passed'] for r in result[1:]),
        contact_force_threshold_N=1., reset=False, trials=result)


def command_at(step, replay):
    if replay and 100 <= step < 300:
        return [.5, 0., 0.]
    if replay and 300 <= step < 400:
        return [0., 0., .4]
    return [0., 0., 0.]


def main(args, run, output_name, replay=False):
    assert args.task == 's10' and args.num_envs == 17
    output = run / output_name
    output.mkdir(exist_ok=False)
    saved = json.loads((run / 'config.json').read_text())
    cfg, _ = task_registry.get_cfgs('s10')
    restore_config(cfg, saved['env'])
    cfg.env.num_envs, cfg.env.episode_length_s = 17, 11.
    cfg.terrain.mesh_type, cfg.terrain.curriculum = 'plane', False
    cfg.commands.curriculum, cfg.commands.heading_command = False, False
    cfg.commands.resampling_time = 1e6
    for name in saved['env']['commands']['ranges']:
        setattr(cfg.commands.ranges, name, [0., 0.])
    cfg.noise.add_noise, cfg.noise.noise_level = False, 0.
    for name in dir(cfg.domain_rand):
        if isinstance(getattr(cfg.domain_rand, name), bool):
            setattr(cfg.domain_rand, name, False)
    spawn = StandingSpawn(isaacgym.gymapi.acquire_gym(), cfg.init_state.pos[2])
    with patch.object(isaacgym.gymapi, 'acquire_gym', return_value=spawn):
        env, _ = task_registry.make_env('s10', args=args, env_cfg=cfg)
    assert len(spawn.created_heights) == 17 and np.allclose(spawn.created_heights, cfg.init_state.pos[2])
    # prepare_sim may advance internal state before tensors are acquired.
    assert ((env.root_states[:, 2] - env.env_origins[:, 2]) > .4).all()
    assert (torch.norm(env.contact_forces[:, env.termination_contact_indices], dim=-1) <= 1.).all()
    write_json(output / 'creation_state.json', dict(root=env.root_states.tolist(),
        created_heights=spawn.created_heights, sim_time=env.gym.get_sim_time(env.sim),
        base_contact_force=env.contact_forces[:, env.termination_contact_indices].tolist(),
        added_warmup_steps=0, state_writes_per_tensor=1))
    checkpoint = torch.load(run / f'model_{args.checkpoint}.pt', map_location='cpu')
    assert checkpoint['iter'] == args.checkpoint
    write_json(output / 'checkpoint.json', dict(iteration=checkpoint['iter'],
        environment_steps=checkpoint['tot_timesteps'], replay=replay,
        command_schedule='0-2s idle, 2-6s vx=.5, 6-8s yaw=.4, 8-10s idle' if replay else '10s zero command'))
    actor = HIMActorCritic(342, 262, 57, 16, **saved['train']['policy']).to(env.device).eval()
    actor.load_state_dict(checkpoint['model_state_dict'])
    # Initialize each state tensor once, without an intermediate randomized reset.
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0.
    env.root_states[:] = env.base_init_state
    env.root_states[:, :3] += env.env_origins
    env.root_states[:, 7:13] = 0.
    leg_ids = [i for i in range(16) if i not in env.wheel_indices.tolist()]
    perturbations = np.zeros((17, 14))
    for seed in range(16):
        perturbations[seed+1] = np.random.default_rng(seed).uniform(-.02, .02, 14)
    env.dof_pos[:, leg_ids] += torch.tensor(perturbations[:, :12], device=env.device, dtype=torch.float32)
    roll, pitch = [torch.tensor(perturbations[:, i], device=env.device, dtype=torch.float32) for i in [12, 13]]
    env.root_states[:, 3:7] = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.commands[:] = 0.
    env.actions[:] = env.last_actions[:] = env.last_last_actions[:] = 0.
    env.obs_buf[:] = env.privileged_obs_buf[:] = 0.
    env.episode_length_buf[:] = 0
    env.base_lin_vel[:] = env.base_ang_vel[:] = 0.
    env.projected_gravity[:] = quat_rotate_inverse(env.root_states[:, 3:7], env.gravity_vec)
    env.compute_observations()
    initial_root = env.root_states.cpu().numpy().copy()
    initial_root[:, :3] -= env.env_origins.cpu().numpy()
    write_json(output / 'initial_states.json', dict(seeds=list(range(16)), nominal_trial=0,
        leg_indices=leg_ids, perturbation_columns='12 leg angles, roll, pitch (radians)',
        perturbations=perturbations.tolist(), dof_pos=env.dof_pos.tolist(),
        dof_velocity=env.dof_vel.tolist(), root_local=initial_root.tolist()))
    write_json(output / 'evaluation_config.json', class_to_dict(cfg))
    export_policy_as_jit(actor, str(output), env=env)
    with torch.no_grad():
        jit = torch.jit.load(str(output / 'policy.pt')).eval()
        export_error = float((actor.act_inference(env.obs_buf)[0].cpu()-jit(env.obs_buf[0].cpu())).abs().max())
        assert export_error < 1e-5
    # Keep each robot's complete trajectory after any fall; never mask it with a reset.
    env.reset_idx = lambda ids: None
    rows = {k: [] for k in ['root', 'tilt', 'velocity', 'dof_velocity', 'dof_position', 'torque',
                           'position_target', 'velocity_target', 'base_contact_force', 'action']}
    original_torque, original_gym = env._compute_torques, env.gym
    def torque(action):
        value = original_torque(action)
        q = env.default_dof_pos + action * env.action_scale
        v = torch.zeros_like(action)
        v[:, env.wheel_indices] = action[:, env.wheel_indices] * cfg.control.vel_scale
        rows['position_target'].append(q.cpu().numpy().copy())
        rows['velocity_target'].append(v.cpu().numpy().copy())
        rows['torque'].append(value.cpu().numpy().copy())
        return value
    env._compute_torques = torque
    class ObservedGym:
        def __getattr__(self, name):
            return getattr(original_gym, name)
        def simulate(self, sim):
            original_gym.simulate(sim)
            original_gym.refresh_dof_state_tensor(sim)
            original_gym.refresh_actor_root_state_tensor(sim)
            original_gym.refresh_net_contact_force_tensor(sim)
            root = env.root_states.clone()
            root[:, :3] -= env.env_origins
            lin = quat_rotate_inverse(root[:, 3:7], root[:, 7:10])
            ang = quat_rotate_inverse(root[:, 3:7], root[:, 10:13])
            gravity = quat_rotate_inverse(root[:, 3:7], env.gravity_vec)
            for key, value in dict(root=root, tilt=torch.acos((-gravity[:, 2]).clamp(-1., 1.)),
                velocity=torch.cat([lin[:, :2], ang[:, 2:3]], dim=1), dof_velocity=env.dof_vel, dof_position=env.dof_pos,
                base_contact_force=env.contact_forces[:, env.termination_contact_indices]).items():
                assert torch.isfinite(value).all(), key
                rows[key].append(value.cpu().numpy().copy())
    env.gym = ObservedGym()
    start_time = env.gym.get_sim_time(env.sim)
    with torch.inference_mode():
        for policy_step in range(500):
            env.commands[:, :3] = torch.tensor(command_at(policy_step, replay), device=env.device)
            env.obs_buf[:, 6:9] = env.commands[:, :3] * env.commands_scale
            action = actor.act_inference(env.obs_buf)
            rows['action'].append(action.cpu().numpy().copy())
            env.step(action)
    elapsed = env.gym.get_sim_time(env.sim)-start_time
    assert abs(elapsed-10) < 1e-5
    rows = {k: np.asarray(v) for k, v in rows.items()}
    assert rows['torque'].shape == (4000, 17, 16)
    assert all(np.isfinite(v).all() for v in rows.values())
    np.savez_compressed(output / 'gym_trace.npz', dt=env.sim_params.dt, **rows)
    # Render the saved Gym pose in the existing MuJoCo viewer; no physics replay.
    from deploy.s10_mujoco import load_model
    metadata = json.loads((output / 'policy.json').read_text())
    model, data, joints, motors = load_model(metadata)
    root = rows['root'][7::8, 0]
    qpos, qvel = np.zeros((500, model.nq)), np.zeros((500, model.nv))
    qpos[:, :3], qpos[:, 3:7] = root[:, :3], root[:, [6, 3, 4, 5]]
    qpos[:, model.jnt_qposadr[joints]] = rows['dof_position'][7::8, 0]
    qvel[:, :3] = root[:, 7:10]
    qvel[:, 3:6] = quat_rotate_inverse(torch.tensor(root[:, 3:7]), torch.tensor(root[:, 10:13])).numpy()
    qvel[:, model.jnt_dofadr[joints]] = rows['dof_velocity'][7::8, 0]
    np.savez_compressed(output / 'gym_video_trace.npz', dt=env.dt, qpos=qpos, qvel=qvel)
    result = stance_summary(rows, initial_root, env.sim_params.dt)
    result.update(actual_seconds=elapsed, export_error=export_error, policy='mean', ppo_updates=0,
        checkpoint_iteration=args.checkpoint, replay=replay,
        pd_saturation_fraction=(np.abs(rows['torque']) >= .99*env.torque_limits.cpu().numpy()).mean((0, 1)).tolist(),
        segment_velocity_means=[rows['velocity'][a:b, 0].mean(0).tolist() for a, b in [(400,800), (2000,2400), (2800,3200), (3600,4000)]])
    write_json(output / 'gym_summary.json', result)
    print('STANDING_RESULT', json.dumps(result), flush=True)


def mujoco_stand(run, output_name, replay=False):
    import mujoco
    from deploy.s10_mujoco import observation, targets, load_model
    output = run / output_name
    assert not (output / 'mujoco_trace.npz').exists()
    cfg = json.loads((output / 'policy.json').read_text())
    policy = torch.jit.load(str(output / 'policy.pt')).eval()
    model, data, joints, motors = load_model(cfg)
    assert np.isclose(model.opt.timestep, .0025) and cfg['decimation'] == 8
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    base = model.body('base_link').id
    initial_position = data.qpos[:3].copy()
    kp, kd, limits = map(np.asarray, (cfg['p_gains'], cfg['d_gains'], cfg['torque_limits']))
    history, action, velocity, force = np.zeros((6, 57), np.float32), np.zeros(16, np.float32), np.empty(6), np.empty(6)
    rows = {k: [] for k in ['qpos', 'qvel', 'torque', 'position_target', 'velocity_target', 'velocity', 'tilt', 'base_contact_force']}
    with torch.inference_mode():
        for policy_step in range(500):
            mujoco.mj_forward(model, data)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 1)
            gravity = data.xmat[base].reshape(3, 3).T @ np.array([0., 0., -1.])
            history[1:] = history[:-1].copy()
            history[0] = observation(cfg, velocity[:3], gravity, command_at(policy_step, replay), data.qpos[qids], data.qvel[vids], action)
            action = policy(torch.from_numpy(history.ravel())).numpy()
            assert np.isfinite(action).all()
            action = np.clip(action, -cfg['clip_actions'], cfg['clip_actions'])
            q, v = targets(cfg, action)
            for _ in range(8):
                tau = np.clip(kp*(q-data.qpos[qids])+kd*(v-data.qvel[vids]), -limits, limits)
                data.ctrl[motors] = tau
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                contact_force = 0.
                for c in range(data.ncon):
                    contact = data.contact[c]
                    if base in model.geom_bodyid[[contact.geom1, contact.geom2]]:
                        mujoco.mj_contactForce(model, data, c, force)
                        contact_force = max(contact_force, np.linalg.norm(force[:3]))
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 1)
                tilt = np.arccos(np.clip(data.xmat[base].reshape(3, 3)[2, 2], -1., 1.))
                for key, value in dict(qpos=data.qpos, qvel=data.qvel, torque=tau, position_target=q,
                    velocity_target=v, velocity=velocity[[3, 4, 2]], tilt=tilt, base_contact_force=contact_force).items():
                    assert np.isfinite(value).all(), key
                    rows[key].append(np.array(value).copy())
    rows = {k: np.asarray(v) for k, v in rows.items()}
    np.savez_compressed(output / 'mujoco_trace.npz', dt=model.opt.timestep, **rows)
    np.savez_compressed(output / 'mujoco_video_trace.npz', dt=model.opt.timestep*8,
                        qpos=rows['qpos'][7::8], qvel=rows['qvel'][7::8])
    hits = np.flatnonzero(rows['base_contact_force'] > 1.)
    tail = slice(2000, None)
    height, tilt = rows['qpos'][tail, 2], rows['tilt'][tail]
    result = dict(passed=bool(not len(hits) and ((height >= .38)&(height <= .46)).all() and (tilt < np.deg2rad(15.)).all()),
        actual_seconds=data.time, pd_steps=4000, policy_steps=500, history_dt=model.opt.timestep*8,
        first_base_contact_seconds=float((hits[0]+1)*model.opt.timestep) if len(hits) else None,
        tail_height_min=float(height.min()), tail_height_max=float(height.max()),
        tail_tilt_max_degrees=float(np.rad2deg(tilt).max()),
        displacement_xy=(data.qpos[:2]-initial_position[:2]).tolist(),
        tail_velocity_mean=rows['velocity'][tail].mean(0).tolist(),
        tail_velocity_rms=np.sqrt(np.square(rows['velocity'][tail]).mean(0)).tolist(),
        tail_wheel_velocity_rms=np.sqrt(np.square(rows['qvel'][tail][:, vids[cfg['wheel_indices']]]).mean(0)).tolist(),
        pd_saturation_fraction=(np.abs(rows['torque']) >= .99*limits).mean(0).tolist(),
        warnings=data.warning.number.tolist(), reset=False, ppo_updates=0, replay=replay,
        checkpoint_iteration=json.loads((output / 'checkpoint.json').read_text())['iteration'],
        segment_velocity_means=[rows['velocity'][a:b].mean(0).tolist() for a, b in [(400,800), (2000,2400), (2800,3200), (3600,4000)]])
    write_json(output / 'mujoco_summary.json', result)
    assert abs(data.time-10.) < 1e-5
    print('MUJOCO_STANDING_RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        assert command_at(99, True) == command_at(400, True) == [0., 0., 0.]
        assert command_at(100, True) == command_at(299, True) == [.5, 0., 0.]
        assert command_at(300, True) == command_at(399, True) == [0., 0., .4]
        assert command_at(200, False) == [0., 0., 0.]
        from unittest.mock import Mock
        backend = Mock()
        pose = isaacgym.gymapi.Transform()
        StandingSpawn(backend, .45).create_actor('env', 'asset', pose, 's10', 0, 1, 0)
        assert np.isclose(pose.p.z, .45) and backend.create_actor.call_count == 1
        rows = dict(root=np.zeros((10, 17, 13)), tilt=np.zeros((10, 17)),
                    base_contact_force=np.zeros((10, 17, 1, 3)),
                    velocity=np.zeros((10, 17, 3)), dof_velocity=np.zeros((10, 17, 16)))
        rows['root'][:, :, 2] = .42
        initial = rows['root'][0].copy()
        assert stance_summary(rows, initial, 1.)['passed']
        rows['base_contact_force'][2, 1, 0, 2] = 2.
        result = stance_summary(rows, initial, 1.)
        assert result['passed'] and result['trials'][1]['first_base_contact_seconds'] == 3.
        rows['root'][7, 2, 2] = .37
        assert not stance_summary(rows, initial, 1.)['passed']
        rows['base_contact_force'][0, 0, 0, 2] = 2.
        assert not stance_summary(rows, initial, 1.)['nominal_passed']
        print('Stance threshold and first-contact self-check passed')
    else:
        is_mujoco, replay = '--mujoco' in sys.argv, '--replay' in sys.argv
        sys.argv = [arg for arg in sys.argv if arg not in ['--mujoco', '--replay']]
        args = get_args()
        assert args.load_run and args.checkpoint in [500, 1000]
        run = Path(LEGGED_GYM_ROOT_DIR) / 'logs/S10_HIM' / args.load_run
        output_name = ('replay_' if replay else 'standing_') + str(args.checkpoint)
        if is_mujoco:
            mujoco_stand(run, output_name, replay)
        else:
            main(args, run, output_name, replay)
