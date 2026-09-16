"""PD-only impulse response: one 0.1 rad/s joint perturbation per robot, no policy.

python legged_gym/scripts/probe_s10_pd.py --engine gym --pd-dt .005 --profile air \
    --output /tmp/s10-pd-air-5ms --task s10 --headless --num_envs 16
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parents[2] / 'logs/S10_HIM/Sep15_08-33-31_scratch_1000_20260915'


def summarize(velocity, torque, limits, dt):
    # Arrays are physical steps x 16 independently excited robots x 16 joints.
    tail = velocity[round(2. / dt):]
    tail_torque = torque[round(2. / dt):]
    return dict(last_second_velocity_rms=np.sqrt(np.square(tail).mean((0, 1))).tolist(),
                last_second_peak_velocity=np.abs(tail).max((0, 1)).tolist(),
                last_second_saturation_fraction=(np.abs(tail_torque) >= .99 * limits).mean((0, 1)).tolist(),
                per_excitation_max_tail_velocity=np.abs(tail).max((0, 2)).tolist())


def gym_probe(args):
    import isaacgym
    from isaacgym import gymtorch
    import torch
    from legged_gym.envs import S10RoughCfg
    from legged_gym.utils import get_args, task_registry
    from legged_gym.utils.helpers import class_to_dict
    from legged_gym.scripts.evaluate_s10 import restore_config

    cli = get_args()
    cfg, _ = task_registry.get_cfgs('s10')
    saved = json.loads((RUN / 'config.json').read_text())
    restore_config(cfg, saved['env'])
    cfg.env.num_envs = 16
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    cfg.commands.curriculum = False
    cfg.noise.add_noise = False
    for name, value in vars(cfg.domain_rand).items():
        if isinstance(value, bool):
            setattr(cfg.domain_rand, name, False)
    cfg.sim.dt = args.pd_dt
    cfg.control.decimation = round(.02 / args.pd_dt)
    if args.profile == 'air':
        cfg.sim.gravity = [0., 0., 0.]
        cfg.init_state.pos = [0., 0., 1.]
    env, _ = task_registry.make_env('s10', args=cli, env_cfg=cfg)
    assert env.num_envs == 16
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0.
    env.root_states[:] = env.base_init_state
    env.root_states[:, :3] += env.env_origins
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    action = torch.zeros_like(env.actions)
    rows = {key: [] for key in ['velocity', 'torque', 'height']}
    for step in range(round(3. / args.pd_dt)):
        if step == round(1. / args.pd_dt):
            env.dof_vel[torch.arange(16), torch.arange(16)] += .1
            env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        torque = env._compute_torques(action)
        rows['torque'].append(torque.cpu().numpy().copy())
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(torque))
        env.gym.simulate(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        rows['velocity'].append(env.dof_vel.cpu().numpy().copy())
        rows['height'].append((env.root_states[:, 2] - env.env_origins[:, 2]).cpu().numpy().copy())
    return rows, env.dof_names, env.torque_limits.cpu().numpy(), class_to_dict(cfg), {}


def mujoco_probe(args):
    import mujoco
    from deploy.s10_mujoco import load_model
    cfg = json.loads((RUN / 'evaluation_900/policy.json').read_text())
    cfg['sim_dt'], cfg['decimation'] = args.pd_dt, round(.02 / args.pd_dt)
    model, initial, joints, motors = load_model(cfg)
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    if args.profile == 'air':
        model.opt.gravity[:] = 0.
        initial.qpos[2] = 1.
    mujoco.mj_forward(model, initial)
    kp, kd, limits = map(np.asarray, (cfg['p_gains'], cfg['d_gains'], cfg['torque_limits']))
    # Frozen-inertia, no-contact linearization of externally computed semi-implicit PD.
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, initial.qM)
    inv = np.linalg.inv(mass)[np.ix_(vids, vids)]
    h = args.pd_dt
    k, d, eye = inv @ np.diag(kp), inv @ np.diag(kd), np.eye(16)
    transition = np.block([[eye - h*h*k, h*eye - h*h*d], [-h*k, eye - h*d]])
    linear = dict(frozen_inertia_spectral_radius=float(np.abs(np.linalg.eigvals(transition)).max()),
                  effective_joint_inertia=(1. / np.diag(inv)).tolist())
    rows = {key: [] for key in ['velocity', 'torque', 'height']}
    states = [mujoco.MjData(model) for _ in range(16)]
    for state in states:
        state.qpos[:] = initial.qpos
        mujoco.mj_forward(model, state)
    for step in range(round(3. / h)):
        vr, tr, hr = [], [], []
        for index, state in enumerate(states):
            if step == round(1. / h):
                state.qvel[vids[index]] += .1
            tau = np.clip(kp * (np.asarray(cfg['default_dof_pos']) - state.qpos[qids]) - kd * state.qvel[vids], -limits, limits)
            state.ctrl[motors] = tau
            mujoco.mj_step(model, state)
            vr.append(state.qvel[vids].copy())
            tr.append(tau)
            hr.append(state.qpos[2])
        rows['velocity'].append(vr)
        rows['torque'].append(tr)
        rows['height'].append(hr)
    assert not any(state.warning.number.any() for state in states)
    return rows, cfg['dof_names'], limits, cfg, linear


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['gym', 'mujoco'])
    parser.add_argument('--pd-dt', type=float, default=.005)
    parser.add_argument('--profile', choices=['stand', 'air'], default='air')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.self_check:
        v = np.zeros((6, 16, 16)); t = v.copy(); v[4:, :, 3] = 2.; t[4:, :, 3] = 10.
        result = summarize(v, t, np.full(16, 10.), .5)
        assert result['last_second_velocity_rms'][3] == 2.
        assert result['last_second_saturation_fraction'][3] == 1.
        assert result['last_second_saturation_fraction'][0] == 0.
        print('PD summary self-check passed')
    else:
        assert args.engine and args.output and args.pd_dt in [.005, .0025]
        args.output.mkdir(parents=True, exist_ok=False)
        sys.argv = [sys.argv[0]] + remaining
        rows, names, limits, cfg, linear = (gym_probe if args.engine == 'gym' else mujoco_probe)(args)
        rows = {key: np.asarray(value) for key, value in rows.items()}
        assert all(np.isfinite(value).all() for value in rows.values())
        result = dict(engine=args.engine, profile=args.profile, pd_dt=args.pd_dt, policy_dt=.02,
                      input='Constant default joint positions and zero wheel targets; +0.1 rad/s on one different joint in each robot at t=1s.',
                      ppo_updates=0, dof_names=names, linearization=linear,
                      final_height=rows['height'][-1].tolist(),
                      **summarize(rows['velocity'], rows['torque'], limits, args.pd_dt))
        np.savez_compressed(args.output / 'trace.npz', **rows)
        (args.output / 'config.json').write_text(json.dumps(cfg, indent=2))
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
        print('PD_PROBE_RESULT', json.dumps(result), flush=True)
