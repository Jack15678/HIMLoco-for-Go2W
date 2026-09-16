"""Evaluate one saved S10 policy, without PPO updates.

python legged_gym/scripts/evaluate_s10.py --task s10 --headless \
    --load_run Sep15_08-33-31_scratch_1000_20260915 --checkpoint 900
Outputs live beside the checkpoint in evaluation_<iteration>/.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # Must precede torch.
import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, export_policy_as_jit
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules import HIMActorCritic
from deploy.s10_mujoco import observation, targets, load_model


SCHEDULE = [
    ('idle', 2, [0., 0., 0.]), ('forward', 5, [.8, 0., 0.]),
    ('stop_forward', 3, [0., 0., 0.]), ('reverse', 4, [-.5, 0., 0.]),
    ('stop_reverse', 3, [0., 0., 0.]), ('lateral', 4, [0., .3, 0.]),
    ('stop_lateral', 3, [0., 0., 0.]), ('yaw', 4, [0., 0., .6]),
    ('stop_yaw', 3, [0., 0., 0.]), ('turn', 5, [.6, 0., .5]),
    ('final_stop', 4, [0., 0., 0.]),
]


def restore_config(obj, values):
    for key, value in values.items():
        current = getattr(obj, key, None)
        if key in ('commands', 'scales') and isinstance(value, dict):
            # Saved sampling/reward recipes must not inherit new default features.
            setattr(obj, key, SimpleNamespace(**{
                name: SimpleNamespace(**item) if isinstance(item, dict) else item
                for name, item in value.items()}))
        elif isinstance(value, dict) and hasattr(current, '__dict__'):
            restore_config(current, value)
        else:
            setattr(obj, key, value)


def summarize(velocity, valid, dt):
    """Last second of each command; only samples before each robot's first fall."""
    result, start = [], 0
    for name, seconds, command in SCHEDULE:
        end = start + round(seconds / dt)
        window = slice(end - round(1. / dt), end)
        samples = velocity[window][valid[window]]
        result.append(dict(name=name, command=command, samples=len(samples),
                           mean=samples.mean(0).tolist() if len(samples) else None,
                           rmse=np.sqrt(np.square(samples - command).mean(0)).tolist() if len(samples) else None))
        start = end
    assert start == len(velocity)
    return result


def evaluate_gym(args, run, output):
    saved = json.loads((run / 'config.json').read_text())
    cfg, _ = task_registry.get_cfgs('s10')
    restore_config(cfg, saved['env'])
    assert class_to_dict(cfg) == saved['env']
    cfg.env.num_envs = args.num_envs or 64
    cfg.env.episode_length_s = sum(row[1] for row in SCHEDULE) + 1
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1e6
    for name in vars(cfg.commands.ranges):
        setattr(cfg.commands.ranges, name, [0., 0.])
    cfg.noise.add_noise = False
    cfg.noise.noise_level = 0.
    for name, value in vars(cfg.domain_rand).items():
        if isinstance(value, bool):
            setattr(cfg.domain_rand, name, False)
    # Existing resets still randomize joint position and root velocity.
    env, _ = task_registry.make_env('s10', args=args, env_cfg=cfg)
    checkpoint = torch.load(run / ('model_%d.pt' % args.checkpoint), map_location='cpu')
    assert checkpoint['iter'] == args.checkpoint
    assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
    actor = HIMActorCritic(env.num_obs, env.num_privileged_obs, env.num_one_step_obs,
                           env.num_actions, **saved['train']['policy']).to(env.device)
    actor.load_state_dict(checkpoint['model_state_dict'])
    actor.eval()
    env.reset()
    obs = env.get_observations()
    export_policy_as_jit(actor, str(output), env=env)
    exported = torch.jit.load(str(output / 'policy.pt')).eval()
    export_error = (actor.act_inference(obs)[0].cpu() - exported(obs[0].cpu())).abs().max().item()
    assert export_error < 1e-5, export_error
    rows = {key: [] for key in ['velocity', 'valid', 'height', 'dof_velocity', 'torque', 'wheel_contact', 'penalized_contact', 'reset']}
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    with torch.inference_mode():
        for name, seconds, command in SCHEDULE:
            for _ in range(round(seconds / env.dt)):
                env.commands[:, :3] = torch.tensor(command, device=env.device)
                obs[:, 6:9] = env.commands[:, :3] * env.commands_scale
                obs, _, _, _, _, reset_ids, _ = env.step(actor.act_inference(obs))
                assert not env.time_out_buf.any()
                alive[reset_ids] = False
                values = dict(velocity=torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), 1),
                              valid=alive, height=env.root_states[:, 2] - env.env_origins[:, 2],
                              dof_velocity=env.dof_vel, torque=env.torques,
                              wheel_contact=env.contact_forces[:, env.feet_indices, 2] > 1.,
                              penalized_contact=env.contact_forces[:, env.penalised_contact_indices].norm(dim=-1).gt(1.).any(1),
                              reset=env.reset_buf)
                for key, value in values.items():
                    rows[key].append(value.cpu().numpy().copy())
            print('GYM_SEGMENT', name, 'surviving', int(alive.sum()), flush=True)
    rows = {key: np.asarray(value) for key, value in rows.items()}
    assert all(np.isfinite(value).all() for value in rows.values())
    np.savez_compressed(output / 'gym_trace.npz', dt=env.dt, **rows)
    (output / 'evaluation_config.json').write_text(json.dumps(class_to_dict(cfg), indent=2))
    valid = rows['valid']
    summary = dict(environments=env.num_envs, dt=env.dt, seed=cfg.seed,
                   terrain='plane', policy='deterministic mean',
                   metric_mask='before first base-contact fall; reset and later samples excluded',
                   resets=int(rows['reset'].sum()), surviving=int(alive.sum()),
                   segments=summarize(rows['velocity'], valid, env.dt), export_max_error=export_error,
                   min_height=float(rows['height'][valid].min()) if valid.any() else None,
                   max_abs_joint_velocity=np.abs(rows['dof_velocity'][valid]).max(0).tolist() if valid.any() else None,
                   torque_saturation_fraction=(np.abs(rows['torque'][valid]) >= .99 * env.torque_limits.cpu().numpy()).mean(0).tolist() if valid.any() else None,
                   wheel_contact_fraction=rows['wheel_contact'][valid].mean(0).tolist() if valid.any() else None,
                   penalized_contact_fraction=float(rows['penalized_contact'][valid].mean()) if valid.any() else None)
    return summary, dict(iteration=checkpoint['iter'], timesteps=checkpoint['tot_timesteps'],
                         training_seconds=checkpoint['tot_time'], initialization=saved['initialization'])


def evaluate_mujoco(output):
    import mujoco
    cfg = json.loads((output / 'policy.json').read_text())
    policy = torch.jit.load(str(output / 'policy.pt'), map_location='cpu').eval()
    model, data, joints, motors = load_model(cfg)
    q_ids, v_ids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    base = model.body('base_link').id
    kp, kd, limits = map(np.asarray, (cfg['p_gains'], cfg['d_gains'], cfg['torque_limits']))
    dt = cfg['sim_dt'] * cfg['decimation']
    history, action = np.zeros((6, 57), np.float32), np.zeros(16, np.float32)
    rows = {key: [] for key in ['velocity', 'qpos', 'qvel', 'valid', 'base_contact']}
    velocity, force = np.empty(6), np.empty(6)
    alive = True
    with torch.inference_mode():
        for name, seconds, command in SCHEDULE:
            for _ in range(round(seconds / dt)):
                mujoco.mj_forward(model, data)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 1)
                gravity = data.xmat[base].reshape(3, 3).T @ np.array([0., 0., -1.])
                history[1:] = history[:-1].copy()
                history[0] = observation(cfg, velocity[:3], gravity, command, data.qpos[q_ids], data.qvel[v_ids], action)
                action = policy(torch.from_numpy(history.ravel())).numpy()
                assert np.isfinite(action).all()
                action = np.clip(action, -cfg['clip_actions'], cfg['clip_actions'])
                q_target, v_target = targets(cfg, action)
                base_contact = False
                for _ in range(cfg['decimation']):
                    data.ctrl[motors] = np.clip(kp * (q_target - data.qpos[q_ids]) + kd * (v_target - data.qvel[v_ids]), -limits, limits)
                    mujoco.mj_step(model, data)
                    for i in range(data.ncon):
                        contact = data.contact[i]
                        if base in model.geom_bodyid[[contact.geom1, contact.geom2]]:
                            mujoco.mj_contactForce(model, data, i, force)
                            base_contact |= np.linalg.norm(force[:3]) > 1.
                alive = alive and not base_contact
                mujoco.mj_forward(model, data)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 1)
                rows['velocity'].append(velocity[[3, 4, 2]].copy())
                rows['qpos'].append(data.qpos.copy())
                rows['qvel'].append(data.qvel.copy())
                rows['valid'].append(alive)
                rows['base_contact'].append(base_contact)
    rows = {key: np.asarray(value) for key, value in rows.items()}
    assert all(np.isfinite(value).all() for value in rows.values())
    np.savez_compressed(output / 'mujoco_trace.npz', dt=dt, **rows)
    falls = np.flatnonzero(rows['base_contact'])
    return dict(dt=dt, warnings=data.warning.number.tolist(), sim_seconds=data.time,
                first_base_contact_seconds=float((falls[0] + 1) * dt) if len(falls) else None,
                survived=bool(alive), reset=False,
                segments=summarize(rows['velocity'][:, None], rows['valid'][:, None], dt))


if __name__ == '__main__':
    args = get_args()
    assert args.task == 's10' and args.load_run and args.checkpoint is not None
    run = Path(LEGGED_GYM_ROOT_DIR) / 'logs/S10_HIM' / args.load_run
    output = run / ('evaluation_%d' % args.checkpoint)
    output.mkdir(exist_ok=False)
    gym, checkpoint = evaluate_gym(args, run, output)
    summary = dict(checkpoint=checkpoint, ppo_updates=0, schedule=SCHEDULE, gym=gym,
                   mujoco=evaluate_mujoco(output))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('EVALUATION_RESULT', json.dumps(summary, allow_nan=False), flush=True)
