"""Sample one frozen S10 policy under its saved training configuration; no updates.

python legged_gym/scripts/diagnose_s10.py --task s10 --headless \
    --load_run Sep15_08-33-31_scratch_1000_20260915 --checkpoint 900 --num_envs 64
python legged_gym/scripts/diagnose_s10.py --self-check
"""
import json
from pathlib import Path
import sys

import numpy as np


def reward_summary(parts, delivered, names, dt):
    total = parts.sum(-1)
    assert np.allclose(delivered, np.maximum(total, 0.), atol=1e-6)
    return dict(samples=int(total.size), negative_before_clip_fraction=float((total < 0).mean()),
                zero_delivered_fraction=float((delivered == 0).mean()),
                positive_samples=int((delivered > 0).sum()),
                before_clip_rate_quantiles=np.quantile(total / dt, [0, .1, .5, .9, 1]).tolist(),
                delivered_rate_mean=float(delivered.mean() / dt),
                component_rate_mean=dict(zip(names, (parts.mean((0, 1)) / dt).tolist())))


def main():
    import isaacgym  # Must precede torch.
    import torch
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.envs import S10RoughCfg  # Registers the tasks.
    from legged_gym.utils import get_args, task_registry
    from legged_gym.utils.helpers import class_to_dict
    from legged_gym.scripts.evaluate_s10 import restore_config
    from rsl_rl.modules import HIMActorCritic

    args = get_args()
    assert args.task == 's10' and args.load_run and args.checkpoint == 900
    run = Path(LEGGED_GYM_ROOT_DIR) / 'logs/S10_HIM' / args.load_run
    output = run / 'diagnosis_900'
    output.mkdir(exist_ok=False)
    saved = json.loads((run / 'config.json').read_text())
    cfg, _ = task_registry.get_cfgs('s10')
    restore_config(cfg, saved['env'])
    assert class_to_dict(cfg) == saved['env']
    cfg.env.num_envs = args.num_envs or 64
    checkpoint = torch.load(run / 'model_900.pt', map_location='cpu')
    assert checkpoint['iter'] == 900
    env, _ = task_registry.make_env('s10', args=args, env_cfg=cfg)
    actor = HIMActorCritic(env.num_obs, env.num_privileged_obs, env.num_one_step_obs,
                           env.num_actions, **saved['train']['policy']).to(env.device).eval()
    actor.load_state_dict(checkpoint['model_state_dict'])
    env.reset()
    obs = env.get_observations()
    names = env.reward_names
    assert 'termination' not in env.reward_scales and cfg.rewards.only_positive_rewards
    rows = {key: [] for key in ['reward_parts', 'delivered', 'torque', 'dof_velocity',
                                'mean', 'action', 'action_delta', 'height', 'tilt',
                                'fall', 'timeout', 'commands', 'terrain_level']}
    current = {}

    # Observe existing calls once, before reset can replace the sampled state.
    for i, (name, original) in enumerate(zip(names, env.reward_functions)):
        def observe_component(original=original, name=name):
            value = original()
            current[name] = (value * env.reward_scales[name]).cpu().numpy().copy()
            return value
        env.reward_functions[i] = observe_component

    original_reward = env.compute_reward
    def observe_reward():
        original_reward()
        rows['reward_parts'].append(np.stack([current[name] for name in names], -1))
        rows['delivered'].append(env.rew_buf.cpu().numpy().copy())
        values = dict(height=env.root_states[:, 2] - env.measured_heights.mean(1),
                      tilt=torch.acos((-env.projected_gravity[:, 2]).clamp(-1., 1.)),
                      fall=env.reset_buf & ~env.time_out_buf, timeout=env.time_out_buf,
                      commands=env.commands[:, :3], terrain_level=env.terrain_levels)
        for key, value in values.items():
            rows[key].append(value.cpu().numpy().copy())
    env.compute_reward = observe_reward

    original_torques = env._compute_torques
    def observe_torques(action):
        torque = original_torques(action)
        rows['torque'].append(torque.cpu().numpy().copy())
        rows['dof_velocity'].append(env.dof_vel.cpu().numpy().copy())
        return torque
    env._compute_torques = observe_torques

    steps = round(20. / env.dt)
    with torch.inference_mode():
        for step in range(steps):
            action = actor.act(obs)
            rows['mean'].append(actor.action_mean.cpu().numpy().copy())
            rows['action'].append(action.cpu().numpy().copy())
            rows['action_delta'].append((action - env.last_actions).cpu().numpy().copy())
            obs, _, _, _, _, _, _ = env.step(action)
            if (step + 1) % 250 == 0:
                print('SAMPLED_STEPS', step + 1, flush=True)
    rows = {key: np.asarray(value) for key, value in rows.items()}
    assert all(np.isfinite(value).all() for value in rows.values())
    assert len(rows['torque']) == steps * cfg.control.decimation
    assert all(torch.equal(value.cpu(), checkpoint['model_state_dict'][key])
               for key, value in actor.state_dict().items())
    groups = {joint: [i for i, name in enumerate(env.dof_names) if f'_{joint}_' in name]
              for joint in ['hipx', 'hipy', 'knee', 'wheel']}
    std = actor.std.detach().cpu().numpy()
    scale = env.action_scale.cpu().numpy().copy()
    scale[env.wheel_indices.cpu().numpy()] = cfg.control.vel_scale
    torque_limits = env.torque_limits.cpu().numpy()
    summary = dict(checkpoint=900, ppo_updates=0, model_unchanged=True,
                   num_envs=env.num_envs, policy_steps=steps, seconds_per_env=steps * env.dt,
                   dt=env.dt, pd_dt=cfg.sim.dt, reward_names=names, dof_names=env.dof_names,
                   sampling='Saved training configuration, stochastic policy, fresh simulator and curriculum state; only num_envs changed.',
                   rewards=reward_summary(rows['reward_parts'], rows['delivered'], names, env.dt),
                   falls=int(rows['fall'].sum()), timeouts=int(rows['timeout'].sum()),
                   groups={},
                   raw_action_clip_fraction=float((np.abs(rows['action']) >= cfg.normalization.clip_actions).mean()))
    for name, ids in groups.items():
        summary['groups'][name] = dict(
            indices=ids, raw_std=std[ids].tolist(), target_std=(std[ids] * scale[ids]).tolist(),
            target_unit='rad/s' if name == 'wheel' else 'rad',
            abs_mean_action_quantiles=np.quantile(np.abs(rows['mean'][:, :, ids]), [.5, .9, .99, 1]).tolist(),
            abs_target_delta_quantiles=np.quantile(np.abs(rows['action_delta'][:, :, ids] * scale[ids]), [.5, .9, .99, 1]).tolist(),
            pd_torque_saturation_fraction=(np.abs(rows['torque'][:, :, ids]) >= .99 * torque_limits[ids]).mean((0, 1)).tolist(),
            pd_torque_rms=np.sqrt(np.square(rows['torque'][:, :, ids]).mean((0, 1))).tolist(),
            pd_abs_velocity_p99=np.quantile(np.abs(rows['dof_velocity'][:, :, ids]), .99, axis=(0, 1)).tolist())
    # Separate initial transients from the latter half of the same rollout.
    summary['last_10_seconds_rewards'] = reward_summary(
        rows['reward_parts'][steps // 2:], rows['delivered'][steps // 2:], names, env.dt)
    np.savez_compressed(output / 'trace.npz', **rows)
    (output / 'config.json').write_text(json.dumps(class_to_dict(cfg), indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('DIAGNOSIS_RESULT', json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        parts = np.array([[[1., -2.], [2., -1.], [0., 0.]]])
        result = reward_summary(parts, np.array([[0., 1., 0.]]), ['positive', 'negative'], .5)
        assert result['samples'] == 3 and result['positive_samples'] == 1
        assert result['negative_before_clip_fraction'] == 1 / 3
        assert result['zero_delivered_fraction'] == 2 / 3
        assert result['component_rate_mean'] == {'positive': 2., 'negative': -2.}
        try:
            reward_summary(parts, np.ones((1, 3)), ['positive', 'negative'], .5)
        except AssertionError:
            pass
        else:
            raise AssertionError('Incorrect delivered rewards were accepted')
        print('reward accounting self-check passed')
    else:
        main()
