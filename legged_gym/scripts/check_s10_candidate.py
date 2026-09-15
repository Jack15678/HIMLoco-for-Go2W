"""Single authorized S10 candidate: zero-PPO gate, never trains.

python legged_gym/scripts/check_s10_candidate.py --self-check
python legged_gym/scripts/check_s10_candidate.py --task s10 --headless --num_envs 64
"""
import copy
import json
from pathlib import Path
import sys

import isaacgym  # Before torch.
import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import S10RoughCfg, S10RoughCfgPPO
from legged_gym.utils import get_args, task_registry, export_policy_as_jit
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.scripts.diagnose_s10 import reward_summary
from rsl_rl.modules import HIMActorCritic
from rsl_rl.runners import HIMOnPolicyRunner


ROOT = Path(LEGGED_GYM_ROOT_DIR)
RUN = ROOT / 'logs/S10_HIM/candidate_2p5ms_std03_06_20260915'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def gate_passes(positive_fraction, saturation, leg_ids):
    return bool(positive_fraction >= .1 and np.max(np.asarray(saturation)[leg_ids]) <= .1)


def self_check():
    from types import SimpleNamespace as NS
    from unittest.mock import Mock, patch
    from legged_gym.envs.base.legged_robot import LeggedRobot
    from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg
    # Exercise the actual step method, with each possible delay sampled once.
    for cfg, stride in [(GO2WRoughCfg(), 1), (S10RoughCfg(), 2)]:
        seen = []
        env = NS(cfg=cfg, device='cpu', num_envs=4, num_actions=16,
                 last_actions=torch.zeros(4, 16), torques=torch.zeros(4, 16),
                 render=lambda: None, gym=Mock(), sim=None,
                 post_physics_step=lambda: ([], None), obs_buf=torch.zeros(4, 342),
                 privileged_obs_buf=None, rew_buf=None, reset_buf=None, extras={})
        def torque(action):
            seen.append(action.clone())
            return torch.zeros_like(action)
        env._compute_torques = torque
        with patch('torch.randint', return_value=torch.arange(4).view(4, 1)) as draw:
            LeggedRobot.step(env, torch.ones(4, 16))
        assert draw.call_args.args[:2] == (0, 4)
        assert len(seen) == cfg.control.decimation
        actual = torch.stack(seen)[:, :, 0]
        expected = torch.arange(cfg.control.decimation)[:, None] >= torch.arange(4) * stride
        assert torch.equal(actual, expected.float())
        assert np.allclose(np.arange(4) * stride * cfg.sim.dt, [0, .005, .01, .015])
    for std in [1., [.3, .3, .3, .6] * 4]:
        model = HIMActorCritic(342, 262, 57, 16, init_noise_std=std)
        assert torch.equal(model.std, torch.as_tensor(std) * torch.ones(16))
    for std in [[.3], [-.1] * 16, float('nan')]:
        try:
            HIMActorCritic(342, 262, 57, 16, init_noise_std=std)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid standard deviation accepted')
    assert gate_passes(.1, np.full(16, .1), list(range(12)))
    assert json.loads(json.dumps({'passed': gate_passes(.1, np.full(16, .1), list(range(12)))}))['passed'] is True
    assert not gate_passes(.099, np.zeros(16), list(range(12)))
    assert not gate_passes(.5, np.full(16, .101), list(range(12)))
    print('SELF_CHECK_PASS: real S10/Go2W delay path, scalar/vector std, gate boundaries')


def main():
    args = get_args()
    assert args.task == 's10' and args.num_envs == 64 and not args.resume
    RUN.mkdir(parents=True, exist_ok=False)
    output = RUN / 'zero_ppo'
    output.mkdir()
    cfg, train_cfg = task_registry.get_cfgs('s10')
    train_cfg.runner.max_iterations = 200
    train_cfg.runner.run_name = RUN.name
    env, _ = task_registry.make_env('s10', args=args, env_cfg=cfg)
    # Explicit model seed makes initialization independent of environment count.
    set_seed(train_cfg.seed)
    runner = HIMOnPolicyRunner(env, class_to_dict(train_cfg), str(RUN), device=args.rl_device)
    actor = runner.alg.actor_critic.eval()
    runner.save(str(RUN / 'model_0.pt'))
    checkpoint = torch.load(RUN / 'model_0.pt', map_location='cpu')
    assert checkpoint['iter'] == checkpoint['tot_timesteps'] == checkpoint['tot_time'] == 0
    assert not checkpoint['optimizer_state_dict']['state']
    assert not checkpoint['estimator_optimizer_state_dict']['state']
    with torch.random.fork_rng():
        torch.manual_seed(train_cfg.seed)
        reference = HIMActorCritic(env.num_obs, env.num_privileged_obs, env.num_one_step_obs,
                                  env.num_actions, **class_to_dict(train_cfg.policy))
    assert all(torch.equal(value, reference.state_dict()[key])
               for key, value in checkpoint['model_state_dict'].items())
    del reference
    expected_names = [f'{leg}_{joint}_joint' for leg in ['fl', 'fr', 'hl', 'hr']
                      for joint in ['hipx', 'hipy', 'knee', 'wheel']]
    assert env.dof_names == expected_names
    assert torch.equal(actor.std.cpu(), torch.tensor([.3, .3, .3, .6] * 4))
    saved_env = class_to_dict(cfg)
    saved_env['env']['num_envs'] = 4096
    baseline = json.loads((ROOT / 'docs/evidence/s10-scratch900-20260915/config.json').read_text())
    expected_env = copy.deepcopy(baseline['env'])
    expected_env['sim']['dt'] = .0025
    expected_env['control'].update(decimation=8, delay_stride=2)
    assert saved_env == expected_env, 'Unexpected environment recipe change'
    assert class_to_dict(train_cfg.algorithm) == baseline['train']['algorithm']
    expected_policy = copy.deepcopy(baseline['train']['policy'])
    expected_policy['init_noise_std'] = [.3, .3, .3, .6] * 4
    assert class_to_dict(train_cfg.policy) == expected_policy
    write_json(RUN / 'config.json', dict(env=saved_env, train=class_to_dict(train_cfg),
               initialization='scratch', interface_version=2, dof_names=env.dof_names))
    write_json(output / 'config.json', class_to_dict(cfg))
    export_policy_as_jit(actor, str(output), env=env)
    metadata = json.loads((output / 'policy.json').read_text())
    from deploy.s10_mujoco import load_model
    model, data, joints, motors = load_model(metadata)
    assert np.isclose(model.opt.timestep, .0025) and metadata['decimation'] == 8
    assert np.isclose(env.dt, .02) and np.isclose(model.opt.timestep * metadata['decimation'], env.dt)
    for profile in ['gym-air', 'gym-stand']:
        prior = json.loads((ROOT / f'docs/evidence/s10-pd-followup/{profile}-2p5ms-config.json').read_text())
        assert saved_env['asset'] == prior['asset']
        assert {k: v for k, v in saved_env['control'].items() if k != 'delay_stride'} == prior['control']
    prior = json.loads((ROOT / 'docs/evidence/s10-pd-followup/mujoco-air-2p5ms-config.json').read_text())
    assert set(metadata) == set(prior)
    assert all(np.allclose(metadata[k], prior[k]) if k == 'sim_dt' else metadata[k] == prior[k] for k in metadata)
    obs = env.get_observations()
    exported = torch.jit.load(str(output / 'policy.pt')).eval()
    with torch.no_grad():
        export_error = float((actor.act_inference(obs)[0].cpu() - exported(obs[0].cpu())).abs().max())
    assert export_error < 1e-5
    write_json(output / 'initialization.json', dict(
        seed=train_cfg.seed, all_model_tensors_equal_fresh_constructor=True,
        networks=['actor', 'critic', 'estimator (encoder, target, prototypes)'],
        ppo_optimizer_state_entries=0, estimator_optimizer_state_entries=0,
        ppo_iterations=0, initial_std=actor.std.tolist(), export_max_error=export_error,
        reused_pd_evidence='docs/evidence/s10-pd-followup/timestep_comparison.json',
        pd_asset_control_match=True, unchanged_algorithm_and_remaining_env=True,
        runtime_pd_dt=env.sim_params.dt, mujoco_pd_dt=model.opt.timestep,
        runtime_policy_dt=env.dt, history_dt=env.dt, delay_ms=[0, 5, 10, 15]))

    names, current = env.reward_names, {}
    rows = {k: [] for k in ['reward_parts', 'delivered', 'torque', 'dof_velocity', 'base_contact_force',
                            'pd_action', 'action', 'mean', 'action_delta', 'fall', 'height', 'tilt']}
    for i, (name, original) in enumerate(zip(names, env.reward_functions)):
        def component(original=original, name=name):
            value = original()
            current[name] = (value * env.reward_scales[name]).cpu().numpy().copy()
            return value
        env.reward_functions[i] = component
    original_reward = env.compute_reward
    def reward():
        original_reward()
        rows['reward_parts'].append(np.stack([current[name] for name in names], -1))
        for key, value in dict(delivered=env.rew_buf, fall=env.reset_buf & ~env.time_out_buf,
                               height=env.root_states[:, 2] - env.measured_heights.mean(1),
                               tilt=torch.acos((-env.projected_gravity[:, 2]).clamp(-1., 1.))).items():
            rows[key].append(value.cpu().numpy().copy())
    env.compute_reward = reward
    original_torque = env._compute_torques
    def torque(action):
        result = original_torque(action)
        rows['torque'].append(result.cpu().numpy().copy())
        rows['pd_action'].append(action.cpu().numpy().copy())
        assert torch.isfinite(result).all(), 'Non-finite PD torque'
        return result
    env._compute_torques = torque
    original_gym = env.gym
    # Read each completed PD substep before the original reset path can erase a fall.
    class ObservedGym:
        def __getattr__(self, name):
            return getattr(original_gym, name)
        def simulate(self, sim):
            original_gym.simulate(sim)
            original_gym.refresh_dof_state_tensor(sim)
            original_gym.refresh_net_contact_force_tensor(sim)
            rows['dof_velocity'].append(env.dof_vel.cpu().numpy().copy())
            rows['base_contact_force'].append(env.contact_forces[:, env.termination_contact_indices].cpu().numpy().copy())
    env.gym = ObservedGym()
    sim_start = env.gym.get_sim_time(env.sim)
    delay_counts = np.zeros(8, dtype=np.int64)
    steps = round(20. / env.dt)
    with torch.inference_mode():
        for step in range(steps):
            previous_obs = obs.clone()
            previous_action = env.last_actions.clone()
            action = actor.act(obs)
            for key, value in dict(action=action, mean=actor.action_mean, action_delta=action-previous_action).items():
                rows[key].append(value.cpu().numpy().copy())
            obs, critic, rew, done, info, reset_ids, _ = env.step(action)
            assert all(torch.isfinite(v).all() for v in [obs, critic, rew, env.dof_pos, env.dof_vel, env.root_states])
            continuous = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            continuous[reset_ids] = False
            assert torch.equal(obs[continuous, 57:], previous_obs[continuous, :-57])
            # First substep that differs from the preceding policy action.
            delays = (env.delayed_actions - previous_action[:, None]).abs().amax(-1).gt(1e-6).int().argmax(1)
            assert all(v in [0, 2, 4, 6] for v in delays.tolist())
            delay_counts += np.bincount(delays.cpu().numpy(), minlength=8)
            if (step + 1) % 250 == 0:
                print('SAMPLED_STEPS', step + 1, flush=True)
    elapsed = env.gym.get_sim_time(env.sim) - sim_start
    assert abs(elapsed - steps * env.dt) < 1e-5
    assert len(rows['torque']) == steps * cfg.control.decimation
    assert all(torch.equal(v.cpu(), checkpoint['model_state_dict'][k]) for k, v in actor.state_dict().items())
    assert not runner.alg.optimizer.state and not actor.estimator.optimizer.state
    rows = {key: np.asarray(value) for key, value in rows.items()}
    np.savez_compressed(output / 'trace.npz', **rows)
    assert all(np.isfinite(value).all() for value in rows.values())
    summary = reward_summary(rows['reward_parts'], rows['delivered'], names, env.dt)
    summary['delivered_rate_quantiles'] = np.quantile(rows['delivered']/env.dt, [0, .1, .5, .9, 1]).tolist()
    limits = env.torque_limits.cpu().numpy()
    saturation = (np.abs(rows['torque']) >= .99 * limits).mean((0, 1))
    groups = {joint: [i for i, name in enumerate(env.dof_names) if f'_{joint}_' in name]
              for joint in ['hipx', 'hipy', 'knee', 'wheel']}
    scale = env.action_scale.cpu().numpy().copy()
    scale[groups['wheel']] = cfg.control.vel_scale
    group_stats = {}
    for name, ids in groups.items():
        group_stats[name] = dict(indices=ids, target_unit='rad/s' if name == 'wheel' else 'rad',
            target_std=(actor.std.detach().cpu().numpy()[ids] * scale[ids]).tolist(),
            abs_target_delta_quantiles=np.quantile(np.abs(rows['action_delta'][:, :, ids] * scale[ids]), [.5, .9, .99, 1]).tolist(),
            pd_saturation_fraction=saturation[ids].tolist(),
            pd_torque_rms=np.sqrt(np.square(rows['torque'][:, :, ids]).mean((0, 1))).tolist())
    positive_fraction = summary['positive_samples'] / summary['samples']
    leg_ids = groups['hipx'] + groups['hipy'] + groups['knee']
    result = dict(passed=gate_passes(positive_fraction, saturation, leg_ids), ppo_iterations=0,
        num_envs=64, policy_steps=steps, pd_steps=steps*8, actual_sim_seconds=elapsed,
        positive_fraction=positive_fraction, max_leg_saturation=float(saturation[leg_ids].max()),
        required_positive_fraction=.1, maximum_leg_saturation=.1,
        rewards=summary, groups=group_stats, dof_names=env.dof_names,
        delay_substep_counts=delay_counts.tolist(), history_shift_checks=steps,
        base_contact_pd_fraction=float((np.linalg.norm(rows['base_contact_force'], axis=-1) > 1).any(-1).mean()),
        falls=int(rows['fall'].sum()), finite=True, model_unchanged=True, optimizers_unchanged=True)
    write_json(output / 'summary.json', result)
    print('CANDIDATE_GATE_RESULT', json.dumps(result), flush=True)
    if not result['passed']:
        print('STOP: candidate gate failed; no PPO training authorized for this result.', flush=True)


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        self_check()
    else:
        main()
