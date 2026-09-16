"""Direct-command checks; CPU sampler check or real Gym check from the rollout gate."""
import ast
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def check_commands(env):
    """Use real reset, callback and observation methods without optimizer updates."""
    assert not env.cfg.commands.heading_command
    ids = torch.arange(env.num_envs, device=env.device)
    sampled = []
    original = env._resample_commands

    def resample(env_ids):
        original(env_ids)
        if len(env_ids):
            sampled.append(env.commands[env_ids, :3].clone())

    env._resample_commands = resample
    try:
        env.reset_idx(ids)
        assert torch.equal(env.commands[:, :3], sampled[-1]), 'Reset overwrote sampled yaw rate'
        yaw = env.commands[:, 2]
        assert (yaw > .5).any() and (yaw < -.5).any()
        assert yaw.min() >= -1 and yaw.max() <= 1
        commands = torch.tensor([[0., 0., .8], [0., 0., -.8], [.5, 0., 0.],
                                 [.6, .2, .6], [.6, -.2, -.6], [0., 0., 0.]], device=env.device)
        wanted = commands[ids % len(commands)]
        env.commands[:, :3] = wanted
        env.commands[:, 3] = 2.9  # Stale heading must not affect direct yaw commands.
        env.common_step_counter = 1  # No random pushes during this command plumbing check.
        before = len(sampled)
        observation_error = 0.
        for i, heading in enumerate([-3., -1., 0., 1., 3.]):
            env.episode_length_buf[:] = i + 1  # Between ordinary 10 s resampling boundaries.
            env.base_quat[:] = 0
            env.base_quat[:, 2] = torch.sin(torch.tensor(heading / 2))
            env.base_quat[:, 3] = torch.cos(torch.tensor(heading / 2))
            old = env.obs_buf.clone()
            env._post_physics_step_callback()
            env.compute_observations()
            assert torch.equal(env.commands[:, :3], wanted)
            assert torch.allclose(env.obs_buf[:, 6:9] / env.commands_scale, wanted, atol=1e-7, rtol=0)
            observation_error = max(observation_error, float((env.obs_buf[:, 6:9] / env.commands_scale - wanted).abs().max()))
            assert torch.equal(env.obs_buf[:, 57:], old[:, :-57]), 'History shifted incorrectly'
        assert len(sampled) == before
        env.episode_length_buf[:] = int(env.cfg.commands.resampling_time / env.dt)
        env._post_physics_step_callback()
        assert len(sampled) == before + 1
        assert torch.equal(env.commands[:, :3], sampled[-1]), 'Callback overwrote resampled yaw rate'
        return dict(status='passed', commands=commands.tolist(), headings=[-3., -1., 0., 1., 3.],
                    reset_and_callback_preserve_sampled_yaw=True, observation_max_error=observation_error,
                    history_shift_frames=1, policy_dt=env.dt, resampling_time=env.cfg.commands.resampling_time)
    finally:
        env._resample_commands = original


def cpu_check():
    # Execute the repository's sampler/callback bodies; no Isaac Gym installation required.
    source = Path(__file__).parents[1] / 'envs/base/legged_robot.py'
    robot = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                 if isinstance(n, ast.ClassDef) and n.name == 'LeggedRobot')
    scope = dict(torch=torch, torch_rand_float=lambda low, high, shape, device:
                 low + (high - low) * torch.rand(shape, device=device))
    for name in ['_resample_commands', '_post_physics_step_callback']:
        method = next(n for n in robot.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    env = SimpleNamespace(num_envs=4096, device='cpu', dt=.02, commands=torch.zeros(4096, 4),
        parking_commands=torch.zeros(4096, dtype=torch.bool),
        cfg=SimpleNamespace(commands=SimpleNamespace(heading_command=False, resampling_time=10.),
                            terrain=SimpleNamespace(measure_heights=False),
                            domain_rand=SimpleNamespace(push_robots=False, disturbance=False)),
        command_ranges=dict(lin_vel_x=[-1., 1.], lin_vel_y=[-.6, .6], ang_vel_yaw=[-1., 1.]),
        episode_length_buf=torch.ones(4096, dtype=torch.long))
    env._resample_commands = lambda ids: scope['_resample_commands'](env, ids)
    torch.manual_seed(1)
    env._resample_commands(torch.arange(4096))
    initial = env.commands.clone()
    assert (initial[:, 2] > .8).any() and (initial[:, 2] < -.8).any()
    assert initial[:, 2].abs().max() <= 1
    scope['_post_physics_step_callback'](env)
    assert torch.equal(env.commands, initial)
    env.episode_length_buf[:] = 500
    scope['_post_physics_step_callback'](env)
    assert not torch.equal(env.commands[:, 2], initial[:, 2])
    print('PASS: actual sampler and callback preserve direct yaw rates between 10 s resamples; 0 updates')


def cpu_restore(evidence):
    from unittest.mock import patch
    from rsl_rl.runners import HIMOnPolicyRunner
    # Reuse the existing recursive equality check without importing GPU-only dependencies.
    path = Path(__file__).with_name('resume_s10_candidate.py')
    method = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'equal_state')
    scope = dict(torch=torch)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    equal_state = scope['equal_state']
    saved = json.loads((evidence / 'config.json').read_text())
    checkpoint = evidence / 'model_1500.pt'
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    env = SimpleNamespace(num_envs=1, num_obs=342, num_privileged_obs=262,
                          num_one_step_obs=57, num_actions=16, reset=lambda: (None, None))
    runner = HIMOnPolicyRunner(env, saved['train'], device='cpu')
    # Only map checkpoint storage to CPU; execute the unmodified production load method.
    with patch('torch.load', return_value=source) as read:
        runner.load(str(checkpoint), load_optimizer=True)
        read.assert_called_once_with(str(checkpoint))
    equal_state(runner.alg.actor_critic.state_dict(), source['model_state_dict'])
    for optimizer, key in [(runner.alg.optimizer, 'optimizer_state_dict'),
                           (runner.alg.actor_critic.estimator.optimizer, 'estimator_optimizer_state_dict')]:
        equal_state(optimizer.state_dict(), source[key])
        assert {float(s['step']) for s in optimizer.state.values()} == {30000.}
    assert runner.current_learning_iteration == 1500 and runner.tot_timesteps == 294912000
    assert runner.alg.learning_rate == source['optimizer_state_dict']['param_groups'][0]['lr']
    assert runner.alg.actor_critic.estimator.learning_rate == source['estimator_optimizer_state_dict']['param_groups'][0]['lr']
    result = dict(status='passed', checkpoint=str(checkpoint), iteration=1500, environment_steps=294912000,
                  adam_steps=[30000, 30000], all_model_and_optimizer_tensors_equal=True, updates=0,
                  learning_rates=[runner.alg.learning_rate, runner.alg.actor_critic.estimator.learning_rate],
                  std=runner.alg.actor_critic.std.tolist(), model_state_tensors=len(source['model_state_dict']))
    print('CPU_RESTORE ' + json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cpu_check()
    if args.evidence:
        result = cpu_restore(args.evidence)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
