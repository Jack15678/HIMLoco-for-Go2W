"""Resume the verified model_200 once; review at 500 in the same 800-update loop."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import time
import traceback

import isaacgym
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import S10RoughCfg
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.scripts.evaluate_s10 import restore_config
from rsl_rl.runners import HIMOnPolicyRunner

ROOT = Path(LEGGED_GYM_ROOT_DIR)
RUN = ROOT / 'logs/S10_HIM/resume_200_to1000_20260915'
SOURCE = ROOT / 'artifacts/s10-resume-input/model_200.pt'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def require_finite(values, where):
    if not torch.stack([torch.isfinite(v).all().to(values[0].device) for v in values]).all().item():
        raise FloatingPointError('NaN/Inf in ' + where)


def equal_state(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal_state(a, b)
    else:
        assert actual == expected


def main():
    args = get_args()
    assert args.task == 's10' and args.num_envs == 4096 and args.max_iterations == 800
    assert args.resume
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Server checkout must be clean'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / 'source_commit.txt').write_text(commit + '\n')
    write_json(RUN / 'training_started.json', dict(start=200, review=500, target=1000,
        additional_budget=800, source_checkpoint=str(SOURCE), source_commit=commit))
    saved = json.loads((ROOT / 'legged_gym/scripts/configs/s10_resume_200.json').read_text())
    cfg, _ = task_registry.get_cfgs('s10')
    restore_config(cfg, saved['env'])
    train = copy.deepcopy(saved['train'])
    train['runner'].update(max_iterations=1000, run_name=RUN.name, resume=True,
                           resume_path=str(SOURCE), checkpoint=200)
    env, _ = task_registry.make_env('s10', args=args, env_cfg=cfg)
    assert class_to_dict(cfg) == saved['env']
    set_seed(train['seed'])
    runner = HIMOnPolicyRunner(env, train, str(RUN), device=args.rl_device)
    runner.load(str(SOURCE), load_optimizer=True)
    actor = runner.alg.actor_critic
    source = torch.load(SOURCE, map_location='cpu')
    equal_state(actor.state_dict(), source['model_state_dict'])
    optimizers = dict(ppo_optimizer_steps=runner.alg.optimizer,
                      estimator_optimizer_steps=actor.estimator.optimizer)
    for optimizer, key in zip(optimizers.values(), ['optimizer_state_dict', 'estimator_optimizer_state_dict']):
        equal_state(optimizer.state_dict(), source[key])
        assert {float(s['step']) for s in optimizer.state.values()} == {4000.}
        assert optimizer.param_groups[0]['lr'] == 1e-5
    assert runner.alg.learning_rate == actor.estimator.learning_rate == 1e-5
    assert runner.current_learning_iteration == 200 and runner.tot_timesteps == 39321600
    assert runner.num_steps_per_env == 48 and runner.save_interval == 50
    write_json(RUN / 'config.json', dict(env=class_to_dict(cfg), train=train,
        initialization='full model_200 + both Adam optimizers', interface_version=2, dof_names=env.dof_names))

    def curriculum():
        return dict(terrain_levels=env.terrain_levels.cpu().clone(),
            terrain_types=env.terrain_types.cpu().clone(), env_origins=env.env_origins.cpu().clone(),
            command_ranges=copy.deepcopy(env.command_ranges), common_step_counter=env.common_step_counter)

    write_json(RUN / 'resume_verification.json', dict(iteration=200, historical_environment_steps=39321600,
        all_model_and_optimizer_tensors_equal_source=True, ppo_optimizer_steps=4000,
        estimator_optimizer_steps=4000, learning_rate=runner.alg.learning_rate,
        estimator_learning_rate=actor.estimator.learning_rate, restored_std=actor.std.tolist(),
        actual_pd_dt=env.sim_params.dt, actual_policy_dt=env.dt, delay_substeps=[0, 2, 4, 6],
        environment_handling='Fresh simulator, domain randomization and terrain generated with saved seed/config; source checkpoint has no physical or curriculum state. Fresh episode lengths randomized once. No physical-step seamless resume claimed.',
        initial_terrain_mean=float(env.terrain_levels.float().mean()),
        initial_terrain_histogram=torch.bincount(env.terrain_levels, minlength=10).tolist(),
        initial_command_ranges=env.command_ranges, review_handling='Same learn(800) call and runner stay alive at 500; no reset/reseed at review.'))
    counters = {key: 4000 for key in optimizers}
    for key, optimizer in optimizers.items():
        original = optimizer.step
        params = [p for group in optimizer.param_groups for p in group['params']]
        def checked_step(*a, original=original, params=params, key=key, **kw):
            require_finite([p.grad for p in params if p.grad is not None], key + ' gradients')
            result = original(*a, **kw)
            counters[key] += 1
            require_finite(params, key + ' parameters')
            if not (actor.std > 0).all().item():
                raise FloatingPointError('Non-positive exploration std')
            return result
        optimizer.step = checked_step
    original_torque, original_step = env._compute_torques, env.step
    def torque(action):
        value = original_torque(action)
        require_finite([value, env.dof_pos, env.dof_vel], 'PD substep')
        return value
    def step(action):
        require_finite([action], 'policy action')
        result = original_step(action)
        require_finite([result[0], result[1], result[2], env.root_states], 'rollout')
        return result
    env._compute_torques, env.step = torque, step
    original_save, original_log = runner.save, runner.log
    def save(path, infos=None):
        require_finite(list(actor.state_dict().values()), 'checkpoint model')
        for optimizer in optimizers.values():
            require_finite([t for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t)], 'Adam state')
            assert {float(s['step']) for s in optimizer.state.values()} == {20.*runner.current_learning_iteration}
        assert runner.tot_timesteps == runner.current_learning_iteration*4096*48
        original_save(path, infos=dict(curriculum=curriculum()))
        if runner.current_learning_iteration == 500:
            write_json(RUN / 'review_ready.json', dict(iteration=500, same_runner_alive=True,
                environment_steps=runner.tot_timesteps, **counters))
            print('REVIEW_READY_500', flush=True)
            decision_path = RUN / 'review_decision.json'
            while not decision_path.exists():
                time.sleep(2)
            decision = json.loads(decision_path.read_text())
            assert decision['iteration'] == 500
            if decision['action'] != 'continue':
                raise RuntimeError('Stopped by 500 review: ' + decision['reason'])
            write_json(RUN / 'review_resumed.json', dict(iteration=500,
                common_step_counter=env.common_step_counter, same_runner=True, decision=decision))
    runner.save = save
    with (RUN / 'metrics.jsonl').open('x') as metrics:
        def log(locs):
            original_log(locs)
            values = dict(iteration=locs['it']+1,
                mean_reward=sum(locs['rewbuffer'])/len(locs['rewbuffer']) if locs['rewbuffer'] else None,
                mean_episode_length_steps=sum(locs['lenbuffer'])/len(locs['lenbuffer']) if locs['lenbuffer'] else None,
                std=actor.std.tolist(), learning_rate=runner.alg.learning_rate,
                estimator_learning_rate=actor.estimator.learning_rate, **runner.alg.update_stats,
                losses={k: locs[k] for k in ['mean_value_loss', 'mean_surrogate_loss', 'mean_estimation_loss', 'mean_swap_loss']},
                reward_components={k: sum(float(row[k]) for row in locs['ep_infos'])/len(locs['ep_infos'])
                    for k in locs['ep_infos'][0]} if locs['ep_infos'] else {},
                terrain_level_mean=float(env.terrain_levels.float().mean()),
                terrain_histogram=torch.bincount(env.terrain_levels, minlength=10).tolist(),
                command_ranges=copy.deepcopy(env.command_ranges),
                total_timesteps=runner.tot_timesteps, total_seconds=runner.tot_time,
                nonfinite_count=0, **counters)
            metrics.write(json.dumps(values, allow_nan=False)+'\n')
            metrics.flush()
        runner.log = log
        try:
            save(str(RUN / 'model_200.pt'))
            print('RESUME_VERIFIED_ZERO_PPO', flush=True)
            runner.learn(num_learning_iterations=800, init_at_random_ep_len=True)
            assert runner.current_learning_iteration == 1000 and runner.tot_timesteps == 196608000
            assert set(counters.values()) == {20000}
            write_json(RUN / 'training_result.json', dict(status='completed', ppo_iterations=1000,
                additional_ppo_iterations=800, environment_steps=runner.tot_timesteps,
                additional_environment_steps=157286400, nonfinite_count=0, **counters))
        except BaseException:
            original_save(str(RUN / 'failure_state.pt'), infos=dict(curriculum=curriculum()))
            torch.save(dict(root=env.root_states, dof=env.dof_state, obs=env.obs_buf,
                torques=env.torques), RUN / 'failure_sim_state.pt')
            write_json(RUN / 'training_result.json', dict(status='failed',
                ppo_iterations=runner.current_learning_iteration, error=traceback.format_exc(), **counters))
            raise
        finally:
            if runner.writer:
                runner.writer.close()


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        require_finite([torch.zeros(2, device='cuda'), torch.tensor(4000.)], 'mixed-device Adam')
        equal_state({'a': [torch.ones(2, device='cuda'), 3]}, {'a': [torch.ones(2), 3]})
        for bad in [float('nan'), float('inf')]:
            try:
                require_finite([torch.zeros(2, device='cuda'), torch.tensor(bad)], 'mixed-device Adam')
            except FloatingPointError:
                pass
            else:
                raise AssertionError('Non-finite value accepted')
        print('PASS: mixed CPU/CUDA optimizer finite check and recursive state comparison')
    else:
        main()
