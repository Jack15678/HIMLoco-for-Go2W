"""One authorized direct-velocity experiment: full model_1500 restore, learn(500)."""
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

import isaacgym  # Before torch.
import torch
from legged_gym.envs import S10RoughCfg
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.scripts.evaluate_s10 import restore_config
from legged_gym.scripts.resume_s10_candidate import equal_state, require_finite, write_json
from rsl_rl.runners import HIMOnPolicyRunner


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    assert args.task == 's10' and args.num_envs == 4096 and args.max_iterations == 500
    assert args.resume and args.headless and args.seed == 1
    assert len(options.source_commit) == 40 and all(c in '0123456789abcdef' for c in options.source_commit)
    # The host verifies this commit and a clean checkout before mounting source read-only.
    run, source_path = options.output, options.evidence / 'model_1500.pt'
    run.mkdir(parents=True, exist_ok=False)
    (run / 'source_commit.txt').write_text(options.source_commit + '\n')
    saved = json.loads((options.evidence / 'config.json').read_text())
    assert saved['env']['commands']['heading_command'] is True
    cfg, _ = task_registry.get_cfgs('s10')
    restore_config(cfg, saved['env'])
    cfg.commands.heading_command = False
    expected = copy.deepcopy(saved['env'])
    expected['commands']['heading_command'] = False
    train = copy.deepcopy(saved['train'])
    train['runner'].update(max_iterations=500, run_name=run.name, resume=True,
                           resume_path=str(source_path), checkpoint=1500)
    assert train['seed'] == saved['env']['seed'] == 1
    write_json(run / 'training_started.json', dict(start=1500, target=2000, additional_budget=500,
        source_checkpoint=str(source_path), source_commit=options.source_commit,
        recipe_change={'commands.heading_command': [True, False]}))
    env, _ = task_registry.make_env('s10', args=args, env_cfg=cfg)
    assert class_to_dict(cfg) == expected, 'Only heading_command may change'
    assert abs(env.dt - .02) < 1e-9 and abs(env.sim_params.dt - .0025) < 1e-9
    assert cfg.control.decimation == 8 and cfg.control.delay_stride == 2
    set_seed(train['seed'])
    runner = HIMOnPolicyRunner(env, train, str(run), device=args.rl_device)
    runner.load(str(source_path), load_optimizer=True)
    actor = runner.alg.actor_critic
    source = torch.load(source_path, map_location='cpu')
    equal_state(actor.state_dict(), source['model_state_dict'])
    optimizers = dict(ppo_optimizer_steps=runner.alg.optimizer,
                      estimator_optimizer_steps=actor.estimator.optimizer)
    for optimizer, key in zip(optimizers.values(), ['optimizer_state_dict', 'estimator_optimizer_state_dict']):
        equal_state(optimizer.state_dict(), source[key])
        assert {float(s['step']) for s in optimizer.state.values()} == {30000.}
    assert runner.alg.learning_rate == source['optimizer_state_dict']['param_groups'][0]['lr']
    assert actor.estimator.learning_rate == source['estimator_optimizer_state_dict']['param_groups'][0]['lr']
    assert runner.current_learning_iteration == 1500 and runner.tot_timesteps == 294912000
    assert runner.num_steps_per_env == 48 and runner.save_interval == 50
    write_json(run / 'config.json', dict(env=class_to_dict(cfg), train=train,
        initialization='full model_1500 + both Adam optimizers', interface_version=2, dof_names=env.dof_names))

    def curriculum():
        return dict(terrain_levels=env.terrain_levels.cpu().clone(),
            terrain_types=env.terrain_types.cpu().clone(), env_origins=env.env_origins.cpu().clone(),
            command_ranges=copy.deepcopy(env.command_ranges), common_step_counter=env.common_step_counter)

    write_json(run / 'resume_verification.json', dict(iteration=1500, historical_environment_steps=294912000,
        all_model_and_optimizer_tensors_equal_source=True, ppo_optimizer_steps=30000,
        estimator_optimizer_steps=30000, learning_rate=runner.alg.learning_rate,
        estimator_learning_rate=actor.estimator.learning_rate, restored_std=actor.std.tolist(),
        actual_pd_dt=env.sim_params.dt, actual_policy_dt=env.dt, delay_substeps=[0, 2, 4, 6],
        environment_handling='Fresh seed1 simulator/RNG/domain randomization and reconstructed curriculum; model_1500 contains no simulator/RNG/full curriculum state. No seamless physical resume claimed.',
        initial_terrain_mean=float(env.terrain_levels.float().mean()),
        initial_terrain_histogram=torch.bincount(env.terrain_levels, minlength=10).tolist(),
        initial_command_ranges=env.command_ranges, same_runner_learn_budget=500))
    counters = {key: 30000 for key in optimizers}
    for key, optimizer in optimizers.items():
        original = optimizer.step
        params = [p for group in optimizer.param_groups for p in group['params']]
        def checked_step(*a, original=original, params=params, key=key, **kw):
            require_finite([p.grad for p in params if p.grad is not None], key + ' gradients')
            result = original(*a, **kw)
            counters[key] += 1
            require_finite(params, key + ' parameters')
            assert (actor.std > 0).all().item(), 'Non-positive exploration std'
            return result
        optimizer.step = checked_step
    original_step = env.step
    def step(action):
        require_finite([action], 'policy action')
        result = original_step(action)
        require_finite([result[0], result[1], result[2], env.root_states, env.torques], 'rollout')
        return result
    env.step = step
    original_save, original_log = runner.save, runner.log
    def save(path, infos=None):
        require_finite(list(actor.state_dict().values()), 'checkpoint model')
        for optimizer in optimizers.values():
            require_finite([t for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t)], 'Adam state')
            assert {float(s['step']) for s in optimizer.state.values()} == {20. * runner.current_learning_iteration}
        assert runner.tot_timesteps == runner.current_learning_iteration * 4096 * 48
        original_save(path, infos=dict(curriculum=curriculum()))
    runner.save = save
    with (run / 'metrics.jsonl').open('x') as metrics:
        def log(locs):
            original_log(locs)
            values = dict(iteration=locs['it'] + 1,
                mean_reward=sum(locs['rewbuffer']) / len(locs['rewbuffer']) if locs['rewbuffer'] else None,
                mean_episode_length_steps=sum(locs['lenbuffer']) / len(locs['lenbuffer']) if locs['lenbuffer'] else None,
                std=actor.std.tolist(), learning_rate=runner.alg.learning_rate,
                estimator_learning_rate=actor.estimator.learning_rate, **runner.alg.update_stats,
                losses={k: locs[k] for k in ['mean_value_loss', 'mean_surrogate_loss', 'mean_estimation_loss', 'mean_swap_loss']},
                reward_components={k: sum(float(row[k]) for row in locs['ep_infos']) / len(locs['ep_infos'])
                    for k in locs['ep_infos'][0]} if locs['ep_infos'] else {},
                terrain_level_mean=float(env.terrain_levels.float().mean()),
                terrain_histogram=torch.bincount(env.terrain_levels, minlength=10).tolist(),
                command_ranges=copy.deepcopy(env.command_ranges), total_timesteps=runner.tot_timesteps,
                total_seconds=runner.tot_time, **counters)
            metrics.write(json.dumps(values, allow_nan=False) + '\n')
            metrics.flush()
        runner.log = log
        try:
            save(str(run / 'model_1500.pt'))
            print('RESUME_VERIFIED_ZERO_UPDATES', flush=True)
            runner.learn(num_learning_iterations=500, init_at_random_ep_len=True)
            assert runner.current_learning_iteration == 2000 and runner.tot_timesteps == 393216000
            assert set(counters.values()) == {40000}
            write_json(run / 'training_result.json', dict(status='completed', ppo_iterations=2000,
                additional_ppo_iterations=500, environment_steps=runner.tot_timesteps,
                additional_environment_steps=98304000, additional_steps_per_optimizer=10000, **counters))
        except BaseException:
            original_save(str(run / 'failure_state.pt'), infos=dict(curriculum=curriculum()))
            torch.save(dict(root=env.root_states, dof=env.dof_state, obs=env.obs_buf,
                            torques=env.torques), run / 'failure_sim_state.pt')
            write_json(run / 'training_result.json', dict(status='failed',
                ppo_iterations=runner.current_learning_iteration, error=traceback.format_exc(), **counters))
            raise
        finally:
            if runner.writer:
                runner.writer.close()
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
