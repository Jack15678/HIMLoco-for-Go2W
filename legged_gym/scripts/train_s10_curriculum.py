"""One scratch 2000-iteration run, or a zero-update real-simulator preflight."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import isaacgym  # Must precede torch.
import torch
from legged_gym.envs import S10RoughCfg
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.scripts.resume_s10_candidate import require_finite, write_json
from rsl_rl.runners import HIMOnPolicyRunner


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--check-only', action='store_true')
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    assert args.task == 's10' and not args.resume and args.initialization == 'scratch'
    assert args.max_iterations == 2000 and args.seed == 1 and args.headless
    assert args.num_envs == (512 if options.check_only else 4096)
    assert len(options.source_commit) == 40 and all(c in '0123456789abcdef' for c in options.source_commit)
    run = options.output
    run.mkdir(parents=True, exist_ok=False)
    (run/'source_commit.txt').write_text(options.source_commit+'\n')
    cfg, train_cfg = task_registry.get_cfgs('s10')
    env, cfg = task_registry.make_env('s10', args=args, env_cfg=cfg)
    train_cfg.seed = 1
    train_cfg.runner.max_iterations = 2000
    train_cfg.runner.resume = False
    train = class_to_dict(train_cfg)
    set_seed(1)
    runner = HIMOnPolicyRunner(env, train, str(run), device=args.rl_device)
    actor = runner.alg.actor_critic
    optimizers = [runner.alg.optimizer, actor.estimator.optimizer]
    assert runner.current_learning_iteration == runner.tot_timesteps == 0
    assert all(not optimizer.state for optimizer in optimizers)
    assert runner.num_steps_per_env == 48 and runner.save_interval == 50
    assert env.dt == .02 and cfg.control.decimation == 8
    torch.testing.assert_close(actor.std, torch.tensor([.3, .3, .3, .6]*4, device=env.device))
    write_json(run/'config.json', dict(env=class_to_dict(cfg), train=train, initialization='scratch',
        interface_version=2, dof_names=env.dof_names, source_commit=options.source_commit))
    coverage = []
    for column in env.terrain_types.unique():
        selected = env.terrain_types == column
        count, high = int(selected.sum()), int(env.extended_speed_envs[selected].sum())
        assert high == round(count*.2)
        coverage.append(dict(column=int(column), terrain=int(env.task_curriculum.kinds[selected][0]),
                             environments=count, extended=high, fraction=high/count))
    write_json(run/'initialization_verification.json', dict(initialization='scratch', iteration=0,
        environment_steps=0, adam_states=[len(o.state) for o in optimizers],
        learning_rates=[o.param_groups[0]['lr'] for o in optimizers], std=actor.std.tolist(),
        terrain_speed_coverage=coverage, terrain_vertices=len(env.terrain.vertices),
        terrain_triangles=len(env.terrain.triangles), requested_iterations=2000,
        preflight_only=options.check_only))
    try:
        if options.check_only:
            # Production rollout/storage, including resets; no compute_returns/update/learn.
            snapshots = []
            resets = 0
            with torch.inference_mode():
                obs, critic = env.get_observations(), env.get_privileged_observations()
                for step in range(600):
                    old = obs.clone()
                    if step < 48:
                        actions = runner.alg.act(obs, critic)
                        snapshots.append(old)
                    else:
                        actions = actor.act_inference(obs)
                    obs, critic, rewards, dones, infos, ids, terminal = env.step(actions)
                    require_finite([obs, critic, rewards, env.root_states, env.torques], 'preflight rollout')
                    resets += len(ids)
                    if len(ids):
                        assert (obs[ids, 57:] == 0).all()
                    torch.testing.assert_close(obs[:, 6:9]/env.commands_scale, env.commands[:, :3])
                    if step < 48:
                        assert torch.equal(runner.alg.transition.observations, old)
                        next_critic = critic.clone(); next_critic[ids] = terminal
                        runner.alg.process_env_step(rewards, dones, infos, next_critic)
                assert torch.equal(runner.alg.storage.observations, torch.stack(snapshots))
                storage = runner.alg.storage
                actor.update_distribution(storage.observations.flatten(0, 1))
                ratio = torch.exp(actor.get_actions_log_prob(storage.actions.flatten(0, 1))-storage.actions_log_prob.flatten())
                assert (ratio-1).abs().max() < .001
                assert all(not o.state for o in optimizers)
                assert runner.current_learning_iteration == runner.tot_timesteps == 0
                write_json(run/'preflight.json', dict(status='passed', updates=0, policy_steps=600,
                    environment_steps=600*env.num_envs, resets=resets,
                    observation_snapshot_error=0., ratio_max_abs_error=float((ratio-1).abs().max()),
                    task_metrics=env.task_curriculum.report()))
            # A reviewable mesh sample, generated from the actual training fine-height map.
            import numpy as np
            np.savez_compressed(run/'pebbles_geometry.npz', heights=env.terrain.pebble_height_field,
                horizontal_scale=cfg.terrain.pebble_horizontal_scale, vertical_scale=cfg.terrain.pebble_vertical_scale)
            print('REAL_GYM_PREFLIGHT_PASSED_ZERO_UPDATES', flush=True)
            return

        original_save, original_log = runner.save, runner.log
        def save(path, infos=None):
            require_finite(list(actor.state_dict().values()), 'checkpoint model')
            for optimizer in optimizers:
                states = [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
                if states:
                    require_finite(states, 'checkpoint Adam')
                    assert {float(s['step']) for s in optimizer.state.values()} == {20.*runner.current_learning_iteration}
            original_save(path, infos=dict(terrain_levels=env.terrain_levels.cpu(), terrain_types=env.terrain_types.cpu(),
                extended_speed_envs=env.extended_speed_envs.cpu(), curriculum=env.task_curriculum.state_dict()))
        runner.save = save
        with (run/'metrics.jsonl').open('x') as metrics:
            def log(locs):
                original_log(locs)
                losses = {k: float(locs[k]) for k in ['mean_value_loss', 'mean_surrogate_loss', 'mean_estimation_loss', 'mean_swap_loss']}
                assert all(math.isfinite(v) for v in losses.values()) and torch.isfinite(actor.std).all() and (actor.std > 0).all()
                row = dict(iteration=locs['it']+1, total_timesteps=runner.tot_timesteps,
                    total_seconds=runner.tot_time, losses=losses, std=actor.std.tolist(),
                    learning_rate=runner.alg.learning_rate, estimator_learning_rate=actor.estimator.learning_rate,
                    terrain_histogram=torch.bincount(env.terrain_levels, minlength=cfg.terrain.num_rows).tolist(),
                    reward_components={k: sum(float(x[k]) for x in locs['ep_infos'])/len(locs['ep_infos'])
                        for k in locs['ep_infos'][0]} if locs['ep_infos'] else {},
                    curriculum=env.task_curriculum.report(), **runner.alg.update_stats)
                metrics.write(json.dumps(row, allow_nan=False)+'\n'); metrics.flush()
            runner.log = log
            runner.save(str(run/'model_0.pt'))
            print('SCRATCH_VERIFIED_STARTING_EXACTLY_2000_ITERATIONS', flush=True)
            runner.learn(2000, init_at_random_ep_len=False)
        assert runner.current_learning_iteration == 2000 and runner.tot_timesteps == 393216000
        assert all({float(s['step']) for s in o.state.values()} == {40000.} for o in optimizers)
        write_json(run/'training_result.json', dict(status='completed', iterations=2000,
            environment_steps=runner.tot_timesteps, optimizer_steps=[40000, 40000]))
    except BaseException:
        write_json(run/'training_result.json', dict(status='failed', iteration=runner.current_learning_iteration,
            environment_steps=runner.tot_timesteps, error=traceback.format_exc()))
        raise
    finally:
        if runner.writer:
            runner.writer.close()
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
