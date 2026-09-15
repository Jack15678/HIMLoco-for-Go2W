"""One fixed full-state model_1500 -> 2000 parking experiment; no review wait."""
import copy
import json
import os
from pathlib import Path
import sys
import traceback

import isaacgym
import torch
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.scripts.evaluate_s10 import restore_config
from legged_gym.scripts.resume_s10_candidate import equal_state, require_finite, write_json
from rsl_rl.runners import HIMOnPolicyRunner

PARKING_SCALES = dict(parking_lin_vel=-.5, parking_ang_vel=-.25, parking_wheel_vel=-.002)


def main():
    check_only = '--check-only' in sys.argv
    if check_only:
        sys.argv.remove('--check-only')
    args = get_args()
    assert args.task == 's10' and args.num_envs == 4096 and args.max_iterations == 500 and args.resume
    source_path, run = Path('/checkpoints/model_1500.pt'), Path('/output')
    commit = os.environ['SOURCE_COMMIT']
    assert len(commit) == 40
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / 'training_started.json').exists(), 'Use a fresh output directory'
    write_json(run / 'training_started.json', dict(start=1500, target=2000,
        additional_budget=0 if check_only else 500, source_checkpoint=str(source_path), source_commit=commit))
    (run / 'source_commit.txt').write_text(commit + '\n')
    saved = json.loads(source_path.with_name('config.json').read_text())
    expected = copy.deepcopy(saved['env'])
    expected['commands'].update(parking_probability=.2, parking_thresholds=[.03, .03, .05])
    expected['rewards']['scales'].update(PARKING_SCALES)
    cfg, _ = task_registry.get_cfgs('s10')
    restore_config(cfg, expected)
    train = copy.deepcopy(saved['train'])
    train['runner'].update(max_iterations=500, run_name='s10_parking_1500_to2000_20260916',
        resume=True, resume_path=str(source_path), checkpoint=1500)
    env, _ = task_registry.make_env('s10', args=args, env_cfg=cfg)
    assert class_to_dict(cfg) == expected
    set_seed(train['seed'])
    runner = HIMOnPolicyRunner(env, train, str(run), device=args.rl_device)
    runner.load(str(source_path), load_optimizer=True)
    actor = runner.alg.actor_critic
    source = torch.load(source_path, map_location='cpu')
    optimizers = [runner.alg.optimizer, actor.estimator.optimizer]
    keys = ['optimizer_state_dict', 'estimator_optimizer_state_dict']

    def verify_source():
        equal_state(actor.state_dict(), source['model_state_dict'])
        for optimizer, key in zip(optimizers, keys):
            equal_state(optimizer.state_dict(), source[key])
            assert {float(s['step']) for s in optimizer.state.values()} == {30000.}
        assert runner.alg.learning_rate == source[keys[0]]['param_groups'][0]['lr']
        assert actor.estimator.learning_rate == source[keys[1]]['param_groups'][0]['lr']
        assert runner.current_learning_iteration == 1500 and runner.tot_timesteps == 294912000

    verify_source()
    assert runner.num_steps_per_env == 48 and runner.save_interval == 50
    assert cfg.seed == train['seed'] == 1
    assert abs(env.dt - .02) < 1e-9 and abs(env.sim_params.dt - .0025) < 1e-9
    write_json(run / 'config.json', dict(env=class_to_dict(cfg), train=train,
        initialization='full model_1500 + both Adam optimizers', interface_version=2, dof_names=env.dof_names))
    write_json(run / 'resume_verification.json', dict(iteration=1500,
        historical_environment_steps=294912000, all_model_and_optimizer_tensors_equal_source=True,
        ppo_optimizer_steps=30000, estimator_optimizer_steps=30000,
        learning_rate=runner.alg.learning_rate, estimator_learning_rate=actor.estimator.learning_rate,
        restored_std=actor.std.tolist(), actual_pd_dt=env.sim_params.dt, actual_policy_dt=env.dt,
        environment_handling='Fresh seed-1 simulator/domain randomization/terrain and rebuilt curriculum. Source has no simulator/RNG/full curriculum state. No physical-state seamless resume.',
        initial_terrain_histogram=torch.bincount(env.terrain_levels, minlength=10).tolist(),
        initial_command_ranges=env.command_ranges, source_commit=commit, delay_substeps=[0, 2, 4, 6]))

    # Observe the existing reward computation, including its pre-clip sum, without changing it.
    names = ['samples', 'parking', 'explicit_parking', 'clipped', 'parking_clipped',
        'raw_reward', 'parking_raw_reward', 'parking_lin_vel', 'parking_ang_vel', 'parking_wheel_vel',
        'resamples', 'explicit_samples', 'in_episode_stop_entries', 'moving_stop_entries']
    totals = torch.zeros(len(names), device=env.device, dtype=torch.float64)
    raw = torch.zeros(env.num_envs, device=env.device)
    terms = {}
    for i, (name, function) in enumerate(zip(env.reward_names, env.reward_functions)):
        def observed_reward(name=name, function=function):
            value = function()
            scaled = value * env.reward_scales[name]
            raw.add_(scaled)
            if name in PARKING_SCALES:
                terms[name] = scaled
            return value
        env.reward_functions[i] = observed_reward
    compute_reward = env.compute_reward
    def reward():
        raw.zero_()
        compute_reward()
        mask = env._parking_mask()
        totals[:10].add_(torch.stack([raw.new_tensor(env.num_envs), mask.sum(), env.parking_commands.sum(),
            (raw < 0).sum(), ((raw < 0) & mask).sum(), raw.sum(), raw[mask].sum(),
            *[terms[name].sum() for name in PARKING_SCALES]]))
    env.compute_reward = reward
    resample = env._resample_commands
    def sample(env_ids):
        previous = env._parking_mask()[env_ids]
        resample(env_ids)
        explicit = env.parking_commands[env_ids]
        entries = explicit & ~previous & (env.episode_length_buf[env_ids] > 0)
        moving = (env.base_lin_vel[env_ids, :2].norm(dim=1) > .1) | (env.base_ang_vel[env_ids, 2].abs() > .1)
        totals[10:].add_(torch.stack([raw.new_tensor(len(env_ids)), explicit.sum(), entries.sum(), (entries & moving).sum()]))
    env._resample_commands = sample

    def statistics():
        values = dict(zip(names, totals.tolist()))
        samples, parking = max(values['samples'], 1), max(values['parking'], 1)
        values.update(parking_fraction=values['parking']/samples,
            explicit_parking_fraction=values['explicit_parking']/samples,
            reward_clipped_fraction=values['clipped']/samples,
            parking_reward_clipped_fraction=values['parking_clipped']/parking,
            parking_mean_raw_reward=values['parking_raw_reward']/parking,
            parking_mean_terms_per_policy_step={k:values[k]/parking for k in PARKING_SCALES})
        return values

    original_save, original_log = runner.save, runner.log
    def save(path, infos=None):
        require_finite(list(actor.state_dict().values()), 'checkpoint model')
        for optimizer in optimizers:
            require_finite([v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)], 'Adam state')
            assert {float(s['step']) for s in optimizer.state.values()} == {20.*runner.current_learning_iteration}
        assert runner.tot_timesteps == runner.current_learning_iteration*4096*48
        original_save(path, infos=dict(terrain_levels=env.terrain_levels.cpu(),
            terrain_types=env.terrain_types.cpu(), command_ranges=copy.deepcopy(env.command_ranges)))
    runner.save = save
    try:
        if check_only:
            # Frozen source policy, same original randomized training environment, > one 10 s command window.
            actor.eval()
            with torch.inference_mode():
                for _ in range(600):
                    env.step(actor.act_inference(env.get_observations()))
                    require_finite([env.obs_buf, env.rew_buf, env.root_states, env.dof_vel, env.torques], 'zero-update rollout')
            verify_source()
            write_json(run / 'zero_update_rollout.json', dict(ppo_updates=0, seconds=12.,
                model_and_both_optimizers_unchanged=True, **statistics()))
            print('ZERO_UPDATE_CHECK_COMPLETE', json.dumps(statistics()), flush=True)
            return
        with (run / 'metrics.jsonl').open('x') as metrics:
            def log(locs):
                original_log(locs)
                require_finite([env.obs_buf, env.rew_buf, env.root_states, actor.std], 'training rollout')
                assert (actor.std > 0).all()
                values = dict(iteration=locs['it']+1, total_timesteps=runner.tot_timesteps,
                    mean_reward=sum(locs['rewbuffer'])/len(locs['rewbuffer']) if locs['rewbuffer'] else None,
                    mean_episode_length_steps=sum(locs['lenbuffer'])/len(locs['lenbuffer']) if locs['lenbuffer'] else None,
                    std=actor.std.tolist(), learning_rate=runner.alg.learning_rate,
                    estimator_learning_rate=actor.estimator.learning_rate, **runner.alg.update_stats,
                    losses={k:locs[k] for k in ['mean_value_loss','mean_surrogate_loss','mean_estimation_loss','mean_swap_loss']},
                    reward_components={k:sum(float(row[k]) for row in locs['ep_infos'])/len(locs['ep_infos'])
                        for k in locs['ep_infos'][0]} if locs['ep_infos'] else {},
                    terrain_level_mean=float(env.terrain_levels.float().mean()), command_ranges=copy.deepcopy(env.command_ranges),
                    parking=statistics(), ppo_optimizer_steps=20*(locs['it']+1), estimator_optimizer_steps=20*(locs['it']+1))
                metrics.write(json.dumps(values, allow_nan=False)+'\n'); metrics.flush()
                totals.zero_()
            runner.log = log
            save(str(run / 'model_1500.pt'))
            print('RESUME_VERIFIED_ZERO_UPDATES', flush=True)
            runner.learn(500, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == 2000 and runner.tot_timesteps == 393216000
        write_json(run / 'training_result.json', dict(status='completed', ppo_iterations=2000,
            additional_ppo_iterations=500, environment_steps=393216000, additional_environment_steps=98304000,
            ppo_optimizer_steps=40000, estimator_optimizer_steps=40000, additional_steps_each_optimizer=10000))
    except BaseException:
        original_save(str(run / 'failure_state.pt'))
        torch.save(dict(root=env.root_states, dof=env.dof_state, obs=env.obs_buf, torques=env.torques), run / 'failure_sim_state.pt')
        write_json(run / 'training_result.json', dict(status='failed', iteration=runner.current_learning_iteration,
            error=traceback.format_exc()))
        raise
    finally:
        if runner.writer:
            runner.writer.close()
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
