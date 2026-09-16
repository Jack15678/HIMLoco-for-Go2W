"""Observe natural spawn groups and early failures with a frozen training policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def summarize(rows):
    """Use only episodes with known outcomes at each horizon, including censored ones."""
    result = dict(episodes=len(rows), completed=sum(not r['censored'] for r in rows),
                  failures=sum(r['failed'] for r in rows),
                  demotion_causes=dict(Counter(r['demotion_reason'] for r in rows if r['delta'] == -1)),
                  failures_by_age={}, horizons={})
    for low, high, label in [(0, .2, '0-.2s'), (.2, 1, '.2-1s'), (1, 3, '1-3s'),
                             (3, 10, '3-10s'), (10, float('inf'), '>10s')]:
        result['failures_by_age'][label] = sum(r['failed'] and low < r['seconds'] <= high for r in rows)
    for horizon in [1, 3]:
        eligible = [r for r in rows if r['seconds'] >= horizon or r['failed']]
        failed = sum(r['failed'] and r['seconds'] <= horizon for r in eligible)
        result['horizons'][str(horizon)] = dict(eligible=len(eligible), failed=failed,
            fraction=failed / len(eligible) if eligible else None)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=120.)
    options, remaining = parser.parse_known_args()
    assert options.seconds >= 3
    import isaacgym  # Must precede torch.
    import torch
    from legged_gym.envs import S10RoughCfg
    from legged_gym.utils import get_args, task_registry
    from legged_gym.utils.helpers import class_to_dict, set_seed
    from legged_gym.scripts.evaluate_s10 import restore_config
    from legged_gym.utils.curriculum import TERRAINS, MODES
    from rsl_rl.modules import HIMActorCritic

    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    assert args.task == 's10' and args.headless and args.num_envs == 4096 and args.seed == 1
    saved = json.loads((options.checkpoint_path.parent / 'config.json').read_text())
    source = torch.load(options.checkpoint_path, map_location='cpu')
    cfg, _ = task_registry.get_cfgs('s10')
    restore_config(cfg, saved['env'])
    env, cfg = task_registry.make_env('s10', args=args, env_cfg=cfg)
    assert class_to_dict(cfg) == saved['env'], 'Observe the saved recipe without changing it'
    course = env.task_curriculum
    info = source['infos']
    assert torch.equal(env.terrain_types.cpu(), info['terrain_types'])
    env.terrain_levels.copy_(info['terrain_levels'])
    env.extended_speed_envs.copy_(info['extended_speed_envs'])
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    course.speed_limits.copy_(info['curriculum']['speed_limits'])
    actor = HIMActorCritic(env.num_obs, env.num_privileged_obs, env.num_one_step_obs,
                          env.num_actions, **saved['train']['policy']).to(env.device).eval()
    actor.load_state_dict(source['model_state_dict'])
    set_seed(1)
    options.output.mkdir(parents=True, exist_ok=False)
    (options.output / 'config.json').write_text(json.dumps(saved, indent=2) + '\n')
    active, episodes = [None] * env.num_envs, []
    tick = 0
    peak_tilt = torch.zeros(env.num_envs, device=env.device)
    first_contact = torch.full_like(peak_tilt, -1.)
    log = (options.output / 'episodes.jsonl').open('w')

    def emit(row):
        episodes.append(row)
        log.write(json.dumps(row, allow_nan=False) + '\n')

    original_reset = env.reset_idx
    def reset(ids):
        original_reset(ids)
        if not len(ids):
            return
        root = env.root_states[ids]
        offsets = root.new_tensor([[0, 0, 0], [.35, .25, 0], [.35, -.25, 0],
                                  [-.35, .25, 0], [-.35, -.25, 0]])
        heights = env._surface_heights(root[:, None, :3] + offsets)
        local = root[:, :3] - env.env_origins[ids]
        data = torch.cat((local, root[:, 7:13], heights.amin(1, keepdim=True),
                          heights.amax(1, keepdim=True), env.terrain_levels[ids, None],
                          course.kinds[ids, None], course.mode[ids, None],
                          env.extended_speed_envs[ids, None]), dim=1).cpu().tolist()
        for index, values in zip(ids.tolist(), data):
            active[index] = dict(env=index, start_step=tick,
                spawn='terrain' if abs(values[0]) > 1.5 else 'platform',
                local_position=values[:3], initial_velocity=values[3:9],
                support_height_range=values[10] - values[9], level=int(values[11]),
                terrain=TERRAINS[int(values[12])], initial_mode=MODES[int(values[13])],
                extended=bool(values[14]), snapshots={})
        peak_tilt[ids] = 0.
        first_contact[ids] = -1.
    env.reset_idx = reset

    original_termination = env.check_termination
    def termination():
        original_termination()
        age = env.episode_length_buf * env.dt
        tilt = torch.rad2deg(torch.acos((-env.projected_gravity[:, 2]).clamp(-1., 1.)))
        peak_tilt[:] = torch.maximum(peak_tilt, torch.where(age <= 3.00001, tilt, 0.))
        loaded = (env.contact_forces[:, env.feet_indices].norm(dim=2) > 1.).sum(1)
        new_contact = (first_contact < 0) & (loaded > 0)
        first_contact[new_contact] = age[new_contact]
        sampled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for step in [1, round(.2 / env.dt), round(1. / env.dt), round(3. / env.dt)]:
            sampled |= env.episode_length_buf == step
        ids = sampled.nonzero(as_tuple=False).flatten()
        if len(ids):
            feet = env.feet_pos[ids]
            # Vertical wheel-center minus nominal radius is a clearance proxy, not collision depth.
            gaps = feet[:, :, 2] - env._surface_heights(feet) - .081
            values = torch.cat((age[ids, None], loaded[ids, None], tilt[ids, None], gaps), dim=1).cpu().tolist()
            for index, value in zip(ids.tolist(), values):
                active[index]['snapshots'][f'{value[0]:.2f}'] = dict(
                    loaded_wheels=int(value[1]), tilt_degrees=value[2], wheel_clearance_proxy=value[3:])
    env.check_termination = termination

    original_finish = course.finish
    def finish(ids, failed):
        seconds = course.episode_seconds[ids].clone()
        score = course.episode_good[ids] / seconds[:, None].clamp_min(1e-9)
        duration = env.episode_length_buf[ids].float() * env.dt
        data = torch.cat((duration[:, None], failed[:, None], seconds[:, None], score,
                          course.episode_target[ids, None], peak_tilt[ids, None], first_contact[ids, None]), dim=1).cpu().tolist()
        delta = original_finish(ids, failed)
        for index, value, change in zip(ids.tolist(), data, delta.tolist()):
            assert active[index] is not None
            reason = 'base_contact' if value[1] else ('tracking' if change < 0 else None)
            emit(dict(active[index], seconds=value[0], failed=bool(value[1]), censored=False,
                delta=change, demotion_reason=reason, scored_seconds=value[2], axis_scores=value[3:6],
                target_seconds=value[6], max_tilt_first3_degrees=value[7], first_contact_seconds=value[8],
                ending_mode=MODES[int(course.mode[index])]))
        return delta
    course.finish = finish

    try:
        # Fresh natural resets, with no artificial zero-action physics step before observation starts.
        env.init_done = False
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        env.init_done = True
        env.compute_observations()
        obs = env.get_observations()
        steps = round(options.seconds / env.dt)
        with torch.inference_mode():
            for tick in range(1, steps + 1):
                obs, _, _, _, _, _, _ = env.step(actor.act(obs))
                assert torch.isfinite(obs).all() and torch.isfinite(env.root_states).all()
                if tick % 500 == 0:
                    log.flush()
                    print(json.dumps(dict(policy_steps=tick, simulated_seconds=tick * env.dt,
                        completed=len(episodes), failures=sum(r['failed'] for r in episodes))), flush=True)
        durations = (env.episode_length_buf * env.dt).tolist()
        for index, row in enumerate(active):
            emit(dict(row, seconds=durations[index], failed=False, censored=True, delta=0,
                demotion_reason=None, max_tilt_first3_degrees=float(peak_tilt[index]),
                first_contact_seconds=float(first_contact[index])))
        assert all(torch.equal(value.cpu(), source['model_state_dict'][key])
                   for key, value in actor.state_dict().items())
        assert not actor.estimator.optimizer.state
        assert sum(r['failed'] for r in episodes) == int(course.failures.sum())
        assert sum(r['delta'] < 0 for r in episodes) == int(course.down_count.sum())
        groups = []
        for terrain in TERRAINS:
            for spawn in ['platform', 'terrain']:
                for level in [None] + sorted({r['level'] for r in episodes if r['terrain'] == terrain}):
                    rows = [r for r in episodes if r['terrain'] == terrain and r['spawn'] == spawn
                            and (level is None or r['level'] == level)]
                    groups.append(dict(terrain=terrain, spawn=spawn, level=level, **summarize(rows)))
        result = dict(checkpoint=str(options.checkpoint_path), iteration=source['iter'],
            ppo_updates=0, model_unchanged=True, policy_steps=steps, seconds_per_env=steps * env.dt,
            num_envs=env.num_envs, sampling='Saved training recipe, stochastic policy, original resets and live curriculum; restored checkpoint levels/membership/caps; fresh physical episodes and counters.',
            interpretation='Natural random spawn strata, not a changed-initialization A/B. Early failures also include command, exploration and randomization effects. Age rates exclude episodes censored before the horizon.',
            initial_histogram=torch.bincount(info['terrain_levels'], minlength=cfg.terrain.num_rows).tolist(),
            final_histogram=torch.bincount(env.terrain_levels, minlength=cfg.terrain.num_rows).tolist(),
            overall=summarize(episodes), groups=groups)
        (options.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print('SPAWN_DIAGNOSIS_COMPLETED_ZERO_UPDATES', flush=True)
    finally:
        log.close()
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        rows = [dict(seconds=t, failed=f, censored=c, delta=-int(f), demotion_reason='base_contact' if f else None)
                for t, f, c in [(.1, True, False), (.5, False, True), (2., True, False), (4., False, True)]]
        result = summarize(rows)
        assert result['horizons']['1'] == dict(eligible=3, failed=1, fraction=1/3)
        assert result['horizons']['3'] == dict(eligible=3, failed=2, fraction=2/3)
        assert result['failures_by_age']['0-.2s'] == result['failures_by_age']['1-3s'] == 1
        assert result['demotion_causes'] == {'base_contact': 2}
        json.dumps(result, allow_nan=False)
        print('Spawn age denominators, censoring and failure bins passed.')
    else:
        main()
