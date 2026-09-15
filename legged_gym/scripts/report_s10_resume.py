"""Verify cumulative checkpoints and summarize/plot the single resumed run."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(run):
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    end = rows[-1]['iteration']
    assert end in [500, 1000]
    assert [r['iteration'] for r in rows] == list(range(201, end+1))
    assert all(r['total_timesteps'] == r['iteration']*4096*48 for r in rows)
    assert all(r['nonfinite_count'] == 0 for r in rows)
    assert all(r['ppo_optimizer_steps'] == r['estimator_optimizer_steps'] == r['iteration']*20 for r in rows)
    assert all(len(r['minibatch_kl']) == len(r['minibatch_learning_rates']) == 20 for r in rows)
    assert all(np.isclose(r['kl_mean'], np.mean(r['minibatch_kl'])) and r['kl_max'] == max(r['minibatch_kl']) for r in rows)
    assert all(0 <= r['clip_fraction'] <= 1 and r['learning_rate'] == r['estimator_learning_rate'] for r in rows)
    checks = []
    initial = torch.load(run / 'model_200.pt', map_location='cpu')
    for iteration in range(200, end+1, 50):
        checkpoint = torch.load(run / f'model_{iteration}.pt', map_location='cpu')
        assert checkpoint['iter'] == iteration and checkpoint['tot_timesteps'] == iteration*4096*48
        assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
        optimizers = {}
        for key in ['optimizer_state_dict', 'estimator_optimizer_state_dict']:
            state = checkpoint[key]['state']
            assert {float(v['step']) for v in state.values()} == {20.*iteration}
            assert all(torch.isfinite(t).all() for entry in state.values() for t in entry.values() if torch.is_tensor(t))
            optimizers[key] = dict(entries=len(state), steps=20*iteration, lr=checkpoint[key]['param_groups'][0]['lr'])
        curriculum = checkpoint['infos']['curriculum']
        checks.append(dict(iteration=iteration, environment_steps=checkpoint['tot_timesteps'], finite=True,
            optimizers=optimizers, std=checkpoint['model_state_dict']['std'].tolist(),
            curriculum_terrain_mean=float(curriculum['terrain_levels'].float().mean()),
            common_step_counter=curriculum['common_step_counter'], command_ranges=curriculum['command_ranges'],
            changed_tensors_from_200={prefix: sum(not torch.equal(v, initial['model_state_dict'][k])
                for k, v in checkpoint['model_state_dict'].items() if k.startswith(prefix+'.'))
                for prefix in ['actor', 'critic', 'estimator']}))
    windows = []
    for start in range(201, end, 20):
        chunk = [r for r in rows if start <= r['iteration'] < start+20]
        windows.append(dict(start=start, end=chunk[-1]['iteration'],
            reward=float(np.mean([r['mean_reward'] for r in chunk])),
            episode_seconds=float(np.mean([r['mean_episode_length_steps'] for r in chunk])*.02),
            terrain=float(np.mean([r['terrain_level_mean'] for r in chunk])),
            tracking_lin=float(np.mean([r['reward_components'].get('rew_tracking_lin_vel', np.nan) for r in chunk])),
            tracking_yaw=float(np.mean([r['reward_components'].get('rew_tracking_ang_vel', np.nan) for r in chunk])),
            kl_mean=float(np.mean([r['kl_mean'] for r in chunk])),
            kl_max=max(r['kl_max'] for r in chunk),
            clip_fraction=float(np.mean([r['clip_fraction'] for r in chunk])),
            lr_min=min(min(r['minibatch_learning_rates']) for r in chunk),
            lr_max=max(max(r['minibatch_learning_rates']) for r in chunk),
            estimation_loss=float(np.mean([r['losses']['mean_estimation_loss'] for r in chunk])),
            swap_loss=float(np.mean([r['losses']['mean_swap_loss'] for r in chunk])),
            command_ranges=chunk[-1]['command_ranges']))
    summary = dict(iteration=end, additional_iterations=end-200, environment_steps=end*4096*48,
        additional_environment_steps=(end-200)*4096*48, optimizer_steps=end*20,
        additional_training_seconds=rows[-1]['total_seconds']-initial['tot_time'],
        nonfinite_count=0, windows=windows, checkpoints=checks,
        kl_max_definition='maximum of minibatch mean adaptive KL, not maximum per-sample KL',
        estimator_loss_caveat='201 onward averages all 20 minibatches; 1-200 logged only last minibatch')
    (run / f'verification_{end}.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    source = run / 'source_metrics.jsonl'
    old = [json.loads(line) for line in source.read_text().splitlines()]
    all_rows, x = old + rows, np.arange(1, end+1)
    fig, axes = plt.subplots(5, 2, figsize=(14, 16), constrained_layout=True)
    ax = axes.ravel()
    ax[0].plot(x, [r['mean_reward'] for r in all_rows], lw=.8)
    ax[0].set(title='Mean episode return')
    ax[1].plot(x, [r['mean_episode_length_steps']*.02 for r in all_rows], lw=.8)
    ax[1].set(title='Mean episode length (seconds)')
    ax[2].plot(x, [r['reward_components'].get('terrain_level', np.nan) for r in all_rows], lw=.8)
    ax[2].set(title='Terrain level (episode log)')
    for key in ['tracking_lin_vel', 'tracking_ang_vel']:
        ax[3].plot(x, [r['reward_components'].get('rew_'+key, np.nan) for r in all_rows], label=key, lw=.8)
    ax[3].set(title='Weighted velocity tracking reward'); ax[3].legend()
    newx = np.arange(201, end+1)
    for key in ['kl_mean', 'kl_max']:
        ax[4].plot(newx, [r[key] for r in rows], label=key, lw=.8)
    ax[4].axhline(.01, color='gray', ls='--', label='desired KL')
    ax[4].set(title='Adaptive KL: mean / maximum minibatch mean'); ax[4].legend()
    ax[5].semilogy(x, [r['learning_rate'] for r in all_rows], lw=.8)
    ax[5].set(title='Actual PPO and estimator learning rate (same)')
    ax[6].plot(newx, [r['clip_fraction'] for r in rows], lw=.8)
    ax[6].set(title='PPO clip fraction')
    for key in ['mean_estimation_loss', 'mean_swap_loss']:
        ax[7].plot(newx, [r['losses'][key] for r in rows], label=key, lw=.8)
    ax[7].set(title='HIM losses: whole-update means from iteration 201'); ax[7].legend()
    std = np.asarray([r['std'] for r in all_rows])
    for offset, (name, color) in enumerate(zip(['hipx', 'hipy', 'knee', 'wheel'], ['#517c98', '#579375', '#b98449', '#ad6579'])):
        for i in range(offset, 16, 4):
            ax[8].plot(x, std[:, i], color=color, lw=.8, label=name if i == offset else None)
    ax[8].set(title='All 16 learned exploration standard deviations'); ax[8].legend()
    ax[9].plot(x, [r['reward_components'].get('max_command_x', np.nan) for r in all_rows])
    ax[9].set(title='Maximum forward command (m/s)')
    for a in ax:
        a.axvline(200, color='gray', ls=':', lw=.8)
        a.axvline(500, color='gray', ls=':', lw=.8)
        a.set_xlabel('Cumulative PPO iteration'); a.grid(alpha=.2)
    fig.savefig(run / f'training_curves_{end}.png', dpi=140)
    plt.close(fig)
    print(json.dumps(dict(iteration=end, windows=windows[-3:], last_checkpoint=checks[-1]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    report(parser.parse_args().run)
