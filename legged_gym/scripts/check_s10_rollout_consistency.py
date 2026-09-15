"""Real Gym rollout consistency gate; never calls learn/update or saves a model.

python legged_gym/scripts/check_s10_rollout_consistency.py --task s10 --headless \
    --num_envs 512 --evidence /evidence --output /output/consistency.json
The evidence directory contains read-only config.json and model_1000.pt.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import isaacgym  # Must precede torch.
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import S10RoughCfg
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.scripts.evaluate_s10 import restore_config
from legged_gym.scripts.resume_s10_candidate import equal_state
from rsl_rl.runners import HIMOnPolicyRunner


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    check, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    assert args.task == 's10' and args.num_envs == 512 and args.headless
    assert not args.resume and args.max_iterations is None
    root = Path(LEGGED_GYM_ROOT_DIR)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    saved = json.loads((check.evidence / 'config.json').read_text())
    cfg, _ = task_registry.get_cfgs('s10')
    restore_config(cfg, saved['env'])
    env, _ = task_registry.make_env('s10', args=args, env_cfg=cfg)
    try:
        expected = copy.deepcopy(saved['env'])
        expected['env']['num_envs'] = 512
        assert class_to_dict(cfg) == expected, 'Only environment count may differ'
        assert env.sim_params.dt == .0025 and cfg.control.decimation == 8 and env.dt == .02
        assert cfg.control.delay_stride == 2
        assert saved['train']['policy']['init_noise_std'] == [.3, .3, .3, .6] * 4
        runner = HIMOnPolicyRunner(env, saved['train'], log_dir=None, device=args.rl_device)
        checkpoint = check.evidence / 'model_1000.pt'
        runner.load(str(checkpoint), load_optimizer=True)
        assert runner.current_learning_iteration == 1000 and runner.num_steps_per_env == 48
        alg, actor = runner.alg, runner.alg.actor_critic
        before = {k: v.clone() for k, v in actor.state_dict().items()}
        optimizers = dict(ppo=alg.optimizer, him=actor.estimator.optimizer)
        optimizer_before = {k: copy.deepcopy(v.state_dict()) for k, v in optimizers.items()}
        counters = dict(ppo_update=0, him_update=0, ppo_step=0, him_step=0)

        def forbid(key):
            def fail(*a, **kw):
                counters[key] += 1
                raise AssertionError('Zero-training gate attempted ' + key)
            return fail

        alg.update, actor.estimator.update = forbid('ppo_update'), forbid('him_update')
        alg.optimizer.step = forbid('ppo_step')
        actor.estimator.optimizer.step = forbid('him_step')
        actor.train()
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
        obs = env.get_observations().to(args.rl_device)
        critic = env.get_privileged_observations().to(args.rl_device)
        originals, critics, terminals, rewritten = [], [], [], []
        history_max_error = 0.
        with torch.inference_mode():
            # Same act -> env.step -> process_env_step ordering as HIMOnPolicyRunner.
            for step in range(runner.num_steps_per_env):
                originals.append(obs.clone())
                critics.append(critic.clone())
                env_owned_obs = obs
                actions = alg.act(obs, critic)
                original_actions = actions.clone()
                obs, critic, rewards, dones, infos, ids, terminal_critic = env.step(actions)
                obs, critic = obs.to(args.rl_device), critic.to(args.rl_device)
                rewards, dones = rewards.to(args.rl_device), dones.to(args.rl_device)
                ids, terminal_critic = ids.to(args.rl_device), terminal_critic.to(args.rl_device)
                terminals.append(dones.clone().bool())
                rewritten.append((env_owned_obs != originals[-1]).any(-1))
                assert torch.equal(alg.transition.observations, originals[-1])
                assert torch.equal(alg.transition.critic_observations, critics[-1])
                if ids.numel():
                    history_max_error = max(history_max_error, float(obs[ids, env.num_one_step_obs:].abs().max()))
                next_critic = critic.clone()
                next_critic[ids] = terminal_critic
                alg.process_env_step(rewards, dones, infos, next_critic)
                assert torch.equal(alg.storage.actions[step], original_actions)
                assert torch.equal(alg.storage.next_privileged_observations[step], next_critic)

            storage = alg.storage
            original_obs = torch.stack(originals).flatten(0, 1)
            actual_obs = storage.observations.flatten(0, 1)
            critic_error = (storage.privileged_observations - torch.stack(critics)).abs()
            obs_error = (actual_obs - original_obs).abs()
            terminal = torch.stack(terminals).flatten()
            changed = torch.stack(rewritten).flatten()
            assert terminal.any() and (~terminal).any()
            assert torch.equal(storage.dones.flatten().bool(), terminal)
            assert torch.equal(changed, terminal), 'Exercise the real reset alias hazard'
            assert obs_error.max() == critic_error.max() == history_max_error == 0
            old_mu, old_sigma = storage.mu.flatten(0, 1), storage.sigma.flatten(0, 1)
            actor.update_distribution(actual_obs)
            mu, sigma = actor.action_mean, actor.action_std
            kl = (torch.log(sigma / old_sigma + 1e-5)
                  + (old_sigma.square() + (old_mu - mu).square()) / (2 * sigma.square()) - .5).sum(-1)
            floor = (torch.log(torch.ones_like(old_sigma) + 1e-5) + .5 - .5).sum(-1)
            ratio = torch.exp(actor.get_actions_log_prob(storage.actions.flatten(0, 1))
                              - storage.actions_log_prob.flatten())
            assert torch.isfinite(kl).all() and torch.isfinite(ratio).all()
            assert (kl - floor).abs().max() < 1e-5
            assert (ratio - 1).abs().max() < 1e-3
            assert ((ratio - 1).abs() > alg.clip_param).sum() == 0

            def metrics(mask):
                return dict(samples=int(mask.sum()), kl_mean=float(kl[mask].mean()),
                            ratio_min=float(ratio[mask].min()), ratio_max=float(ratio[mask].max()),
                            ratio_max_abs_error=float((ratio[mask] - 1).abs().max()),
                            observation_max_abs_error=float(obs_error[mask].max()))

            equal_state(actor.state_dict(), before)
            optimizer_steps = {}
            for key, optimizer in optimizers.items():
                equal_state(optimizer.state_dict(), optimizer_before[key])
                old_steps = sorted({float(s['step']) for s in optimizer_before[key]['state'].values()})
                new_steps = sorted({float(s['step']) for s in optimizer.state.values()})
                optimizer_steps[key] = dict(before=old_steps, after=new_steps, added=counters[key + '_step'])
            assert not any(counters.values())
            result = dict(status='passed', source_commit=commit, checkpoint=str(checkpoint),
                          checkpoint_iteration=runner.current_learning_iteration,
                          checkpoint_use='Read-only diagnosis, not initialization for future scratch training',
                          num_envs=env.num_envs, policy_steps=runner.num_steps_per_env,
                          setup='Original training config except 512 envs; runner reset and random episode ages; no forced reset or warmup',
                          all_samples=metrics(torch.ones_like(terminal)), terminal=metrics(terminal),
                          nonterminal=metrics(~terminal), raw_env_observation_rewritten_samples=int(changed.sum()),
                          saved_observation_changed_samples=int((obs_error != 0).any(-1).sum()),
                          critic_observation_max_abs_error=float(critic_error.max()),
                          reset_history_max_abs_error=history_max_error,
                          next_critic_and_actions_equal=True,
                          kl_numerical_floor=float(floor.mean()), kl_max_abs_floor_error=float((kl - floor).abs().max()),
                          action_mean_max_abs_error=float((mu - old_mu).abs().max()), clip_fraction=0.,
                          calls=counters, optimizer_steps=optimizer_steps, optimizer_states_unchanged=True,
                          all_model_state_tensors_unchanged=True, model_state_tensor_count=len(before),
                          parameter_elements=sum(p.numel() for p in actor.parameters()),
                          pd_dt=env.sim_params.dt, policy_dt=env.dt, delay_stride=cfg.control.delay_stride)
            check.output.parent.mkdir(parents=True, exist_ok=True)
            check.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            print('CONSISTENCY_RESULT ' + json.dumps(result, allow_nan=False), flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
