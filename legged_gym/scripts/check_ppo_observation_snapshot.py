"""CPU regression: PPO/HIMPPO must own pre-step actor and critic observations.

PYTHONPATH=rsl_rl python legged_gym/scripts/check_ppo_observation_snapshot.py
No simulator, checkpoint, or parameter updates.
"""
import torch
from rsl_rl.algorithms import PPO, HIMPPO
from rsl_rl.modules import ActorCritic, HIMActorCritic


def main():
    torch.manual_seed(1)
    with torch.inference_mode():
        for algorithm in (PPO, HIMPPO):
            for shared_critic in (False, True):
                critic_size = 342 if shared_critic else 262
                dims = dict(actor_hidden_dims=[16], critic_hidden_dims=[16])
                actor = (HIMActorCritic(342, critic_size, 57, 16, **dims)
                         if algorithm is HIMPPO else ActorCritic(342, critic_size, 16, **dims))
                before = {k: v.clone() for k, v in actor.state_dict().items()}
                alg = algorithm(actor)
                alg.init_storage(2, 2, [342], [critic_size], [16])
                obs = torch.randn(2, 342)
                critic = obs if shared_critic else torch.randn(2, critic_size)
                originals, critics = [], []
                for step in range(2):
                    originals.append(obs.clone())
                    critics.append(critic.clone())
                    actions = alg.act(obs, critic).clone()
                    # Reuse buffers: one episode resets, the other advances normally.
                    obs[0] = 0
                    obs[1] += 1
                    if not shared_critic:
                        critic[0] = 0
                        critic[1] += 1
                    assert torch.equal(alg.transition.observations, originals[-1])
                    assert torch.equal(alg.transition.critic_observations, critics[-1])
                    args = (torch.zeros(2), torch.tensor([True, False]), {})
                    alg.process_env_step(*args, *([critic] if algorithm is HIMPPO else []))
                    storage = alg.storage
                    assert torch.equal(storage.observations[step], originals[-1])
                    assert torch.equal(storage.privileged_observations[step], critics[-1])
                    assert torch.equal(storage.actions[step], actions)
                    actor.update_distribution(storage.observations[step])
                    torch.testing.assert_close(actor.action_mean, storage.mu[step], rtol=0, atol=0)
                    torch.testing.assert_close(actor.get_actions_log_prob(actions),
                                               storage.actions_log_prob[step, :, 0], rtol=0, atol=0)
                    torch.testing.assert_close(actor.evaluate(storage.privileged_observations[step]),
                                               storage.values[step], rtol=0, atol=0)
                assert torch.equal(storage.observations, torch.stack(originals))
                assert torch.equal(storage.privileged_observations, torch.stack(critics))
                assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())
                assert not alg.optimizer.state
                if algorithm is HIMPPO:
                    assert not actor.estimator.optimizer.state
                print(f'PASS: {algorithm.__name__}, shared_critic={shared_critic}, 0 updates')


if __name__ == '__main__':
    main()
