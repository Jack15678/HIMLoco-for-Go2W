"""CPU synthetic check of actual HIMPPO.update logging; no simulator/checkpoint."""
import math
from types import SimpleNamespace
import torch
from rsl_rl.algorithms import HIMPPO


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        losses = iter([(2., 4.), (6., 8.)])
        self.estimator = SimpleNamespace(update=lambda *a, **kw: next(losses))

    def act(self, obs):
        self.action_mean = self.weight.expand(3, 1)
        self.action_std = torch.ones(3, 1)
        self.entropy = self.weight.expand(3)*0

    def get_actions_log_prob(self, actions):
        return torch.tensor([math.log(.5), 0., math.log(1.5)]) + self.weight*0

    def evaluate(self, obs):
        return self.weight.expand(3, 1)


if __name__ == '__main__':
    actor = Actor()
    alg = HIMPPO(actor, num_learning_epochs=1, num_mini_batches=2, schedule='adaptive')
    z, one = torch.zeros(3, 1), torch.ones(3, 1)
    batch = (z, z, z, z, z, one, z, z, z, one)
    alg.storage = SimpleNamespace(mini_batch_generator=lambda *a: iter([batch, batch]), clear=lambda: None)
    value, surrogate, estimation, swap = alg.update()
    assert estimation == 4. and swap == 6., (estimation, swap)
    stats = alg.update_stats
    assert abs(stats['clip_fraction'] - 2./3) < 1e-6
    expected_kl = float(torch.log(torch.tensor(1. + 1e-5)))
    assert abs(stats['kl_mean'] - expected_kl) < 1e-7
    assert stats['kl_max'] == max(stats['minibatch_kl'])
    assert len(stats['minibatch_kl']) == len(stats['minibatch_learning_rates']) == 2
    print('PASS: actual update returns mean HIM losses; KL and clip fraction match synthetic minibatches')
