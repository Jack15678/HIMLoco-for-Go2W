"""CPU save-hook regression check: no Gym, simulator, rollout or PPO updates."""
import copy
import io
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rsl_rl'))
from rsl_rl.export_s10 import DOF_NAMES, export_checkpoint
from rsl_rl.modules import HIMActorCritic
from rsl_rl.runners import HIMOnPolicyRunner


def main():
    torch.set_num_threads(1)
    runner = HIMOnPolicyRunner.__new__(HIMOnPolicyRunner)
    runner.policy_cfg = dict(actor_hidden_dims=[16], critic_hidden_dims=[16])
    actor = HIMActorCritic(342, 262, 57, 16, **runner.policy_cfg).train()
    # Mixed submodule modes detect accidental eval()/train() on the live network.
    actor.estimator.target.eval()
    runner.alg = NS(actor_critic=actor, optimizer=torch.optim.Adam(actor.parameters()))
    runner.env = NS(
        cfg=NS(asset=NS(name='s10', self_collisions=1), control=NS(vel_scale=5., decimation=8),
               normalization=NS(clip_actions=100., clip_observations=100.), init_state=NS(pos=[0., 0., .45])),
        obs_scales=NS(ang_vel=.25, lin_vel=2., dof_pos=1., dof_vel=.05, height_measurements=5.),
        sim_params=NS(dt=.0025), dof_names=DOF_NAMES, wheel_indices=torch.tensor([3, 7, 11, 15]),
        default_dof_pos=torch.tensor([[0., -.3, .6, 0.] * 2 + [0., .3, -.6, 0.] * 2]),
        p_gains=torch.tensor([80., 80., 80., 0.] * 4), d_gains=torch.tensor([2., 2., 2., .6] * 4),
        action_scale=torch.tensor([.125, .25, .25, 0.] * 4), torque_limits=torch.tensor([50., 50., 50., 14.] * 4),
        dof_vel_limits=torch.tensor([25.76, 25.76, 25.76, 65.5] * 4), commands_scale=torch.tensor([2., 2., .25]),
        num_one_step_obs=57, num_actions=16)
    runner.num_actor_obs, runner.num_critic_obs = 342, 262
    runner.current_learning_iteration, runner.tot_timesteps, runner.tot_time = 7, 123, 4.5
    before = copy.deepcopy(actor.state_dict())
    modes = [m.training for m in actor.modules()]
    devices = [p.device for p in actor.parameters()]
    optimizers = [runner.alg.optimizer, actor.estimator.optimizer]
    # Give optimizer state real tensors without performing a training step.
    for optimizer in optimizers:
        parameter = optimizer.param_groups[0]['params'][0]
        optimizer.state[parameter] = dict(step=torch.tensor(5.), exp_avg=torch.ones_like(parameter),
                                          exp_avg_sq=torch.ones_like(parameter))
    optimizer_bytes = io.BytesIO()
    torch.save([o.state_dict() for o in optimizers], optimizer_bytes)
    rng = torch.get_rng_state().clone()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / 'model_7.pt'
        runner.save(checkpoint)
        assert checkpoint.is_file()
        report = json.loads((root / 'onnx/model_7/export_verification.json').read_text())
        assert report['passed'] and report['sample_count'] == 34
        sidecar = json.loads((root / 'onnx/model_7/policy.json').read_text())
        assert sidecar['policy_dt'] == .02 and sidecar['dof_names'] == DOF_NAMES
        assert sidecar['input_shape'] == [1, 342] and sidecar['output_shape'] == [1, 16]
        saved = torch.load(checkpoint, weights_only=False)
        assert saved['iter'] == 7 and saved['tot_timesteps'] == 123
        assert saved['s10_export']['metadata']['p_gains'] == [80., 80., 80., 0.] * 4
        runner.current_learning_iteration = 8
        runner.save(root / 'model_8.pt')
        assert (root / 'onnx/model_7/policy.onnx').is_file() and (root / 'onnx/model_8/policy.onnx').is_file()
        messages = io.StringIO()
        handler = logging.StreamHandler(messages)
        logging.getLogger().addHandler(handler)
        try:
            with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'export', stderr='injected native/dependency failure')):
                runner.save(root / 'model_9.pt')
        finally:
            logging.getLogger().removeHandler(handler)
        assert 'ONNX export FAILED' in messages.getvalue()
        assert torch.load(root / 'model_9.pt', weights_only=False)['iter'] == 8
        assert not (root / 'onnx/model_9/export_verification.json').exists()
        # Non-S10 saves must not load ONNX dependencies or call the exporter.
        runner.env.cfg.asset.name = 'go2w'
        with patch('subprocess.run', side_effect=AssertionError('unexpected export')) as export:
            runner.save(root / 'go2w.pt')
            export.assert_not_called()
        saved['s10_export']['dimensions'][0] = 174
        torch.save(saved, root / 'bad_shape.pt')
        try:
            export_checkpoint(root / 'bad_shape.pt')
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid model shape was accepted')
    assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())
    assert modes == [m.training for m in actor.modules()]
    assert devices == [p.device for p in actor.parameters()]
    after_bytes = io.BytesIO()
    torch.save([o.state_dict() for o in optimizers], after_bytes)
    assert optimizer_bytes.getvalue() == after_bytes.getvalue()
    assert torch.equal(rng, torch.get_rng_state())
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and np.array_equal(numpy_rng[1], after_numpy[1]) and numpy_rng[2:] == after_numpy[2:]
    assert python_rng == random.getstate()
    print('PASS: CPU save/export, unique paths, failure preserves checkpoint, non-S10 untouched, invalid shape rejected; weights/modes/devices/optimizers/RNG unchanged; PPO updates=0')


if __name__ == '__main__':
    main()
