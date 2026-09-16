"""S10 HIM ONNX export and numerical verification, without Isaac Gym."""
import argparse
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from rsl_rl.modules import HIMActorCritic


DOF_NAMES = [f'{leg}_{joint}_joint' for leg in ('fl', 'fr', 'hl', 'hr')
             for joint in ('hipx', 'hipy', 'knee', 'wheel')]


def policy_metadata(env):
    """The existing JIT sidecar, also captured with new checkpoints."""
    return dict(
        robot=env.cfg.asset.name, interface_version=2, reset_history='zero',
        dof_names=list(env.dof_names), wheel_indices=env.wheel_indices.tolist(),
        default_dof_pos=env.default_dof_pos[0].tolist(),
        p_gains=env.p_gains.tolist(), d_gains=env.d_gains.tolist(),
        action_scale=env.action_scale.tolist(), vel_scale=env.cfg.control.vel_scale,
        torque_limits=env.torque_limits.tolist(), dof_vel_limits=env.dof_vel_limits.tolist(),
        commands_scale=env.commands_scale.tolist(),
        obs_scales={k: getattr(env.obs_scales, k) for k in dir(env.obs_scales) if not k.startswith('_')},
        clip_actions=env.cfg.normalization.clip_actions,
        clip_observations=env.cfg.normalization.clip_observations,
        sim_dt=env.sim_params.dt, decimation=env.cfg.control.decimation,
        self_collisions=env.cfg.asset.self_collisions, initial_position=env.cfg.init_state.pos)


def contract_metadata(metadata):
    metadata = copy.deepcopy(metadata)
    if (metadata['robot'] != 's10' or metadata['interface_version'] != 2
            or metadata['reset_history'] != 'zero' or metadata['dof_names'] != DOF_NAMES
            or metadata['wheel_indices'] != [3, 7, 11, 15]):
        raise ValueError('Expected S10 interface v2, zero history and HIM joint order')
    for name, width in [('default_dof_pos', 16), ('p_gains', 16), ('d_gains', 16),
                        ('action_scale', 16), ('torque_limits', 16), ('dof_vel_limits', 16),
                        ('commands_scale', 3), ('initial_position', 3)]:
        value = np.asarray(metadata[name])
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid metadata {name}: expected {width} finite values')
    for name in ('sim_dt', 'decimation', 'clip_actions', 'clip_observations', 'vel_scale'):
        if not np.isfinite(metadata[name]) or metadata[name] <= 0:
            raise ValueError(f'Invalid metadata {name}')
    metadata.update(onnx_contract_version=1, policy_type='s10_him', input_name='obs',
                    input_shape=[1, 342], output_name='actions', output_shape=[1, 16],
                    one_step_observation_dim=57, history_length=6, history_order='newest_first',
                    policy_dt=metadata['sim_dt'] * metadata['decimation'])
    return metadata


def legacy_export_config(saved, metadata):
    """Validate an evaluated sidecar against the checkpoint's saved config.

    Limits came from Isaac Gym's imported asset and are absent from old config.json.
    They must be supplied from the matching evaluated policy.json, never new defaults.
    """
    cfg, names = saved['env'], saved['dof_names']
    if names != DOF_NAMES or cfg['asset']['name'] != 's10' or saved['interface_version'] != 2:
        raise ValueError('Not a saved S10 interface v2 configuration')
    if saved['train']['runner']['policy_class_name'] != 'HIMActorCritic':
        raise ValueError('Expected HIMActorCritic')
    control, norm = cfg['control'], cfg['normalization']
    scales = norm['obs_scales']
    expected = dict(robot='s10', interface_version=2, reset_history='zero', dof_names=names,
                    wheel_indices=[3, 7, 11, 15],
                    default_dof_pos=[cfg['init_state']['default_joint_angles'][n] for n in names],
                    vel_scale=control['vel_scale'], commands_scale=[scales['lin_vel']] * 2 + [scales['ang_vel']],
                    obs_scales=scales, clip_actions=norm['clip_actions'], clip_observations=norm['clip_observations'],
                    sim_dt=cfg['sim']['dt'], decimation=control['decimation'],
                    self_collisions=cfg['asset']['self_collisions'], initial_position=cfg['init_state']['pos'])
    for field, gain in [('p_gains', 'stiffness'), ('d_gains', 'damping')]:
        # Same last-matching substring rule as LeggedRobot._init_buffers.
        expected[field] = [[v for k, v in control[gain].items() if k in n][-1] for n in names]
    scale = control['action_scale']
    expected['action_scale'] = [scale[n] for n in names] if isinstance(scale, dict) else [scale] * 16
    for name, value in expected.items():
        actual = metadata[name]
        if isinstance(value, dict):
            matches = actual.keys() == value.keys() and all(np.isclose(actual[k], v) for k, v in value.items())
        elif isinstance(value, str) or name == 'dof_names':
            matches = actual == value
        else:
            matches = np.asarray(actual).shape == np.asarray(value).shape and np.allclose(actual, value, rtol=1e-6, atol=1e-7)
        if not matches:
            raise ValueError(f'Sidecar disagrees with saved training configuration: {name}')
    metadata = dict(metadata, **expected)
    dims = cfg['env']
    return dict(metadata=metadata, policy=saved['train']['policy'],
                dimensions=[dims['num_observations'], dims['num_privileged_obs'],
                            dims['num_one_step_observations'], dims['num_actions']])


class HIMOnnxPolicy(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.encoder = copy.deepcopy(actor_critic.estimator.encoder).cpu().eval()
        self.actor = copy.deepcopy(actor_critic.actor).cpu().eval()

    def forward(self, obs):
        parts = self.encoder(obs)
        velocity, latent = parts[..., :3], parts[..., 3:]
        latent = F.normalize(latent, dim=-1, p=2)
        return self.actor(torch.cat((obs[:, :57], velocity, latent), dim=-1))


def verification_samples(metadata, observations=None):
    initial = np.zeros((1, 342), np.float32)
    initial[0, 5] = -1.  # Current upright frame; five older frames and last action are zero.
    samples = [('zero', np.zeros_like(initial)), ('initial_current_frame_zero_history', initial)]
    rng = np.random.default_rng(0)  # Does not touch training RNG state.
    for i in range(32):
        frames = rng.normal(0, .5, (6, 57)).astype(np.float32)
        frames[:, 5] = -1.
        frames[:, 9 + np.array(metadata['wheel_indices'])] = 0.
        frames = np.clip(frames, -metadata['clip_observations'], metadata['clip_observations'])
        samples.append((f'distinct_history_{i}', frames.reshape(1, 342)))
    if observations is not None:
        recorded = np.load(observations, allow_pickle=False)
        if recorded.dtype != np.float32 or recorded.ndim != 2 or recorded.shape[1] != 342 or not len(recorded):
            raise ValueError('Recorded observations must be a nonempty float32 .npy array [N,342]')
        samples.extend((f'recorded_{i}', row[None]) for i, row in enumerate(recorded))
    return samples


def verify_onnx(path, actor_critic, metadata, observations=None, reference_jit=None):
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    for nodes, name, shape in [(inputs, 'obs', [1, 342]), (outputs, 'actions', [1, 16])]:
        if len(nodes) != 1 or (nodes[0].name, nodes[0].type, nodes[0].shape) != (name, 'tensor(float)', shape):
            raise ValueError(f'Invalid ONNX interface: expected {name} float32 {shape}')
    jit = torch.jit.load(str(reference_jit), map_location='cpu').eval() if reference_jit else None
    rows = []
    with torch.inference_mode():
        for name, obs in verification_samples(metadata, observations):
            if not np.isfinite(obs).all():
                raise ValueError(f'Non-finite input: {name}')
            expected = actor_critic.act_inference(torch.from_numpy(obs)).numpy()
            actual = session.run(['actions'], {'obs': obs})[0]
            if actual.dtype != np.float32 or actual.shape != (1, 16) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError(f'Invalid/non-finite output: {name}')
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4, err_msg=name)
            delta = np.abs(actual - expected)
            row = dict(sample=name, max_abs_error=float(delta.max()),
                       max_rel_error=float((delta / np.maximum(np.abs(expected), 1e-8)).max()))
            if jit is not None:
                jit_actions = jit(torch.from_numpy(obs[0])).numpy()[None]
                np.testing.assert_allclose(jit_actions, expected, atol=1e-5, rtol=1e-4, err_msg=f'JIT {name}')
                row['jit_max_abs_error'] = float(np.abs(jit_actions - expected).max())
            rows.append(row)
    return dict(passed=True, input_name='obs', input_shape=[1, 342], input_type='float32',
                output_name='actions', output_shape=[1, 16], output_type='float32',
                reference='HIMActorCritic.act_inference (checkpoint weights, CPU FP32)',
                reference_jit=str(Path(reference_jit).resolve()) if reference_jit else None,
                recorded_observations=str(Path(observations).resolve()) if observations else None,
                samples=rows, sample_count=len(rows), max_abs_error=max(r['max_abs_error'] for r in rows),
                max_rel_error=max(r['max_rel_error'] for r in rows), relative_error_denominator_floor=1e-8,
                atol=1e-5, rtol=1e-4, torch_version=torch.__version__, onnx_version=onnx.__version__,
                onnxruntime_version=ort.__version__, providers=session.get_providers(), opset=17)


def export_checkpoint(checkpoint, output=None, config=None, metadata=None, observations=None, reference_jit=None):
    checkpoint = Path(checkpoint).resolve()
    output = Path(output) if output else checkpoint.parent / 'onnx' / checkpoint.stem
    # Every module is rebuilt on CPU from the saved snapshot; no live training module is touched.
    with torch.random.fork_rng(devices=[]):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        spec = saved.get('s10_export')
        config_path = Path(config) if config else checkpoint.parent / 'config.json'
        if spec is None:
            if metadata is None:
                raise ValueError('Legacy checkpoint requires --metadata from its evaluated policy.json (asset limits)')
            spec = legacy_export_config(json.loads(config_path.read_text(encoding='utf-8')),
                                        json.loads(Path(metadata).read_text(encoding='utf-8')))
        elif config is not None or metadata is not None:
            raise ValueError('Checkpoint has embedded export configuration; do not override it')
        if spec['dimensions'][::2] != [342, 57] or spec['dimensions'][3] != 16:
            raise ValueError('S10 HIM requires 342 observations, 57 per frame and 16 actions')
        sidecar = contract_metadata(spec['metadata'])
        with redirect_stdout(io.StringIO()):
            actor_critic = HIMActorCritic(*spec['dimensions'], **spec['policy']).cpu().eval()
        actor_critic.load_state_dict(saved['model_state_dict'], strict=True)
        model = HIMOnnxPolicy(actor_critic).eval()
        output.mkdir(parents=True, exist_ok=True)
        # Publish only verified files. An exception leaves the checkpoint and prior export intact.
        with tempfile.TemporaryDirectory(prefix='.export-', dir=output) as temporary:
            temp = Path(temporary)
            torch.onnx.export(model, torch.zeros(1, 342), str(temp / 'policy.onnx'),
                              input_names=['obs'], output_names=['actions'], opset_version=17)
            report = verify_onnx(temp / 'policy.onnx', actor_critic, sidecar, observations, reference_jit)
            report.update(checkpoint=str(checkpoint), checkpoint_iteration=saved['iter'],
                          configuration_source='checkpoint:s10_export' if 's10_export' in saved else str(config_path.resolve()),
                          asset_limits_source='checkpoint:s10_export' if 's10_export' in saved else str(Path(metadata).resolve()))
            (temp / 'policy.json').write_text(json.dumps(sidecar, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            (temp / 'export_verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            for name in ('policy.onnx', 'policy.json', 'export_verification.json'):
                (temp / name).replace(output / name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: checkpoint_parent/onnx/checkpoint_stem/')
    parser.add_argument('--config', type=Path, help='Legacy saved training config.json, defaults beside checkpoint')
    parser.add_argument('--metadata', type=Path, help='Legacy matching evaluated policy.json; checked against saved config')
    parser.add_argument('--reference-jit', type=Path, help='Also compare the already evaluated 1-D TorchScript')
    parser.add_argument('--observations', type=Path, help='Optional captured float32 .npy [N,342]')
    report = export_checkpoint(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
