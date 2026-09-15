"""HIM S10 TorchScript runner. Commands are physical vx, vy, yaw-rate.

python deploy/s10_mujoco.py --policy /path/policy.pt --seconds 10
The matching policy.json is produced by play/export. No SDK or ROS connection.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def observation(cfg, omega, gravity, command, q, dq, last_action):
    error = np.asarray(q) - cfg['default_dof_pos']
    error[cfg['wheel_indices']] = 0.
    scales = cfg['obs_scales']
    frame = np.concatenate((np.asarray(omega) * scales['ang_vel'], gravity,
                            np.asarray(command) * cfg['commands_scale'],
                            error * scales['dof_pos'], np.asarray(dq) * scales['dof_vel'],
                            last_action))
    return np.clip(frame, -cfg['clip_observations'], cfg['clip_observations']).astype(np.float32)


def targets(cfg, action):
    action = np.clip(action, -cfg['clip_actions'], cfg['clip_actions'])
    q = np.asarray(cfg['default_dof_pos']) + action * cfg['action_scale']
    q[cfg['wheel_indices']] = 0.
    dq = np.zeros_like(q)
    dq[cfg['wheel_indices']] = action[cfg['wheel_indices']] * cfg['vel_scale']
    return q, dq


def load_model(cfg):
    import mujoco
    xml = Path(__file__).resolve().parents[1] / 'resources/robots/s10/mjcf/S10.xml'
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = cfg['sim_dt']
    names = cfg['dof_names']
    assert len(names) == len(set(names)) == 16
    joints = np.array([model.joint(name).id for name in names])
    motors = np.array([model.actuator(name).id for name in names])
    assert np.array_equal(model.actuator_trnid[motors, 0], joints)
    if cfg['self_collisions'] == 1:
        # Separate robot and world collision bits, preserving robot/ground contact.
        robot = (model.geom_bodyid != 0) & ((model.geom_contype | model.geom_conaffinity) != 0)
        world = model.geom_bodyid == 0
        model.geom_contype[robot] = 2
        model.geom_conaffinity[robot] = 1
        model.geom_contype[world] = 1
        model.geom_conaffinity[world] = 2
    data = mujoco.MjData(model)
    data.qpos[:3] = cfg['initial_position']
    data.qpos[model.jnt_qposadr[joints]] = cfg['default_dof_pos']
    mujoco.mj_forward(model, data)
    return model, data, joints, motors


def main():
    import mujoco
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--command', nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.policy.with_suffix('.json').read_text())
    assert cfg['robot'] == 's10' and cfg['reset_history'] == 'zero'
    policy = torch.jit.load(str(args.policy), map_location='cpu').eval()
    model, data, joints, motors = load_model(cfg)
    q_ids, v_ids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    base_id = model.body('base_link').id
    history = np.zeros((6, 57), np.float32)
    action = np.zeros(16, np.float32)
    policy_steps, physics_steps = 0, 0
    kp, kd, limits = map(np.asarray, (cfg['p_gains'], cfg['d_gains'], cfg['torque_limits']))
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
    try:
        with torch.inference_mode():
            while policy_steps < round(args.seconds / (model.opt.timestep * cfg['decimation'])) and (viewer is None or viewer.is_running()):
                start = time.monotonic()
                mujoco.mj_forward(model, data)  # Match derived body state to the latest integrated q/dq.
                velocity = np.empty(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                        base_id, velocity, 1)
                gravity = data.xmat[base_id].reshape(3, 3).T @ np.array([0., 0., -1.])
                history[1:] = history[:-1].copy()
                history[0] = observation(cfg, velocity[:3], gravity, args.command,
                                         data.qpos[q_ids], data.qvel[v_ids], action)
                action = policy(torch.from_numpy(history.ravel())).numpy()
                policy_steps += 1
                assert np.isfinite(action).all(), 'Non-finite policy output'
                action = np.clip(action, -cfg['clip_actions'], cfg['clip_actions'])
                q_target, v_target = targets(cfg, action)
                for _ in range(cfg['decimation']):
                    data.ctrl[motors] = np.clip(kp * (q_target - data.qpos[q_ids]) +
                                                kd * (v_target - data.qvel[v_ids]), -limits, limits)
                    mujoco.mj_step(model, data)
                    physics_steps += 1
                assert np.isfinite(data.qpos).all(), 'Non-finite simulation state'
                if viewer:
                    viewer.sync()
                    time.sleep(max(0., cfg['sim_dt'] * cfg['decimation'] - (time.monotonic() - start)))
        print(json.dumps(dict(sim_time=data.time, base_position=data.qpos[:3].tolist(),
                              warnings=data.warning.number.tolist(), ppo_updates=0,
                              physics_steps=physics_steps, policy_steps=policy_steps,
                              pd_dt=model.opt.timestep)))
    finally:
        if viewer:
            viewer.close()


if __name__ == '__main__':
    main()
