"""CPU check of production wheel rewards and PD stepping; no Gym or training.

python legged_gym/scripts/check_s10_wheel_rewards.py
"""
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    root = Path(__file__).resolve().parents[2]
    # Reuse the repository's AST-based CPU checks to avoid importing the Gym binary.
    scope = dict(torch=torch, inspect=inspect, BaseTask=object,
                 SimpleNamespace=SimpleNamespace,
                 gymtorch=SimpleNamespace(unwrap_tensor=lambda tensor: tensor))
    for relative, names in [
        ('envs/base/base_config.py', ['BaseConfig']),
        ('envs/base/legged_robot_config.py', ['LeggedRobotCfg', 'LeggedRobotCfgPPO']),
        ('envs/go2w/go2w_config.py', ['GO2WRoughCfg', 'GO2WRoughCfgPPO']),
        ('envs/s10/s10_config.py', ['S10RoughCfg', 'S10RoughCfgPPO']),
        ('envs/base/legged_robot.py', ['LeggedRobot']),
        ('utils/helpers.py', ['class_to_dict']),
        ('scripts/evaluate_s10.py', ['restore_config']),
    ]:
        path = root / 'legged_gym' / relative
        tree = ast.parse(path.read_text(encoding='utf-8'))
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)

    cfg = scope['S10RoughCfg']()
    as_dict = scope['class_to_dict']
    scales = as_dict(cfg.rewards.scales)
    assert scales['action_rate'] == -.01 and scales['wheel_action_rate'] == -.02
    assert scales['wheel_torque_excess'] == -.1 and cfg.rewards.only_positive_rewards
    parking_scales = dict(parking_lin_vel=-.5, parking_ang_vel=-.25, parking_wheel_vel=-.002)
    assert {name: scales[name] for name in parking_scales} == parking_scales
    assert cfg.commands.parking_probability == .2
    assert cfg.commands.parking_thresholds == [.03, .03, .05]
    legacy = json.loads((root / 'legged_gym/scripts/configs/s10_resume_200.json').read_text())['env']
    restored = scope['S10RoughCfg']()
    scope['restore_config'](restored, legacy)
    assert as_dict(restored) == legacy  # Old recipes must not inherit the new rewards.
    assert getattr(restored.commands, 'parking_probability', 0.) == 0.
    assert vars(restored.commands.ranges) == legacy['commands']['ranges']
    restored = scope['S10RoughCfg']()
    scope['restore_config'](restored, as_dict(cfg))
    assert as_dict(restored) == as_dict(cfg)

    env = scope['LeggedRobot'].__new__(scope['LeggedRobot'])
    env.cfg, env.device, env.num_envs, env.num_actions = cfg, 'cpu', 3, 16
    env.dt = cfg.sim.dt * cfg.control.decimation
    env.wheel_indices = torch.tensor([3, 7, 11, 15])
    wheels = env.wheel_indices
    env.torque_limits = torch.tensor([50., 50., 50., 14.] * 4)
    env.default_dof_pos = env.dof_pos = torch.zeros(3, 16)
    env.dof_vel = torch.zeros(3, 16)
    env.last_actions = torch.zeros(3, 16)
    env.action_scale = torch.tensor([.125, .25, .25, 0.] * 4)
    env.p_gains, env.d_gains = torch.tensor([80., 80., 80., 0.] * 4), torch.tensor([2., 2., 2., .6] * 4)
    env.Kp_factors = env.Kd_factors = 1.
    env.torques = torch.zeros(3, 16)
    env.obs_buf, env.privileged_obs_buf = torch.zeros(3, 342), None
    env.rew_buf, env.reset_buf, env.extras = torch.zeros(3), torch.zeros(3), {}
    env.sim, env.render = None, lambda: None
    env.reward_scales = {name: scales[name] for name in ['wheel_action_rate', 'wheel_torque_excess']}
    env._prepare_reward_function()
    torch.testing.assert_close(env.wheel_torque_budget, torch.full((4,), 11.2))

    # Default parking terms are registered and penalize motion immediately at zero command.
    env.reward_scales = parking_scales.copy()
    env._prepare_reward_function()
    env.commands = torch.tensor([[0., 0., 0., 0.], [.03, .03, .05, 0.], [0., 0., .8, 0.]])
    env.base_lin_vel = torch.tensor([[.6, .8, 0.]] * 3)
    env.base_ang_vel = torch.tensor([[0., 0., 2.]] * 3)
    env.dof_vel[:, wheels] = 3.
    cfg.rewards.only_positive_rewards = False
    env.compute_reward()
    torch.testing.assert_close(env.rew_buf, torch.tensor([-1.006, -1.006, 0.]) * env.dt)
    env.dof_vel.zero_()
    env.reward_scales = {name: scales[name] for name in ['wheel_action_rate', 'wheel_torque_excess']}
    env._prepare_reward_function()

    env.actions = torch.ones(3, 16)
    env.last_actions[:, wheels] = -1.
    torch.testing.assert_close(env._reward_wheel_action_rate(), torch.full((3,), 16.))
    env.actions[:, 0] = 100.  # Leg changes must not enter the wheel-only penalty.
    torch.testing.assert_close(env._reward_wheel_action_rate(), torch.full((3,), 16.))
    env.actions[:, wheels] = env.last_actions[:, wheels]
    assert not env._reward_wheel_action_rate().any()
    env.last_actions.zero_()

    applied = []
    action = torch.zeros(3, 16)
    action[0, wheels], action[1, wheels], action[2, 0] = 8., -100. / 3., 100.
    # First substep requests +24/-100 Nm. Both clip to 14 Nm in magnitude.
    # Thereafter measured speed matches the target: the last-substep penalty is zero.
    def simulate():
        env.dof_vel[:, wheels] = action[:, wheels] * cfg.control.vel_scale

    env.gym = SimpleNamespace(set_dof_actuation_force_tensor=lambda sim, tau: applied.append(tau.clone()),
                              simulate=lambda sim: simulate(), fetch_results=lambda *args: None,
                              refresh_dof_state_tensor=lambda sim: None)

    def post_physics_step():
        env.compute_reward()
        return [], None

    env.post_physics_step = post_physics_step
    cfg.domain_rand.delay = False
    cfg.rewards.only_positive_rewards = False  # Expose the signed terms for this numerical check.
    for decimation in (4, 8):
        cfg.control.decimation = decimation
        env.dof_vel.zero_()
        applied.clear()
        env.step(action)
        expected = torch.tensor([4 * (24 / 11.2 - 1) ** 2,
                                 4 * (100 / 11.2 - 1) ** 2, 0.]) / decimation
        torch.testing.assert_close(env._reward_wheel_torque_excess(), expected)
        assert expected[1] > expected[0] > 0
        assert not env.raw_torques[:, wheels].any()
        assert len(applied) == decimation
        torch.testing.assert_close(applied[0][:, wheels], torch.tensor([[14.] * 4, [-14.] * 4, [0.] * 4]))
        assert all((tau.abs() <= env.torque_limits).all() for tau in applied)
        torch.testing.assert_close(env.rew_buf, env.dt * (
            scales['wheel_action_rate'] * env._reward_wheel_action_rate() + scales['wheel_torque_excess'] * expected))
        env.step(action)
        assert not env._reward_wheel_torque_excess().any()  # No leakage between policy steps.

    # Invalid physical budgets fail before stepping, instead of dividing by zero.
    for fraction in (0., -1., 1.1, float('nan')):
        cfg.rewards.soft_torque_limit = fraction
        env.reward_scales = {'wheel_torque_excess': -.1}
        try:
            env._prepare_reward_function()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid wheel torque budget accepted')

    # Legacy/Go2W recipes do not require the new buffers or budget setting.
    env.cfg = scope['GO2WRoughCfg']()
    assert 'wheel_action_rate' not in as_dict(env.cfg.rewards.scales)
    env.reward_scales = {'wheel_torque_excess': 0.}
    env._prepare_reward_function()
    del env.wheel_torque_excess, env.wheel_torque_budget
    env.cfg.domain_rand.delay = False
    env.step(torch.zeros(3, 16))
    print('PASS: wheel-only smoothing, pre-clip excess, all-substep averaging, per-step reset, '
          'hard torque clipping, dt weighting, config restore and disabled legacy path; 0 updates')


if __name__ == '__main__':
    main()
