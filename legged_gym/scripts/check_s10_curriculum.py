"""Run with Python + torch; no simulator and no policy updates."""
import importlib.util
from pathlib import Path
import torch
import numpy as np
from types import SimpleNamespace
import ast
import inspect

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('curriculum', ROOT / 'legged_gym/utils/curriculum.py')
curriculum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curriculum)


def main():
    torch.manual_seed(1)
    columns = torch.arange(4096) * 22 // 4096
    mask = curriculum.stratified_extended_mask(columns)
    for column in columns.unique():
        group = columns == column
        assert abs(float(mask[group].float().mean()) - .2) < .006
    assert mask[columns >= 20].any() and mask[columns < 4].any()
    assert not torch.equal(mask, torch.arange(4096) < 819)
    print('Every terrain column has randomly assigned approximately 20% extended-speed environments.')
    spec = importlib.util.spec_from_file_location('pebbles', ROOT / 'legged_gym/utils/pebbles.py')
    pebbles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pebbles)
    raw = pebbles.pebble_heightfield(8., 8., .025, .001, .08, 5., np.random.default_rng(2))
    assert raw.shape == (321, 321) and 40 < raw.max() <= 80
    assert (raw[100:221, 100:221] == 0).all()  # 3 m spawn platform.
    assert not raw[[0, -1], :].any() and not raw[:, [0, -1]].any()
    assert (raw > 0).sum() > 5000
    assert np.unique(raw).size > 30  # Curved caps, not binary-height blocks.
    print('Pebble caps have curved collision heights, a clear spawn platform and seamless zero boundaries.')
    scope = dict(inspect=inspect)
    for relative, names in [('envs/base/base_config.py', ['BaseConfig']),
        ('envs/base/legged_robot_config.py', ['LeggedRobotCfg', 'LeggedRobotCfgPPO']),
        ('envs/go2w/go2w_config.py', ['GO2WRoughCfg', 'GO2WRoughCfgPPO']),
        ('envs/s10/s10_config.py', ['S10RoughCfg', 'S10RoughCfgPPO'])]:
        path = ROOT / 'legged_gym' / relative
        nodes = [x for x in ast.parse(path.read_text()).body if isinstance(x, ast.ClassDef) and x.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    cfg = scope['S10RoughCfg']()
    n = 4096
    env = SimpleNamespace(cfg=cfg, num_envs=n, device='cpu', dt=.02, common_step_counter=1,
        terrain_types=columns, terrain_levels=torch.ones(n, dtype=torch.long),
        terrain=SimpleNamespace(column_categories=[0, 0, 1, 1]+[2]*7+[3]*4+[4]*5+[5]*2, env_length=8.),
        commands=torch.zeros(n, 4), parking_commands=torch.zeros(n, dtype=torch.bool), extended_speed_envs=mask,
        base_lin_vel=torch.zeros(n, 3), base_ang_vel=torch.zeros(n, 3),
        root_states=torch.zeros(n, 13), env_origins=torch.zeros(n, 3), max_episode_length_s=20.)
    course = curriculum.TaskCurriculum(env)
    ids = torch.arange(n)
    course.sample(ids)
    for mode in range(8):
        selected = course.mode == mode
        assert selected.any()
        commands = env.commands[selected, :3]
        if mode < 6:
            axis = mode // 2
            assert (commands[:, axis].abs() >= cfg.commands.minimum_speed[axis]).all()
            assert (commands[:, [i for i in range(3) if i != axis]] == 0).all()
        elif mode == 7:
            assert not commands.any()
    assert (env.commands[(course.mode == 2) | (course.mode == 3), 1].abs() < .2).any()
    # Perfect pure yaw can progress without net translation; wrong uncommanded axes cannot.
    course.mode[:] = 4
    env.commands[:, :3] = torch.tensor([0., 0., .6])
    env.base_ang_vel[:, 2] = .6
    target = torch.ones(n, dtype=torch.bool)
    for _ in range(205):
        course.record(target, target)
    course.terrain_ready[:] = True
    delta = course.finish(ids, torch.zeros(n, dtype=torch.bool))
    assert (delta == 1).all()
    # Same command, unwanted sideways drift: normal-speed episodes must demote.
    env.base_lin_vel[:, 1] = .5
    for _ in range(205):
        course.record(target, target)
    assert (course.finish(ids, torch.zeros(n, dtype=torch.bool))[~mask] == -1).all()
    env.base_lin_vel.zero_()
    for _ in range(205):
        course.record(~target, target)
    assert not (course.finish(ids, torch.zeros(n, dtype=torch.bool)) > 0).any()
    # Single-mode evidence never certifies every terrain task. Each speed axis is independent.
    course.terrain_seconds.zero_(); course.terrain_good.zero_()
    course.terrain_seconds[:, :, 0] = 100
    course.terrain_good[:, :, 0] = 100
    course.edge_seconds.zero_(); course.edge_good.zero_()
    course.edge_seconds[4, 1] = course.edge_good[4, 1] = 100
    previous = course.speed_limits.clone()
    course.advance_window()
    assert not course.terrain_ready.any()
    expected = previous.clone(); expected[4, 1] += cfg.commands.speed_increments[1]
    torch.testing.assert_close(course.speed_limits, expected)
    # A short fall fails even before there was enough tracking opportunity.
    assert (course.finish(ids[:1], torch.ones(1, dtype=torch.bool)) == -1).all()
    print('Mode isolation, low-speed sideways, yaw promotion, cross-axis drift, terrain exposure, failure and independent speed progression passed.')


if __name__ == '__main__':
    main()
