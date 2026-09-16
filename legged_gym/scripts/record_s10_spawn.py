"""Short, slowed camera examples of the original S10 reset; no learning updates."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import isaacgym
from isaacgym import gymapi
import numpy as np
import torch
from PIL import Image
from legged_gym.envs import LeggedRobot
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import set_seed
from legged_gym.scripts.evaluate_s10 import restore_config
from legged_gym.scripts.evaluate_s10_follow import Video
from legged_gym.utils.curriculum import TERRAINS, MODES
from rsl_rl.modules import HIMActorCritic


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    assert args.task == 's10' and args.headless and args.num_envs == 22 and args.seed == 1
    saved = json.loads((options.checkpoint_path.parent / 'config.json').read_text())
    checkpoint = torch.load(options.checkpoint_path, map_location='cpu')
    cfg, _ = task_registry.get_cfgs('s10')
    restore_config(cfg, saved['env'])
    cfg.env.num_envs = 22  # One robot per terrain column avoids overlapping robots in the camera.
    original_create = LeggedRobot.create_sim
    def create_sim(env):
        env.graphics_device_id = env.sim_device_id
        original_create(env)
    with patch.object(LeggedRobot, 'create_sim', create_sim):
        env, cfg = task_registry.make_env('s10', args=args, env_cfg=cfg)
    info = checkpoint['infos']
    selected = []
    for column in env.terrain_types.cpu():
        candidates = (info['terrain_types'] == column).nonzero(as_tuple=False).flatten()
        selected.append(int(candidates[info['terrain_levels'][candidates].argmin()]))
    env.terrain_levels.copy_(info['terrain_levels'][selected])
    env.extended_speed_envs.copy_(info['extended_speed_envs'][selected])
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    env.task_curriculum.speed_limits.copy_(info['curriculum']['speed_limits'])
    actor = HIMActorCritic(env.num_obs, env.num_privileged_obs, env.num_one_step_obs,
                          env.num_actions, **saved['train']['policy']).to(env.device).eval()
    actor.load_state_dict(checkpoint['model_state_dict'])
    set_seed(1)
    env.init_done = False
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.init_done = True
    env.compute_observations()
    obs = env.get_observations()
    options.output.mkdir(parents=True, exist_ok=False)
    root = env.root_states.cpu().clone()
    local = root[:, :3] - env.env_origins.cpu()
    course = env.task_curriculum
    pool = []
    for i in range(env.num_envs):
        pool.append(dict(env=i, terrain=TERRAINS[int(course.kinds[i])], level=int(env.terrain_levels[i]),
            spawn='terrain' if abs(float(local[i, 0])) > 1.5 else 'platform',
            command=env.commands[i, :3].tolist(), mode=MODES[int(course.mode[i])],
            root=root[i].tolist(), local_position=local[i].tolist(), dof_pos=env.dof_pos[i].tolist()))
    clips = []
    for terrain, spawn in [('stairs_up', 'platform'), ('stairs_up', 'terrain'), ('stairs_down', 'terrain')]:
        candidates = [p for p in pool if p['terrain'] == terrain and p['spawn'] == spawn]
        assert candidates, (terrain, spawn, pool)
        # Prefer a naturally sampled parking command to make initial settling visible.
        pick = min(candidates, key=lambda p: (p['mode'] != 'parking', p['env']))
        clips.append(dict(pick, name=f'{terrain}_{spawn}', end=None, frame=None, trace=[]))
    camera_params = gymapi.CameraProperties()
    camera_params.width, camera_params.height, camera_params.horizontal_fov = 960, 640, 60
    writers = []
    for clip in clips:
        i = clip['env']
        clip['camera'] = env.gym.create_camera_sensor(env.envs[i], camera_params)
        assert clip['camera'] >= 0
        writer = Video(options.output / (clip['name'] + '.mp4'), clip['name'],
                       'S10 model1500 | training reset | 0.5x', fps=25)
        writers.append(writer)
    original_check = env.check_termination
    tick = 0
    def check():
        original_check()
        # Capture before automatic reset so a fall or boundary exit cannot be hidden.
        for clip in clips:
            if clip['end'] is None:
                p = env.root_states[clip['env'], :3].cpu().numpy()
                env.gym.set_camera_location(clip['camera'], env.envs[clip['env']],
                    gymapi.Vec3(*(p + [-1.5, -1.7, .75])), gymapi.Vec3(*(p + [0, 0, -.08])))
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        for clip, writer in zip(clips, writers):
            i = clip['env']
            if clip['end'] is None:
                rgb = env.gym.get_camera_image(env.sim, env.envs[i], clip['camera'], gymapi.IMAGE_COLOR).reshape(640, 960, 4)[:, :, :3]
                clip['frame'] = rgb.copy()
                if env.reset_buf[i]:
                    hit = (env.contact_forces[i, env.termination_contact_indices].norm(dim=1) > 1).any()
                    clip['end'] = dict(seconds=tick * env.dt, reason='base_contact' if hit else 'timeout_or_boundary')
                velocity = torch.cat((env.base_lin_vel[i, :2], env.base_ang_vel[i, 2:3])).tolist()
                tilt = float(torch.rad2deg(torch.acos((-env.projected_gravity[i, 2]).clamp(-1, 1))))
                clip['trace'].append(dict(seconds=tick * env.dt, velocity=velocity, tilt=tilt,
                    loaded_wheels=int((env.contact_forces[i, env.feet_indices].norm(dim=1) > 1).sum())))
                if tick in [1, 10, 50]:
                    Image.fromarray(clip['frame']).save(options.output / f"{clip['name']}_{tick:03d}.png")
            if clip['end']:
                writer.source = f"S10 model1500 | FROZEN after {clip['end']['reason']} at {clip['end']['seconds']:.2f}s"
            writer.frame(clip['frame'], tick * env.dt, clip['command'], clip['trace'][-1]['velocity'])
    env.check_termination = check
    try:
        with torch.inference_mode():
            for tick in range(1, 301):
                obs, _, _, _, _, _, _ = env.step(actor.act(obs))
        assert all(torch.equal(v.cpu(), checkpoint['model_state_dict'][k]) for k, v in actor.state_dict().items())
        (options.output / 'recording.json').write_text(json.dumps(dict(iteration=checkpoint['iter'], ppo_updates=0,
            policy='stochastic training policy', preview_envs=22, simulation_seconds=300 * env.dt, playback_speed=.5,
            selection='One robot per column, minimum saved level in each column; natural original reset. Prefer existing parking command. Illustrative samples, not replay of earlier diagnostic trajectories.',
            clips=[{k:v for k,v in c.items() if k not in ['frame','camera']} for c in clips]), indent=2, allow_nan=False)+'\n')
        print('SPAWN_VIDEOS_COMPLETE_ZERO_UPDATES', flush=True)
    finally:
        for writer in writers:
            writer.close()
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
