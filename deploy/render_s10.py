"""Render an evaluate_s10.py MuJoCo trajectory using installed MuJoCo and ffmpeg."""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from s10_mujoco import load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True, help='Exported policy.json metadata')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trace = np.load(args.trace)
    model, data, _, _ = load_model(json.loads(args.policy.read_text()))
    assert trace['qpos'].shape[1] == model.nq and trace['qvel'].shape[1] == model.nv
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.2, 125., -22.
    visual = mujoco.MjvOption()
    visual.geomgroup[1] = 0  # Official XML uses group 1 for translucent collision shapes.
    width, height, stride = 960, 640, 2
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
               '-video_size', f'{width}x{height}', '-framerate', str(1. / (float(trace['dt']) * stride)),
               '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22',
               '-movflags', '+faststart', str(args.output)]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            for i in range(0, len(trace['qpos']), stride):
                data.qpos[:], data.qvel[:] = trace['qpos'][i], trace['qvel'][i]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3]
                camera.lookat[2] = .25
                renderer.update_scene(data, camera, scene_option=visual)
                process.stdin.write(renderer.render().tobytes())
        finally:
            process.stdin.close()
            return_code = process.wait()
        assert return_code == 0, return_code
    print(args.output)
