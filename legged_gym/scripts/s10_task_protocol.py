"""Independent terrain/task probes; every probe starts afresh on the surface."""
import numpy as np
import s10_eval_protocol as legacy
import importlib.util
from pathlib import Path

# Loading pure geometry must not import the Gym package (or torch before isaacgym).
_spec = importlib.util.spec_from_file_location('pebbles', Path(__file__).parents[1] / 'utils/pebbles.py')
_pebbles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pebbles)
pebble_heightfield = _pebbles.pebble_heightfield

VERSION = 's10-terrain-task-v2'
HS, VS, EXTENT = .025, .001, 12.
SCENES = legacy.SCENES + ['pebbles_2cm', 'pebbles_6cm']
PROBES = dict(forward=[.5, 0, 0], reverse=[-.5, 0, 0], left=[0, .3, 0], right=[0, -.3, 0],
    yaw_left=[0, 0, .6], yaw_right=[0, 0, -.6], mixed=[.5, .2, .5], parking=[0, 0, 0],
    fast_forward=[1.25, 0, 0], fast_reverse=[-1.25, 0, 0], fast_left=[0, .75, 0],
    fast_right=[0, -.75, 0], fast_yaw_left=[0, 0, 1.25], fast_yaw_right=[0, 0, -1.25])
PROBE, SEED = 'forward', 701
write_json = legacy.write_json


def schedule(scene):
    return [(0, 3, [0, 0, 0]), (3, 11, PROBES[PROBE]), (11, 17, [0, 0, 0])]


def command(t, scene):
    return next((v for a, b, v in schedule(scene) if a <= t < b), [0, 0, 0])


def geometry(scene):
    x = np.arange(-EXTENT, EXTENT + HS/2, HS)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    h = np.zeros_like(xx)
    if scene.startswith('rough'):
        amplitude = .01 if '1cm' in scene else .02
        coarse = np.random.default_rng(SEED).uniform(-amplitude, amplitude, (len(x)//8+1, len(x)//8+1))
        h = np.repeat(np.repeat(coarse, 8, 0), 8, 1)[:len(x), :len(x)]
    elif scene.startswith('slope'):
        h = xx*np.tan(np.deg2rad(8))*(1 if '_up_' in scene else -1)
    elif scene.startswith('stairs'):
        height = .05 if '5cm' in scene else .08
        h = np.floor((xx+.15)/.3)*height*(1 if '_up_' in scene else -1)
    elif scene.startswith('obstacles'):
        height = .05 if '5cm' in scene else .08
        h[(np.mod(xx+.15, 1.2) < .3) & (np.mod(yy+.4, 1.2) < .8)] = height
    elif scene.startswith('pebbles'):
        height = .02 if '2cm' in scene else .06
        h = pebble_heightfield(2*EXTENT, 2*EXTENT, HS, VS, height, 5.,
                               np.random.default_rng(SEED), platform_size=0) * VS
    raw = np.rint(h/VS).astype(np.int16)
    # A loaded wheel must meet the actual sparse obstacle, not just the named map.
    active = raw != 0 if scene.startswith(('obstacles', 'pebbles')) else np.ones_like(raw, dtype=bool)
    if scene == 'flat':
        active[:] = False
    meta = dict(protocol=VERSION, scene=scene, probe=PROBE, seed=SEED, horizontal_scale=HS,
        vertical_scale=VS, bounds_xy=[-EXTENT, EXTENT], height_range=[float(h.min()), float(h.max())],
        schedule=schedule(scene), spawn='on target surface; no approach platform',
        geometry='fixed mesh; continuous slopes/stairs across the map; unshifted pebble caps',
        initial_yaw_degrees=[0, 90, 180], target_contact_definition='loaded wheel center over an active geometry cell')
    return raw, active, meta


def ground(raw, xy):
    return legacy.ground(raw, xy, HS, VS, EXTENT)


def mesh_ground(vertices, xy):
    return legacy.mesh_ground(vertices, xy, HS, EXTENT)


def summarize(rows, scene, end_reason):
    t, vel = rows['time'], rows['velocity']
    result = dict(protocol=VERSION, scene=scene, probe=PROBE, seed=SEED, end_reason=end_reason,
        duration=float(t[-1]), segments=[],
        command_observation_max_error=float(np.abs(rows['command']-rows['observed_command']).max()),
        max_tilt_degrees=float(np.rad2deg(rows['tilt']).max()),
        base_contact_max_N=float(rows['base_contact'].max()),
        clearance_range=[float(rows['clearance'].min()), float(rows['clearance'].max())],
        pd_saturation_fraction=rows['saturated'].mean(0).tolist())
    for start, end, target in schedule(scene):
        mask = (t > start+1e-8) & (t <= end+1e-8)
        if not mask.any():
            result['segments'].append(dict(start=start, end=end, status='not_reached', passed=False))
            continue
        target = np.asarray(target)
        indices = np.flatnonzero(mask)
        v, times = vel[mask], t[mask]
        error = v-target
        steady = times > start+1.
        tolerance = np.array([.05, .05, .1])+.1*abs(target) if target.any() else np.array([.01, .01, .02])
        good = (abs(error) <= tolerance).all(axis=1)
        streak = np.convolve(good.astype(int), np.ones(400, dtype=int), 'valid') if len(v) >= 400 else []
        held = np.flatnonzero(np.asarray(streak) == 400)
        pos = rows['position'][mask, :2]
        heading = np.unwrap(rows['heading'][mask])
        before = max(0, indices[0]-1)
        origin = rows['position'][before, :2]
        target_time = rows['target_contact'][mask].astype(bool)
        eligible = steady & (target_time if scene != 'flat' else True)
        coverage = float(eligible.sum()*.0025)
        score = float(good[eligible].mean()) if eligible.any() else None
        complete = times[-1] >= end-.003
        seg = dict(start=start, end=end, command=target.tolist(), status='complete' if complete else 'interrupted',
            rmse=np.sqrt((error**2).mean(axis=0)).tolist(),
            target_surface_rmse=np.sqrt((error[eligible]**2).mean(axis=0)).tolist() if eligible.any() else None,
            tail_mean=v[times > max(start, times[-1]-2)].mean(axis=0).tolist(),
            tail_std=v[times > max(start, times[-1]-2)].std(axis=0).tolist(),
            target_surface_seconds=coverage, target_success_fraction=score,
            time_to_all_axes_held_1s=float(times[held[0]]-start) if len(held) else None,
            position_displacement=float(np.linalg.norm(pos[-1]-origin)),
            path_length=float(np.linalg.norm(np.diff(np.vstack((origin, pos)), axis=0), axis=1).sum()),
            heading_net_change=float(heading[-1]-heading[0]),
            heading_path=float(abs(np.diff(heading)).sum()),
            passed=bool(complete and coverage >= 2 and score is not None and score >= .9),
            evidence='sufficient' if coverage >= 2 else 'insufficient_target_contact')
        if not target.any() and steady.any():
            hold = pos[steady]
            hold_heading = heading[steady]
            seg['holding_displacement'] = float(np.linalg.norm(hold[-1]-hold[0]))
            seg['holding_path'] = float(np.linalg.norm(np.diff(hold, axis=0), axis=1).sum())
            seg['holding_heading_path'] = float(abs(np.diff(hold_heading)).sum())
            seg['braking_distance_1s'] = float(np.linalg.norm(pos[np.flatnonzero(~steady)[-1]]-origin)) if (~steady).any() else None
            seg['wheel_velocity_rms'] = np.sqrt((rows['qvel'][mask][steady][:, [9, 13, 17, 21]]**2).mean(axis=0)).tolist()
            seg['passed'] &= seg['holding_path'] <= .05 and seg['holding_heading_path'] <= np.deg2rad(3)
        result['segments'].append(seg)
    return result


def self_check():
    raw, active, _ = geometry('pebbles_6cm')
    assert raw.shape == (961, 961) and raw.max() <= 60 and active.any()
    for scene in ['slope_up_8deg', 'slope_down_8deg', 'stairs_up_8cm', 'stairs_down_8cm']:
        raw, active, _ = geometry(scene)
        assert active[480, 480]
        sign = 1 if '_up_' in scene else -1
        assert sign*(ground(raw, np.array([[1., 0.]]))-ground(raw, np.array([[-1., 0.]])))[0] > 0
    n = 6800
    rows = dict(time=np.arange(1, n+1)*.0025, velocity=np.zeros((n, 3)), command=np.zeros((n, 3)),
        observed_command=np.zeros((n, 3)), tilt=np.zeros(n), base_contact=np.zeros(n), clearance=np.full(n, .425),
        saturated=np.zeros((n, 16)), position=np.zeros((n, 3)), heading=np.zeros(n),
        target_contact=np.zeros(n, bool), qvel=np.zeros((n, 22)))
    result = summarize(rows, 'pebbles_6cm', 'completed')
    assert not any(s['passed'] for s in result['segments'])
    assert result['segments'][1]['evidence'] == 'insufficient_target_contact'
    rows['target_contact'][:] = True
    rows['velocity'][1200:4400, 0] = .5
    assert summarize(rows, 'pebbles_6cm', 'completed')['segments'][1]['passed']
    rows['velocity'][1200:4400, 1] = .2
    assert not summarize(rows, 'pebbles_6cm', 'completed')['segments'][1]['passed']
    print('Task protocol geometry, target-contact evidence and uncommanded-axis regression checks passed.')


if __name__ == '__main__':
    self_check()
