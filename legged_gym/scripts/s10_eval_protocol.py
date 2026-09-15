"""Fixed, zero-training S10 evaluation geometry, commands and measurements (v1)."""
import json
from pathlib import Path
import numpy as np

VERSION = 's10-follow-terrain-v1'
FLAT = [(0, 5, [0, 0, 0]), (5, 15, [.5, 0, 0]), (15, 25, [1, 0, 0]),
        (25, 35, [-.5, 0, 0]), (35, 40, [0, 0, 0]), (40, 50, [0, 0, .8]),
        (50, 60, [0, 0, -.8]), (60, 70, [.6, 0, .6]), (70, 80, [.6, 0, -.6]),
        (80, 90, [0, 0, 0])]
ROUGH = [(0, 5, [0, 0, 0]), (5, 20, [.3, 0, 0]), (20, 35, [.6, 0, 0]),
         (35, 50, [-.6, 0, 0]), (50, 54, [0, 0, .6]), (54, 58, [0, 0, -.6]),
         (58, 60, [0, 0, 0])]
SCENES = ['flat', 'rough_1cm', 'rough_2cm', 'slope_up_8deg', 'slope_down_8deg',
          'stairs_up_5cm', 'stairs_up_8cm', 'stairs_down_5cm', 'stairs_down_8cm',
          'obstacles_5cm', 'obstacles_8cm']
HS, VS, EXTENT = .1, .001, 40.


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def schedule(scene):
    return FLAT if scene == 'flat' else ROUGH


def command(t, scene):
    return next((v for a, b, v in schedule(scene) if a <= t < b), [0, 0, 0])


def geometry(scene):
    """80 m square, safe spawn, +x enters terrain; no neighboring terrain tiles."""
    assert scene in SCENES
    x = np.arange(-EXTENT, EXTENT + HS/2, HS)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    h = np.zeros_like(xx)
    active = xx >= 1.
    if scene.startswith('rough'):
        amplitude = .01 if scene == 'rough_1cm' else .02
        # Fixed 20 cm patches. Bounded heights, with a 10 cm triangulated transition.
        coarse = np.random.default_rng(123).integers(-round(amplitude/VS), round(amplitude/VS)+1,
                                                   (len(x)//2+1, len(x)//2+1))
        h = np.repeat(np.repeat(coarse, 2, 0), 2, 1)[:len(x), :len(x)] * VS
        h[~active] = 0
    elif scene.startswith('slope'):
        rise = np.clip(xx-1., 0., 9.) * np.tan(np.deg2rad(8.))
        h = rise if '_up_' in scene else 9*np.tan(np.deg2rad(8.))-rise
        active &= xx <= 10.
    elif scene.startswith('stairs'):
        height = .05 if '5cm' in scene else .08
        steps = np.clip(np.floor((xx-1.+1e-7)/.3)+1, 0, 12)
        h = height * (steps if '_up_' in scene else 12-steps)
        active &= xx <= 4.6
    elif scene.startswith('obstacles'):
        height = .05 if '5cm' in scene else .08
        active &= (np.mod(xx-1.+1e-7, 2.) < .3) & (np.mod(yy+1.5, 4.) < 3.)
        h[active] = height
    else:
        active[:] = False
    raw = np.rint(h/VS).astype(np.int16)
    meta = dict(protocol=VERSION, scene=scene, horizontal_scale=HS, vertical_scale=VS,
                bounds_xy=[-EXTENT, EXTENT], height_range=[float(raw.min()*VS), float(raw.max()*VS)],
                seed=123, target_entry_x=1., slope_direction='+x',
                geometry='heightfield triangles; slopes above .25 gradient are made vertical by Isaac Gym',
                steps=12 if scene.startswith('stairs') else None,
                step_width=.3 if scene.startswith('stairs') else None,
                schedule=schedule(scene))
    return raw, active, meta


def ground(raw, xy):
    """Piecewise triangular height matching convert_heightfield_to_trimesh(None)."""
    p = (np.asarray(xy)+EXTENT)/HS
    ij = np.clip(np.floor(p).astype(int), 0, np.array(raw.shape)-2)
    f = np.clip(p-ij, 0, 1)
    i, j = ij[..., 0], ij[..., 1]
    a, b, c, d = raw[i,j], raw[i+1,j], raw[i,j+1], raw[i+1,j+1]
    # Isaac Gym divides the cell along (i,j)--(i+1,j+1).
    return np.where(f[...,1] >= f[...,0], a+(d-c)*f[...,0]+(c-a)*f[...,1],
                    a+(b-a)*f[...,0]+(d-b)*f[...,1]) * VS


def mesh_ground(vertices, xy):
    """Query the actual shifted triangles around each cell, including vertical steps."""
    p=np.asarray(xy)
    ij=np.floor((p+EXTENT)/HS).astype(int)
    shifts=np.array([(i,j) for i in [-1,0,1] for j in [-1,0,1]])
    cells=np.clip(ij[:,None,:]+shifts,0,np.array(vertices.shape[:2])-2)
    i,j=cells[...,0],cells[...,1]
    a,b,c,d=vertices[i,j],vertices[i+1,j],vertices[i,j+1],vertices[i+1,j+1]
    triangles=np.stack([np.stack([a,d,c],axis=-2),np.stack([a,b,d],axis=-2)],axis=-3).reshape(len(p),-1,3,3)
    a,b,c=triangles[:,:,0],triangles[:,:,1],triangles[:,:,2]
    u,v,q=b-a,c-a,p[:,None,:]-a[:,:,:2]
    den=u[:,:,0]*v[:,:,1]-u[:,:,1]*v[:,:,0]
    safe=np.where(np.abs(den)>1e-12,den,1.)
    s=(q[:,:,0]*v[:,:,1]-q[:,:,1]*v[:,:,0])/safe
    t=(u[:,:,0]*q[:,:,1]-u[:,:,1]*q[:,:,0])/safe
    valid=(np.abs(den)>1e-12)&(s>=-1e-5)&(t>=-1e-5)&(s+t<=1+1e-5)
    z=np.where(valid,a[:,:,2]+s*u[:,:,2]+t*v[:,:,2],-np.inf).max(1)
    assert np.isfinite(z).all(), 'Terrain height query outside mesh'
    return z


def summarize(rows, scene, end_reason):
    t, actual, cmd = rows['time'], rows['velocity'], rows['command']
    out = dict(protocol=VERSION, scene=scene, duration=float(t[-1]), end_reason=end_reason,
               command_observation_max_error=float(np.abs(cmd-rows['observed_command']).max()),
               max_tilt_degrees=float(np.rad2deg(rows['tilt']).max()),
               clearance_range=[float(rows['clearance'].min()), float(rows['clearance'].max())],
               base_contact_max_N=float(rows['base_contact'].max()),
               pd_saturation_fraction=rows['saturated'].mean(0).tolist(),
               path_length=float(np.linalg.norm(np.diff(rows['position'][:,:2],axis=0),axis=1).sum()),
               target_contact_seconds=float(rows['target_contact'].sum()*.0025),
               target_contact_x_range=None, segments=[])
    hit = rows['target_contact'].astype(bool)
    if hit.any():
        out['target_contact_x_range'] = [float(rows['target_contact_x'][hit,0].min()), float(rows['target_contact_x'][hit,1].max())]
    out['terrain_coverage'] = 'flat' if scene == 'flat' else ('contacted; inspect contact span' if hit.any() else 'not_entered')
    out['response_time_definition']='90% is first directional crossing; stable is within +/-10% of nonzero target for >=1s. Null means not reached for nonzero targets, not applicable for zero targets.'
    for start, end, target in schedule(scene):
        mask = (t > start+1e-8) & (t <= end+1e-8)
        if not mask.any():
            out['segments'].append(dict(start=start,end=end,status='not_reached'))
            continue
        times, v = t[mask], actual[mask]
        tail = v[times > max(start,end-5)]
        # If interrupted before its final 5 seconds, report the observed tail explicitly.
        if not len(tail): tail = v[times > max(start,times[-1]-5)]
        target = np.asarray(target)
        reach, reverse, stable = [], [], []
        for axis in range(3):
            hit90 = np.flatnonzero(np.sign(target[axis])*v[:,axis] >= .9*abs(target[axis])) if target[axis] else []
            reach.append(float(times[hit90[0]]-start) if len(hit90) else None)
            changed = np.flatnonzero(np.sign(target[axis])*v[:,axis] > 0) if target[axis] else []
            reverse.append(float(times[changed[0]]-start) if len(changed) else None)
            band = (np.abs(v[:,axis]-target[axis]) <= .1*abs(target[axis])) if target[axis] else np.zeros(len(v),bool)
            streak = np.convolve(band.astype(int), np.ones(400,dtype=int), mode='valid') if len(v)>=400 else []
            held = np.flatnonzero(np.asarray(streak)==400)
            stable.append(float(times[held[0]]-start) if len(held) else None)
        pos = rows['position'][mask,:2]
        out['segments'].append(dict(start=start,end=end,command=target.tolist(),
            status='complete' if times[-1] >= end-.003 else 'interrupted',
            rmse=np.sqrt(np.mean((v-target)**2,axis=0)).tolist(),
            tail_mean=tail.mean(0).tolist(),tail_error=(tail.mean(0)-target).tolist(),
            tail_seconds=min(5.,float(times[-1]-start)), time_to_90_percent=reach,
            time_to_10_percent_band_held_1s=stable,
            time_to_target_sign=reverse,
            stop_displacement=float(np.linalg.norm(pos[-1]-pos[0])) if not target.any() else None))
    return out


def self_check():
    assert command(50,'flat') == [0,0,-.8] and command(80,'flat') == [0,0,0]
    for scene in SCENES:
        raw, active, meta = geometry(scene)
        assert raw.shape == (801,801) and raw.dtype == np.int16
        center = raw[:,400].astype(float)*VS
        if scene.startswith('slope'):
            sign = 1 if '_up_' in scene else -1
            assert np.isclose((center[490]-center[420])/7, sign*np.tan(np.deg2rad(8)), atol=.001)
        if scene.startswith('stairs'):
            deltas = np.diff(center[400:450]); sign = 1 if '_up_' in scene else -1
            assert np.count_nonzero(abs(deltas)>.01) == 12
            assert np.allclose(deltas[abs(deltas)>.01],sign*(.05 if '5cm' in scene else .08))
        assert not active[400,400]
    raw = np.array([[0,10],[20,30]],dtype=np.int16)
    assert np.allclose(ground(raw,np.array([[-39.975,-39.975]])),.0075)
    t=np.arange(1,2001)*.0025+5
    v=np.zeros((len(t),3));v[:100,0]=.5
    rows=dict(time=t,velocity=v,command=np.tile([.5,0,0],(len(t),1)),observed_command=np.tile([.5,0,0],(len(t),1)),
              tilt=np.zeros(len(t)),clearance=np.ones(len(t))*.42,base_contact=np.zeros(len(t)),saturated=np.zeros((len(t),16)),
              position=np.zeros((len(t),3)),target_contact=np.zeros(len(t)))
    result=summarize(rows,'flat','completed')['segments'][1]
    assert result['time_to_90_percent'][0] is not None and result['time_to_10_percent_band_held_1s'][0] is None
    rows['velocity'][:,0]=.5
    assert summarize(rows,'flat','completed')['segments'][1]['time_to_10_percent_band_held_1s'][0] is not None
    print('Protocol commands, terrain signs/heights and interpolation self-check passed')


if __name__ == '__main__':
    self_check()
