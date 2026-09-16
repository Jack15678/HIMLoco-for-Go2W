"""Derive braking/holding metrics from frozen s10-follow-terrain-v1 PD traces."""
import argparse
import json
from pathlib import Path
import numpy as np


def parking_windows(time, command, position, heading, velocity, wheels, initial_position, initial_heading):
    time, heading = np.asarray(time), np.unwrap(heading)
    mask = (np.abs(command) <= [.03, .03, .05]).all(1)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    result = []
    dt = float(np.median(np.diff(time)))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        receipt = time[start-1] if start else 0.
        origin = position[start-1] if start else np.asarray(initial_position)
        yaw0 = heading[start-1] if start else initial_heading
        indices = np.arange(start, end)
        brake = indices[time[indices] <= receipt + 1. + dt/10]
        hold = indices[time[indices] > receipt + 1. + dt/10]
        tail = indices[time[indices] > max(receipt, time[end-1]-2.)]
        def motion(ids, p0, h0):
            if not len(ids):
                return None
            xy = np.vstack([p0[:2], position[ids,:2]])
            yaw = np.r_[h0, heading[ids]]
            return dict(net_xy_m=float(np.linalg.norm(xy[-1]-xy[0])),
                path_xy_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
                max_xy_from_start_m=float(np.linalg.norm(xy-xy[0],axis=1).max()),
                net_heading_rad=float(yaw[-1]-yaw[0]), accumulated_heading_rad=float(np.abs(np.diff(yaw)).sum()))
        stop_at = None
        slow = (np.linalg.norm(velocity[start:end,:2],axis=1) <= .01) & (np.abs(velocity[start:end,2]) <= .02)
        count = 0
        for i, value in enumerate(slow):
            count = count+1 if value else 0
            if count >= round(1./dt):
                stop_at = float(time[start+i]-receipt)
                break
        result.append(dict(receipt_seconds=float(receipt), end_seconds=float(time[end-1]),
            receipt_position_m=np.asarray(origin).tolist(), receipt_heading_rad=float(yaw0),
            window_definition='First 1 s braking; remaining holding; tail last 2 s. Windows do not imply settled.',
            braking=motion(brake,origin,yaw0),
            holding=motion(hold,position[brake[-1]],heading[brake[-1]]) if len(brake) else None,
            whole=motion(indices,origin,yaw0),
            low_speed_definition='Horizontal speed <= .01 m/s and abs yaw rate <= .02 rad/s for 1 s; drift reported independently.',
            first_one_second_low_speed_confirmed_at_s=stop_at,
            tail_velocity_mean=velocity[tail].mean(0).tolist(),
            tail_velocity_rms=np.sqrt(np.square(velocity[tail]).mean(0)).tolist(),
            tail_velocity_abs_max=np.abs(velocity[tail]).max(0).tolist(),
            tail_wheel_rms_rad_s=np.sqrt(np.square(wheels[tail]).mean(0)).tolist()))
    return result


def analyze(assessment):
    result = {}
    for file in sorted(assessment.glob('**/trace.npz')):
        with np.load(file) as r:
            if 'command' not in r:
                continue
            result[str(file.relative_to(assessment)).replace('\\','/')] = parking_windows(
                r['time'],r['command'],r['position'],r['heading'],r['velocity'],r['qvel'][:,-16:][:,[3,7,11,15]],
                [0.,0.,.45],0.)
    standing = assessment/'stand_run/standing'
    initial = json.loads((standing/'initial_states.json').read_text())
    for backend in ['gym','mujoco']:
        with np.load(standing/f'{backend}_trace.npz') as r:
            dt=float(r['dt']); time=(np.arange(len(r['velocity']))+1)*dt
            n=r['velocity'].shape[1] if backend=='gym' else 1
            for trial in range(n):
                q = r['root'][:,trial] if backend=='gym' else r['qpos']
                quat=q[:,[6,3,4,5]] if backend=='gym' else q[:,3:7]
                w,x,y,z=quat.T
                heading=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                velocity=r['velocity'][:,trial] if backend=='gym' else r['velocity']
                wheel=(r['dof_velocity'][:,trial] if backend=='gym' else r['qvel'][:,-16:])[:,[3,7,11,15]]
                p0=initial['root_local'][trial][:3] if backend=='gym' else [0.,0.,.45]
                # Original Gym perturbations are roll/pitch only; initial yaw is zero.
                entry=parking_windows(time,np.zeros((len(time),3)),q[:,:3],heading,velocity,wheel,p0,0.)[0]
                for key in ['torque','velocity_target']:
                    values=(r[key][:,trial] if backend=='gym' else r[key])[:,[3,7,11,15]]
                    entry['tail_wheel_'+key+'_rms']=np.sqrt(np.square(values[time>8]).mean(0)).tolist()
                    entry['wheel_'+key+'_abs_max']=np.abs(values).max(0).tolist()
                result[f'standing/{backend}/{trial}']=entry
    return result


def self_check():
    t=np.arange(1,41)*.1
    cmd=np.zeros((40,3));cmd[:10,0]=.5
    position=np.c_[t,np.zeros((40,2))]
    vel=np.c_[np.ones(40),np.zeros((40,2))]
    vel[15]=0  # A single zero crossing must never count as settled.
    r=parking_windows(t,cmd,position,t*.1,vel,np.ones((40,4)),[0,0,0],0)[0]
    assert np.isclose(r['receipt_seconds'],1.) and np.isclose(r['braking']['net_xy_m'],1.)
    assert np.isclose(r['holding']['net_xy_m'],2.) and r['first_one_second_low_speed_confirmed_at_s'] is None
    assert np.isclose(r['whole']['accumulated_heading_rad'],.3)
    print('PASS: command-receipt origin, separate 1 s braking/holding windows, no instant-zero settled claim')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assessment',type=Path,nargs='?')
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_check:
        self_check()
    else:
        result=analyze(args.assessment)
        output=args.output or args.assessment/'parking_metrics.json'
        output.write_text(json.dumps(result,indent=2,allow_nan=False))
        print(output)
