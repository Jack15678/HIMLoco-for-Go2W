"""Frozen evaluation only: 90 s flat following or 60 s measured Gym terrain."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from s10_eval_protocol import (VERSION, SCENES, HS, VS, EXTENT, geometry, ground, mesh_ground,
                              command, schedule, summarize, write_json)


class Video:
    def __init__(self, path, scene, source):
        import matplotlib
        from PIL import ImageFont
        self.scene, self.source = scene, source
        self.font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),17)
        self.process = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
            '-pixel_format','rgb24','-video_size','960x640','-framerate','20','-i','-',
            '-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','23','-movflags','+faststart',str(path)],
            stdin=subprocess.PIPE)

    def frame(self, rgb, t, cmd, actual):
        from PIL import Image, ImageDraw
        im = Image.fromarray(np.asarray(rgb,dtype=np.uint8))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0,0,960,70),fill='black')
        draw.text((10,3), f'{self.source} | {self.scene} | fixed trial 0 | {t:6.2f} s', fill='white',font=self.font)
        draw.text((10,25), f'cmd vx/vy/yaw {cmd[0]:+.2f}/{cmd[1]:+.2f}/{cmd[2]:+.2f} | actual {actual[0]:+.2f}/{actual[1]:+.2f}/{actual[2]:+.2f}',fill='white',font=self.font)
        draw.text((10,47), 'Camera follows position with fixed world direction. No resets.',fill='white',font=self.font)
        self.process.stdin.write(np.asarray(im).tobytes())

    def close(self):
        self.process.stdin.close()
        assert self.process.wait() == 0


def finish(out, rows, scene, reason):
    rows = {k:np.asarray(v) for k,v in rows.items()}
    assert rows and all(np.isfinite(v).all() for v in rows.values())
    assert np.max(np.abs(rows['command']-rows['observed_command'])) < 1e-6
    np.savez_compressed(out/'trace.npz', **rows)
    result = summarize(rows,scene,reason)
    write_json(out/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
    for i,name in enumerate(['vx (m/s)','vy (m/s)','yaw rate (rad/s)']):
        a=ax.flat[i]; a.plot(rows['time'],rows['command'][:,i],label='issued',lw=1.4)
        a.plot(rows['time'],rows['observed_command'][:,i],label='policy observation',ls='--')
        a.plot(rows['time'],rows['velocity'][:,i],label='actual',lw=.7)
        a.set(xlabel='seconds',ylabel=name);a.legend();a.grid(alpha=.2)
    ax.flat[3].plot(rows['time'],np.unwrap(rows['heading']));ax.flat[3].set(xlabel='seconds',ylabel='unwrapped heading (rad)')
    ax.flat[4].plot(rows['position'][:,0],rows['position'][:,1]);ax.flat[4].set(xlabel='x (m)',ylabel='y (m)');ax.flat[4].axis('equal')
    ax.flat[5].plot(rows['time'],rows['clearance'],label='height above local surface')
    ax.flat[5].plot(rows['time'],np.rad2deg(rows['tilt']),label='tilt (deg)');ax.flat[5].legend()
    fig.suptitle(f'{VERSION}: {scene}; {reason}; {rows["time"][-1]:.2f} seconds')
    fig.savefig(out/'following.png',dpi=130);plt.close(fig)
    return result


def empty_rows():
    return {k:[] for k in ['time','command','observed_command','velocity','position','heading',
        'tilt','clearance','base_contact','saturated','target_contact','target_contact_x','qpos','qvel']}


def gym_run(args):
    import isaacgym
    from isaacgym import gymapi, gymtorch, gymutil, terrain_utils
    from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse
    import torch
    from legged_gym.envs import S10RoughCfg, LeggedRobot
    from legged_gym.scripts.evaluate_s10 import restore_config
    from legged_gym.utils.helpers import class_to_dict, export_policy_as_jit, set_seed
    from rsl_rl.modules import HIMActorCritic

    saved = json.loads((args.run/'config.json').read_text())
    cfg = S10RoughCfg();restore_config(cfg,saved['env'])
    task_probe = getattr(args, 'protocol', 'v1') == 'v2'
    n = 1 if args.scene == 'flat' and not task_probe else 3
    cfg.env.num_envs = n
    cfg.env.episode_length_s = 100.
    cfg.commands.heading_command = cfg.commands.curriculum = False
    cfg.commands.resampling_time = 1e6
    cfg.noise.add_noise = False
    for k in dir(cfg.domain_rand):
        if isinstance(getattr(cfg.domain_rand,k),bool): setattr(cfg.domain_rand,k,False)
    cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False
    raw, active, meta = geometry(args.scene)
    offsets = np.array([[0.,100.*i,0.] for i in range(n)],dtype=np.float32)
    offsets[:,2] = ground(raw,offsets[:,:2]*0)
    if task_probe:
        support = np.array([[x,y] for x in [-.4, 0., .4] for y in [-.4, 0., .4]])
        offsets[:, 2] = ground(raw, support).max()

    class Spawn:
        def __init__(self, backend): self.backend=backend; self.count=0
        def __getattr__(self,name): return getattr(self.backend,name)
        def create_actor(self, env, asset, pose, *rest):
            p=offsets[self.count].copy();p[2]+=.45;self.count+=1
            pose.p=gymapi.Vec3(*p)
            return self.backend.create_actor(env,asset,pose,*rest)

    class EvaluationRobot(LeggedRobot):
        def _get_env_origins(self):
            self.custom_origins=False
            self.env_origins=torch.tensor(offsets,device=self.device)
        def create_sim(self):
            self.gym=Spawn(self.gym)
            self.up_axis_idx=2
            self.sim=self.gym.create_sim(self.sim_device_id,self.sim_device_id,self.physics_engine,self.sim_params)
            assert self.sim is not None
            if args.scene == 'flat': self._create_ground_plane()
            else:
                vertices,triangles=terrain_utils.convert_heightfield_to_trimesh(raw,HS,VS,None if args.scene.startswith('pebbles') else .25)
                self.eval_vertices=vertices.reshape(*raw.shape,3).copy()
                self.eval_vertices[:,:,:2]-=EXTENT
                for i in range(n):
                    params=gymapi.TriangleMeshParams()
                    params.nb_vertices,params.nb_triangles=len(vertices),len(triangles)
                    params.transform.p=gymapi.Vec3(-EXTENT,100.*i-EXTENT,0.)
                    params.static_friction=cfg.terrain.static_friction
                    params.dynamic_friction=cfg.terrain.dynamic_friction
                    params.restitution=cfg.terrain.restitution
                    self.gym.add_triangle_mesh(self.sim,vertices.ravel(),triangles.ravel(),params)
            self._create_envs()

    set_seed(1)
    sim=gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim),sim)
    sim.up_axis=gymapi.UP_AXIS_Z;sim.use_gpu_pipeline=True
    sim.physx.use_gpu=True
    env=EvaluationRobot(cfg,sim,gymapi.SIM_PHYSX,'cuda:0',True)
    if args.scene!='flat':
        profile_x=np.linspace(0,12,121)
        meta['actual_mesh_profile_x']=profile_x.tolist()
        meta['actual_mesh_profile_height']=mesh_ground(env.eval_vertices,np.c_[profile_x,np.zeros_like(profile_x)]).tolist()
    checkpoint=torch.load(args.run/f'model_{args.iteration}.pt',map_location='cpu')
    assert checkpoint['iter']==args.iteration and checkpoint['tot_timesteps']==args.iteration*4096*48
    actor=HIMActorCritic(342,262,57,16,**saved['train']['policy']).to(env.device).eval()
    actor.load_state_dict(checkpoint['model_state_dict'])
    assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
    export_policy_as_jit(actor,str(args.output),env=env)
    write_json(args.output/'geometry.json',meta)
    write_json(args.output/'evaluation_config.json',class_to_dict(cfg))
    np.savez_compressed(args.output/'terrain.npz',height=raw,active=active,horizontal_scale=HS,vertical_scale=VS)
    env.dof_pos[:]=env.default_dof_pos;env.dof_vel[:]=0
    env.root_states[:]=env.base_init_state
    env.root_states[:,:3]+=env.env_origins
    env.root_states[:,7:]=0
    perturb=np.zeros((n,14),dtype=np.float32)
    if args.scene != 'flat' or task_probe:
        perturb=np.stack([np.random.default_rng(i).uniform(-.02,.02,14) for i in range(n)]).astype(np.float32)
    legs=[i for i in range(16) if i%4!=3]
    env.dof_pos[:,legs]+=torch.tensor(perturb[:,:12],device=env.device)
    initial_yaw = np.deg2rad([0, 90, 180]).astype(np.float32) if task_probe else np.zeros(n,dtype=np.float32)
    env.root_states[:,3:7]=quat_from_euler_xyz(*[torch.tensor(v,device=env.device) for v in [perturb[:,12],perturb[:,13],initial_yaw]])
    env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env.root_states))
    env.commands[:]=0;env.actions[:]=0;env.obs_buf[:]=0;env.privileged_obs_buf[:]=0
    write_json(args.output/'initial_states.json',dict(seeds=[None] if n==1 else list(range(n)),initial_yaw=initial_yaw.tolist(),perturbations=perturb.tolist(),root=env.root_states.tolist(),dof=env.dof_pos.tolist(),warmup_steps=0))
    camera_params=gymapi.CameraProperties();camera_params.width=960;camera_params.height=640;camera_params.horizontal_fov=60
    camera=env.gym.create_camera_sensor(env.envs[0],camera_params);assert camera>=0
    video=Video(args.output/'representative.mp4',args.scene,'Isaac Gym actual geometry')
    rows=[empty_rows() for _ in range(n)]; alive=np.ones(n,bool);reason=['completed']*n
    limits=env.torque_limits.cpu().numpy();total=round((args.seconds or schedule(args.scene)[-1][1])/.02)
    # Evaluation deliberately uses the existing PD and observation functions without the
    # training step's command resampling, reward/curriculum or automatic episode reset.
    with torch.inference_mode():
        for step in range(total):
            cmd=np.asarray(command(step*.02,args.scene),dtype=np.float32)
            env.base_quat[:]=env.root_states[:,3:7]
            env.base_ang_vel[:]=quat_rotate_inverse(env.base_quat,env.root_states[:,10:13])
            env.base_lin_vel[:]=quat_rotate_inverse(env.base_quat,env.root_states[:,7:10])
            env.projected_gravity[:]=quat_rotate_inverse(env.base_quat,env.gravity_vec)
            env.commands[:,:3]=torch.tensor(cmd,device=env.device)
            env.compute_observations()
            env.obs_buf.clamp_(-cfg.normalization.clip_observations,cfg.normalization.clip_observations)
            observed=(env.obs_buf[:,6:9]/env.commands_scale).cpu().numpy()
            assert np.allclose(observed,cmd,atol=1e-7)
            action=actor.act_inference(env.obs_buf).clamp(-cfg.normalization.clip_actions,cfg.normalization.clip_actions)
            assert torch.isfinite(action).all()
            env.actions[:]=action
            for sub in range(8):
                tau=env._compute_torques(action)
                tau[torch.tensor(~alive,device=env.device)]=0
                assert torch.isfinite(tau).all()
                env.gym.set_dof_actuation_force_tensor(env.sim,gymtorch.unwrap_tensor(tau))
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim,True)
                for refresh in ['refresh_dof_state_tensor','refresh_actor_root_state_tensor','refresh_net_contact_force_tensor','refresh_rigid_body_state_tensor']:
                    getattr(env.gym,refresh)(env.sim)
                root=env.root_states.cpu().numpy().copy();root[:,1]-=offsets[:,1]
                quat=env.root_states[:,3:7]
                lin=quat_rotate_inverse(quat,env.root_states[:,7:10])
                ang=quat_rotate_inverse(quat,env.root_states[:,10:13])
                vel=torch.stack([lin[:,0],lin[:,1],ang[:,2]],dim=1).cpu().numpy()
                grav=quat_rotate_inverse(quat,env.gravity_vec).cpu().numpy()
                tilt=np.arccos(np.clip(-grav[:,2],-1,1))
                base=np.linalg.norm(env.contact_forces[:,env.termination_contact_indices].cpu().numpy(),axis=-1).max(1)
                feet=env.rigid_body_states.view(n,env.num_bodies,13)[:,env.feet_indices,:3].cpu().numpy().copy()
                feet[:,:,1]-=offsets[:,1,None]
                footforce=np.linalg.norm(env.contact_forces[:,env.feet_indices].cpu().numpy(),axis=-1)
                indices=np.clip(np.floor((feet[:,:,:2]+EXTENT)/HS).astype(int),0,len(raw)-1)
                target_feet=active[indices[:,:,0],indices[:,:,1]] & (footforce>1.)
                entered=target_feet.any(1)
                bounded_xy=np.clip(root[:,:2],-EXTENT+.2,EXTENT-.2)
                clear=root[:,2]-(mesh_ground(env.eval_vertices,bounded_xy) if args.scene!='flat' else 0.)
                sat=np.abs(tau.cpu().numpy())>=.99*limits
                q,dq=env.dof_pos.cpu().numpy(),env.dof_vel.cpu().numpy()
                t=(step*8+sub+1)*.0025
                for i in range(n):
                    if not alive[i]: continue
                    x,y,z,w=root[i,3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                    contact_x=feet[i,target_feet[i],0]
                    span=[float(contact_x.min()),float(contact_x.max())] if len(contact_x) else [0.,0.]
                    values=dict(time=t,command=cmd,observed_command=observed[i],velocity=vel[i],position=root[i,:3],heading=yaw,tilt=tilt[i],clearance=clear[i],base_contact=base[i],saturated=sat[i],target_contact=entered[i],target_contact_x=span,qpos=np.r_[root[i,:3],root[i,[6,3,4,5]],q[i]],qvel=np.r_[root[i,7:10],ang[i].cpu().numpy(),dq[i]])
                    assert all(np.isfinite(v).all() for v in values.values())
                    for k,value in values.items(): rows[i][k].append(np.array(value).copy())
                    if base[i]>1. or tilt[i]>np.deg2rad(45.) or clear[i]<.2:
                        alive[i]=False;reason[i]='fall_or_base_contact'
                    elif np.abs(root[i,:2]).max()>EXTENT-1.:
                        alive[i]=False;reason[i]='out_of_bounds'
                if (step*8+sub)%20==0 and len(rows[0]['time']) and rows[0]['time'][-1]==t:
                    p=root[0,:3]
                    env.gym.set_camera_location(camera,env.envs[0],gymapi.Vec3(*(p+[-2.,-2.,1.2])),gymapi.Vec3(*p))
                    env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
                    rgb=env.gym.get_camera_image(env.sim,env.envs[0],camera,gymapi.IMAGE_COLOR).reshape(640,960,4)[:,:,:3]
                    video.frame(rgb,t,cmd,vel[0])
                if not alive.any(): break
            if not alive.any(): break
    video.close()
    summaries=[]
    for i in range(n):
        out=args.output/f'trial_{i}';out.mkdir()
        summaries.append(finish(out,rows[i],args.scene,reason[i]))
    write_json(args.output/'summary.json',dict(protocol=VERSION,scene=args.scene,iteration=args.iteration,trials=summaries,representative_trial=0,ppo_updates=0))
    env.gym.destroy_sim(env.sim)


def mujoco_run(args):
    import torch
    import mujoco
    from deploy.s10_mujoco import load_model,observation,targets
    cfg=json.loads((args.policy/'policy.json').read_text())
    policy=torch.jit.load(str(args.policy/'policy.pt')).eval()
    model,data,joints,motors=load_model(cfg)
    assert np.isclose(model.opt.timestep,.0025) and cfg['decimation']==8 and args.scene=='flat'
    qids,vids=model.jnt_qposadr[joints],model.jnt_dofadr[joints]
    base=model.body('base_link').id
    kp,kd,limits=map(np.asarray,(cfg['p_gains'],cfg['d_gains'],cfg['torque_limits']))
    history=np.zeros((6,57),np.float32);action=np.zeros(16,np.float32)
    velocity,force=np.zeros(6),np.zeros(6);rows=empty_rows();reason='completed'
    camera=mujoco.MjvCamera();camera.distance=2.9;camera.azimuth=225;camera.elevation=-24
    model.vis.global_.offwidth=960;model.vis.global_.offheight=640
    visual=mujoco.MjvOption();visual.geomgroup[1]=0
    video=Video(args.output/'representative.mp4','flat','MuJoCo physics')
    with mujoco.Renderer(model,height=640,width=960) as renderer, torch.inference_mode():
        for step in range(round((args.seconds or schedule('flat')[-1][1])/.02)):
            cmd=np.asarray(command(step*.02,'flat'),dtype=np.float32)
            mujoco.mj_forward(model,data)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,base,velocity,1)
            gravity=data.xmat[base].reshape(3,3).T@np.array([0.,0.,-1.])
            history[1:]=history[:-1].copy()
            history[0]=observation(cfg,velocity[:3],gravity,cmd,data.qpos[qids],data.qvel[vids],action)
            observed=history[0,6:9]/cfg['commands_scale']
            action=policy(torch.from_numpy(history.ravel())).numpy()
            assert np.isfinite(action).all()
            action=np.clip(action,-cfg['clip_actions'],cfg['clip_actions'])
            q,v=targets(cfg,action)
            for sub in range(8):
                tau=np.clip(kp*(q-data.qpos[qids])+kd*(v-data.qvel[vids]),-limits,limits)
                data.ctrl[motors]=tau;mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
                mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,base,velocity,1)
                contact=0.
                for ci in range(data.ncon):
                    c=data.contact[ci]
                    if base in model.geom_bodyid[[c.geom1,c.geom2]]:
                        mujoco.mj_contactForce(model,data,ci,force);contact=max(contact,float(np.linalg.norm(force[:3])))
                rotation=data.xmat[base].reshape(3,3);tilt=np.arccos(np.clip(rotation[2,2],-1,1))
                t=(step*8+sub+1)*.0025;vel=velocity[[3,4,2]].copy()
                values=dict(time=t,command=cmd,observed_command=observed,velocity=vel,position=data.qpos[:3],heading=np.arctan2(rotation[1,0],rotation[0,0]),tilt=tilt,clearance=data.qpos[2],base_contact=contact,saturated=np.abs(tau)>=.99*limits,target_contact=False,target_contact_x=[0.,0.],qpos=data.qpos,qvel=data.qvel)
                assert all(np.isfinite(v).all() for v in values.values())
                for k,value in values.items(): rows[k].append(np.array(value).copy())
                if (step*8+sub)%20==0:
                    camera.lookat[:]=data.qpos[:3];renderer.update_scene(data,camera,scene_option=visual)
                    video.frame(renderer.render(),t,cmd,vel)
                if contact>1. or tilt>np.deg2rad(45.) or data.qpos[2]<.2:
                    reason='fall_or_base_contact';break
            if reason!='completed':break
    video.close();result=finish(args.output,rows,'flat',reason)
    result.update(iteration=args.iteration,ppo_updates=0,warnings=data.warning.number.tolist(),physics_version=mujoco.__version__)
    write_json(args.output/'summary.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--iteration',type=int,required=True)
    parser.add_argument('--scene',choices=SCENES+['pebbles_2cm','pebbles_6cm'],default='flat')
    parser.add_argument('--protocol',choices=['v1','v2'],default='v1')
    parser.add_argument('--probe',default='forward')
    parser.add_argument('--geometry-seed',type=int,default=701)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['gym','mujoco'],default='gym')
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--seconds',type=float,help='Short infrastructure smoke only; omit for protocol results.')
    args=parser.parse_args()
    if args.protocol == 'v2':
        import s10_task_protocol as protocol
        if args.probe not in protocol.PROBES:
            parser.error('Unknown probe: '+args.probe)
        protocol.PROBE, protocol.SEED = args.probe, args.geometry_seed
        for name in ['VERSION', 'SCENES', 'HS', 'VS', 'EXTENT', 'geometry', 'ground', 'mesh_ground', 'command', 'schedule', 'summarize']:
            globals()[name] = getattr(protocol, name)
    elif args.scene.startswith('pebbles'):
        parser.error('Pebble probes require --protocol v2')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'invocation.json',dict(protocol=VERSION,**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    (gym_run if args.backend=='gym' else mujoco_run)(args)
