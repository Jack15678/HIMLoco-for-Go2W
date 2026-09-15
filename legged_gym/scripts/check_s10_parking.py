"""CPU zero-update check of actual shared methods, without importing the Gym binary."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root/'legged_gym/envs/base/legged_robot.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LeggedRobot')
    names = ['_resample_commands', '_update_heading_commands', '_parking_mask',
        '_reward_parking_lin_vel', '_reward_parking_ang_vel', '_reward_parking_wheel_vel',
        '_reward_stand_still', '_reward_dof_vel', '_compute_torques']
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, torch_rand_float=lambda lo,hi,shape,device: (hi-lo)*torch.rand(*shape,device=device)+lo,
        wrap_to_pi=lambda a: (a+torch.pi) % (2*torch.pi)-torch.pi,
        quat_apply=lambda q,v: v+2*torch.cross(q[:,:3],torch.cross(q[:,:3],v,dim=-1)+q[:,3:]*v,dim=-1))
    exec(compile(ast.Module(body=methods,type_ignores=[]), '<shared parking methods>', 'exec'), scope)
    Robot = type('Robot', (), {name:scope[name] for name in names})
    env = Robot(); env.device='cpu'; env.num_envs=200000
    env.cfg=NS(commands=NS(heading_command=True,parking_probability=0.,parking_thresholds=[.03,.03,.05]),
        control=NS(vel_scale=5.,control_type='P'))
    env.command_ranges=dict(lin_vel_x=[-1.,1.],lin_vel_y=[-.6,.6],heading=[-3.14,3.14])
    env.commands=torch.zeros(env.num_envs,4);env.parking_commands=torch.zeros(env.num_envs,dtype=torch.bool)
    env.base_quat=torch.zeros(env.num_envs,4);env.base_quat[:,3]=1
    env.forward_vec=torch.zeros(env.num_envs,3);env.forward_vec[:,0]=1
    ids=torch.arange(env.num_envs); coverage={}
    for probability in [0.,.2]:
        torch.manual_seed(1);env.cfg.commands.parking_probability=probability
        env._resample_commands(ids);env._update_heading_commands(slice(None))
        coverage[str(probability)] = dict(samples=env.num_envs,
            exact_zero_fraction=float((env.commands[:,:3]==0).all(1).float().mean()),
            threshold_mask_fraction=float(env._parking_mask().float().mean()),
            xy_zero_fraction=float((env.commands[:,:2]==0).all(1).float().mean()),
            explicit_fraction=float(env.parking_commands.float().mean()))
    assert coverage['0.0']['exact_zero_fraction'] < .0001
    assert .19 < coverage['0.2']['exact_zero_fraction'] < .21
    env.commands[:,3]=2.;env._update_heading_commands(slice(None))
    assert (env.commands[env.parking_commands,:3]==0).all()
    torch.testing.assert_close(env.commands[~env.parking_commands,2], torch.ones((~env.parking_commands).sum()))
    env._resample_commands(ids[:0]);env._update_heading_commands(ids[:3])
    assert (env.commands[env.parking_commands,:3]==0).all()
    # Every axis boundary, signed boundary, and a pure nonzero yaw command.
    env.num_envs=10
    env.commands=torch.tensor([[0.,0.,0.,0.],[.03,.03,.05,0.],[-.03,-.03,-.05,0.],
        [.03001,0.,0.,0.],[0.,.03001,0.,0.],[0.,0.,.05001,0.],
        [0.,0.,.8,0.],[-.03001,0.,0.,0.],[0.,-.03001,0.,0.],[0.,0.,-.05001,0.]])
    assert env._parking_mask().tolist() == [True]*3+[False]*7
    env.base_lin_vel=torch.tensor([[.6,.8,0.]]).repeat(10,1)
    env.base_ang_vel=torch.tensor([[0.,0.,2.]]).repeat(10,1)
    env.dof_vel=torch.zeros(10,16);env.wheel_indices=[3,7,11,15];env.dof_vel[:,env.wheel_indices]=2
    env.default_dof_pos=torch.zeros(1,16);env.dof_pos=torch.zeros(10,16)
    env.action_scale=torch.tensor([.125,.25,.25,0.]*4)
    env.p_gains=torch.tensor([80.,80.,80.,0.]*4);env.d_gains=torch.tensor([2.,2.,2.,.6]*4)
    env.Kp_factors=env.Kd_factors=1.;env.torque_limits=torch.ones(16)*100
    expected=torch.tensor([1.]*3+[0.]*7)
    torch.testing.assert_close(env._reward_parking_lin_vel(),expected)
    torch.testing.assert_close(env._reward_parking_ang_vel(),2*expected)
    torch.testing.assert_close(env._reward_parking_wheel_vel(),2*expected)
    assert (env._reward_dof_vel()==0).all() and (env.dof_vel[:,env.wheel_indices]==2).all()
    env.commands[:]=0;env.dof_vel[:]=0
    action=torch.zeros(10,16);action[:,env.wheel_indices]=.2
    tau=env._compute_torques(action)
    torch.testing.assert_close(tau[:,env.wheel_indices],torch.full((10,4),.6))
    assert (env._reward_parking_wheel_vel()==0).all()  # Nonzero targets are not actual motion.
    env.dof_vel[:,env.wheel_indices]=-3
    assert (env._reward_parking_wheel_vel()==3).all()
    print(json.dumps(dict(coverage=coverage, mask_boundaries=True, moving_zero_command_immediate=True,
        actual_wheel_speed_only=True, nonzero_wheel_target_rad_s=1., nonzero_holding_torque_Nm=.6,
        parameter_updates=0), indent=2))


if __name__ == '__main__':
    main()
