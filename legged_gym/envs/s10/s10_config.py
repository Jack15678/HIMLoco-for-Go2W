"""S10 steady RL control; provenance: resources/robots/s10/SOURCE.md."""
from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO


class S10RoughCfg(GO2WRoughCfg):
    class init_state(GO2WRoughCfg.init_state):
        pos = [0., 0., 0.45]
        default_joint_angles = {
            f'{leg}_{joint}_joint': angle
            for leg, angles in [('fl', [0., -.3, .6, 0.]), ('fr', [0., -.3, .6, 0.]),
                                ('hl', [0., .3, -.6, 0.]), ('hr', [0., .3, -.6, 0.])]
            for joint, angle in zip(['hipx', 'hipy', 'knee', 'wheel'], angles)
        }

    class control(GO2WRoughCfg.control):
        stiffness = {'hipx': 80., 'hipy': 80., 'knee': 80., 'wheel': 0.}
        damping = {'hipx': 2., 'hipy': 2., 'knee': 2., 'wheel': .6}
        action_scale = {f'{leg}_{joint}_joint': scale
                        for leg in ['fl', 'fr', 'hl', 'hr']
                        for joint, scale in [('hipx', .125), ('hipy', .25), ('knee', .25), ('wheel', 0.)]}
        vel_scale = 5.
        decimation = 8  # 2.5 ms PD, 20 ms policy/history/reward integration.
        delay_stride = 2  # Uniform 0/2/4/6 substeps = 0/5/10/15 ms.

    class sim(GO2WRoughCfg.sim):
        dt = .0025

    class asset(GO2WRoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/s10/urdf/S10.urdf'
        name = 's10'
        foot_name = 'wheel'
        wheel_name = ['wheel']
        penalize_contacts_on = ['hipy', 'knee', 'base_link']
        terminate_after_contacts_on = ['base_link']
        flip_visual_attachments = False

    class rewards(GO2WRoughCfg.rewards):
        base_height_target = .425  # FK: 2 * .18 * cos(.3) + .081 = .424921 m.


class S10RoughCfgPPO(GO2WRoughCfgPPO):
    class policy(GO2WRoughCfgPPO.policy):
        init_noise_std = [.3, .3, .3, .6] * 4  # hipx, hipy, knee, wheel per leg.

    class runner(GO2WRoughCfgPPO.runner):
        experiment_name = 'S10_HIM'
        save_interval = 50
        max_iterations = None  # A new run requires an explicit approved budget.
        resume = False
