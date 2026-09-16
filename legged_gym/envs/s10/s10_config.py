"""S10 steady RL control; provenance: resources/robots/s10/SOURCE.md."""
from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO


class S10RoughCfg(GO2WRoughCfg):
    class terrain(GO2WRoughCfg.terrain):
        num_cols = 22  # Original 20 columns, plus two pebble columns.
        pebble_columns = 2
        pebble_horizontal_scale = .025
        pebble_vertical_scale = .001
        pebble_height_range = [.01, .08]
        pebble_density_range = [2., 5.]
        task_spawn_fraction = .5  # Half start on terrain for turning/sideways/parking coverage.

    class commands(GO2WRoughCfg.commands):
        extended_speed_fraction = .2  # Random membership within every terrain column.
        task_curriculum = True
        # forward, reverse, left, right, yaw-left, yaw-right, mixed, parking
        mode_probabilities = [.18, .12, .10, .10, .10, .10, .10, .20]
        base_speed_limits = [1., .6, 1.]
        max_speed_limits = [1.5, .9, 1.5]
        speed_increments = [.1, .05, .1]
        minimum_speed = [.15, .1, .2]
        tracking_absolute_tolerance = [.1, .08, .15]
        tracking_relative_tolerance = .2
        parking_tracking_tolerance = [.02, .02, .04]
        transition_seconds = 1.
        minimum_episode_seconds = 3.
        curriculum_window_s = 20.
        minimum_bucket_seconds = 10.
        promote_score = .8
        demote_score = .4
        heading_command = False  # Sample yaw rate directly; actor input remains vx/vy/yaw rate.
        parking_probability = .2
        parking_thresholds = [.03, .03, .05]  # Command vx/vy (m/s), yaw rate (rad/s).

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
        soft_torque_limit = .8  # Wheel reward budget: 11.2 Nm for the 14 Nm asset; calibrate to hardware.

        class scales(GO2WRoughCfg.rewards.scales):
            # Initial tuning values; wheel action penalty totals -.03 with inherited action_rate.
            wheel_action_rate = -.02
            wheel_torque_excess = -.1
            parking_lin_vel = -.5
            parking_ang_vel = -.25
            parking_wheel_vel = -.002


class S10RoughCfgPPO(GO2WRoughCfgPPO):
    class policy(GO2WRoughCfgPPO.policy):
        init_noise_std = [.3, .3, .3, .6] * 4  # hipx, hipy, knee, wheel per leg.

    class runner(GO2WRoughCfgPPO.runner):
        experiment_name = 'S10_HIM'
        save_interval = 50
        max_iterations = None  # A new run requires an explicit approved budget.
        resume = False
