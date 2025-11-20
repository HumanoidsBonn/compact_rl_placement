from gym.envs.registration import register

register(
    id='BaseEnv-v0',
    entry_point='fragment_gym.env.base_env:BasePybulletEnv',
    kwargs={'render': False}
)
register(
    id='BaseEnvGUI-v0',
    entry_point='fragment_gym.env.base_env:BasePybulletEnv',
    kwargs={'render': True}
)

register(
    id='RobotiqEnv-v0',
    entry_point='fragment_gym.env.robotiq_env:RobotEnv',
    kwargs={'render': False}
)
register(
    id='RobotiqEnvGUI-v0',
    entry_point='fragment_gym.env.robotiq_env:RobotEnv',
    kwargs={'render': True}
)

register(
    id='MainEnv-v0',
    entry_point='fragment_gym.env.main_env:MainEnv',
    kwargs={'render': False}
)
register(
    id='MainEnvGUI-v0',
    entry_point='fragment_gym.env.main_env:MainEnv',
    kwargs={'render': True}
)

register(
    id='GraspeAndPlaceEnvGUI-v0',
    entry_point='fragment_gym.rl_env.grasp_and_place_task:GraspeAndPlace',
    kwargs={'render': True}
)
register(
    id='GraspeAndPlaceEnv-v0',
    entry_point='fragment_gym.rl_env.grasp_and_place_task:GraspeAndPlace',
    kwargs={'render': False}
)

register(
    id='BaselineScalingFrescoEnvGUI-v0',
    entry_point='fragment_gym.baseline.baseline_scaling_fresco:BaselineScalingFrescoEnv',
    kwargs={'render': True}
)
register(
    id='BaselineScalingFrescoEnv-v0',
    entry_point='fragment_gym.baseline.baseline_scaling_fresco:BaselineScalingFrescoEnv',
    kwargs={'render': False}
)

register(
    id='BaselineRelativePlacingEnvGUI-v0',
    entry_point='fragment_gym.baseline.baseline_relative_placing:BaselineRelativePlacingEnv',
    kwargs={'render': True}
)
register(
    id='BaselineRelativePlacingEnv-v0',
    entry_point='fragment_gym.baseline.baseline_relative_placing:BaselineRelativePlacingEnv',
    kwargs={'render': False}
)