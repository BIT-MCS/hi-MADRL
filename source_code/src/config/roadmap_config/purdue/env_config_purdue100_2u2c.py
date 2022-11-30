from src.config.roadmap_config.purdue.env_config_purdue100 import env_config


env_config.update(
    {
        'name': 'purdue100_2u2c',
        'num_uav': 2,
    }
)


env_config['obs_range'] = min(env_config['max_x'], env_config['max_y']) / 4