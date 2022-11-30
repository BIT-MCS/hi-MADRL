from src.config.roadmap_config.purdue.env_config_purdue100_2u2c import env_config


env_config.update(
    {
        'name': 'purdue100_2u2c_QoS1',
        'sinr_demand': 1.0,
        'max_data_amount': 2e9,
    }
)


env_config['obs_range'] = min(env_config['max_x'], env_config['max_y']) / 4