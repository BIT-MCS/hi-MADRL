from src.config.roadmap_config.purdue.env_config_purdue import env_config


env_config.update(
    {
        'name': 'purdue100',
        'num_human': 100,
    }
)


env_config['obs_range'] = min(env_config['max_x'], env_config['max_y']) / 4  # 可见半径是较短边长的1/4