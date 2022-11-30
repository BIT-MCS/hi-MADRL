from src.config.roadmap_config.NCSU.env_config_NCSU import env_config


env_config.update(
    {
        'name': 'NCSU100',
        'num_human': 100,
    }
)


env_config['obs_range'] = min(env_config['max_x'], env_config['max_y']) / 4