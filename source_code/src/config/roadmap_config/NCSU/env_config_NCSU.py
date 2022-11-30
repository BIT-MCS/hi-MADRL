from src.config.base_config import env_config
from src.envs.roadmap_env.roadmap_utils import Roadmap

rm = Roadmap('NCSU')
env_config.update(
    {
        'name': 'NCSU',
        'max_x': round(rm.max_dis_x),  # 2718,  # more precisely, 2718.3945272795013
        'max_y': round(rm.max_dis_y),  # 3255,  # more precisely, 3255.4913305859623
        'v_car': 15,  # m/s, same with har1
        'num_human': 33,
    }
)


env_config['obs_range'] = min(env_config['max_x'], env_config['max_y']) / 4