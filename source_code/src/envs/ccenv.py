from src.envs.noma_env.utils import compute_distance

def get_ccenv(env_class):
    class CCEnv(env_class):
        def __init__(self, *args, **kwargs):
            env_class.__init__(self, *args, **kwargs)
            self.copo_kind = args[0]['args'].copo_kind  # copoinfonei_r, uav_r, car_r, global_r
            self.have_nei_freq = None  # seem to be deprecated

        def reset(self):
            self.have_nei_freq = {k: 0 for k in self.agent_name_list}  # have_nei_freq
            state = super(CCEnv, self).reset()
            return state

        def step(self, actions):
            '''stepinfonei_rewards, global_rewards, have_nei_freq'''
            o, mask, r, d, info = super(CCEnv, self).step(actions)
            # print(self.nei_matrix)
            def _find_neibours_by_nei_matrix(agent_name):
                '''nei_matrix'''
                u1 = self.agent_name_list.index(agent_name)
                neighbours = []
                for u2, agent_name in enumerate(info.keys()):
                    if self.nei_matrix[u1][u2] == 1:
                        dis = compute_distance(self.obj(u1), self.obj(u2))
                        if self.debug_use_nei_max_distance and dis > self.config["neighbours_distance"]:
                            print('')
                            continue
                        neighbours.append(agent_name)
                return neighbours

            def _find_uav_and_car_neibours_by_nei_matrix(agent_name):
                '''nei_matrix'''
                u1 = self.agent_name_list.index(agent_name)
                neighbours_uav = []
                neighbours_car = []
                for u2, agent_name in enumerate(info.keys()):
                    if self.nei_matrix[u1][u2] == 1:
                        dis = compute_distance(self.obj(u1), self.obj(u2))
                        if self.debug_use_nei_max_distance and dis > self.config["neighbours_distance"]:
                            print('')
                            continue
                        neighbours_uav.append(agent_name) if self.is_uav(u2) else neighbours_car.append(agent_name)
                return neighbours_uav, neighbours_car

            '''infonei_rglobal_r'''
            assert self.copo_kind in (1, 2)
            for agent_name in info.keys():
                # for copo_kind=1
                neighbours = _find_neibours_by_nei_matrix(agent_name)
                nei_rewards = [r[n] for n in neighbours]
                if nei_rewards:
                    info[agent_name]["nei_rewards"] = sum(nei_rewards) / len(nei_rewards)
                    self.have_nei_freq[agent_name] += 1
                else:  # OK =
                    info[agent_name]["nei_rewards"] = r[agent_name]
                # for copo_kind=2
                nei_uav, nei_car = _find_uav_and_car_neibours_by_nei_matrix(agent_name)
                nei_uav_rewards = [r[n] for n in nei_uav]
                nei_car_rewards = [r[n] for n in nei_car]
                if nei_uav_rewards:
                    info[agent_name]["uav_rewards"] = sum(nei_uav_rewards) / len(nei_uav_rewards)
                else:
                    info[agent_name]["uav_rewards"] = r[agent_name]  # OK uav=
                if nei_car_rewards:
                    info[agent_name]["car_rewards"] = sum(nei_car_rewards) / len(nei_car_rewards)
                else:
                    info[agent_name]["car_rewards"] = r[agent_name]  # OK car=
                info[agent_name]["global_rewards"] = sum(r.values()) / len(r.values())
                if d['__all__']:
                    info[agent_name]['have_nei_freq'] = self.have_nei_freq[agent_name] / self.num_timestep
            return o, mask, r, d, info

    name = env_class.__name__
    name = f"CC{name}"
    CCEnv.__name__ = name
    CCEnv.__qualname__ = name
    return CCEnv
