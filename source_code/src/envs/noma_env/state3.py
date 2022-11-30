# inspection(ok): normalize[0,1][-1,1]cc_obsmax

import torch


class Building():
    def __init__(self, px, py, pz, r, data, env_config):
        self.px = px
        self.py = py
        self.pz = pz
        self.h = pz - 1e-3
        self.r = r
        self.data = data
        self.env_config = env_config

    def set_zero(self):
        self.px = self.py = self.pz = self.h = self.r = self.data = 0

    def to_tuple(self, normalize=False):
        if normalize:
            return (self.px / self.env_config['max_x'],
                    self.py / self.env_config['max_y'],
                    self.pz / self.env_config['max_z'],
                    self.h / self.env_config['max_z'],
                    self.r / self.env_config['max_r'],
                    self.data / self.env_config['max_data_amount']
                    )
        else:
            return self.px, self.py, self.pz, self.h, self.r, self.data

class UAVState():
    def __init__(self, px, py, pz, theta, energy, env_config):
        self.px = px
        self.py = py
        self.pz = pz
        self.theta = theta
        self.energy = energy
        self.env_config = env_config

    def set_zero(self):
        self.px = self.py = self.pz = self.theta = self.energy = 0

    def to_tuple(self, normalize=False):
        if normalize:
            return (self.px / self.env_config['max_x'],
                    self.py / self.env_config['max_y'],
                    (self.theta - self.env_config['theta_range'][0]) / (self.env_config['theta_range'][1] - self.env_config['theta_range'][0]),
                    self.energy / self.env_config['max_uav_energy']
                    )
        else:
            return self.px, self.py, self.theta, self.energy


class CarState():
    def __init__(self, px, py, theta, energy, env_config):
        self.px = px
        self.py = py
        self.pz = 0
        self.theta = theta
        self.energy = energy
        self.env_config = env_config

    def set_zero(self):
        self.px = self.py = self.pz = self.theta = self.energy = 0


    def to_tuple(self, normalize=False):
        if normalize:
            return (self.px / self.env_config['max_x'],
                        self.py / self.env_config['max_y'],
                        (self.theta - self.env_config['theta_range'][0]) / (self.env_config['theta_range'][1] - self.env_config['theta_range'][0]),
                        self.energy / self.env_config['max_car_energy']
                        )
        else:
            return self.px, self.py, self.theta, self.energy

class HumanState():
    def __init__(self, px, py, pz, theta, data, env_config):
        self.px = px
        self.py = py
        self.pz = pz
        self.theta = theta
        self.data = data
        self.env_config = env_config

    def set_zero(self):
        self.px = self.py = self.pz = self.theta = self.data = 0

    def to_tuple(self, normalize=False):
        if normalize:
            return (self.px / self.env_config['max_x'],
                    self.py / self.env_config['max_y'],
                    self.data / self.env_config['max_data_amount']
                    )
        else:
            return self.px, self.py, self.data

class JointState():
    def __init__(self, uav_states, car_states, human_states):
        for uav_state in uav_states:
            assert isinstance(uav_state, UAVState)
        for car_state in car_states:
            assert isinstance(car_state, CarState)
        for human_state in human_states:
            assert isinstance(human_state, HumanState)
        # for building_state in building_states:
        #     assert isinstance(building_state, Building)

        self.uav_states = uav_states
        self.car_states = car_states
        self.human_states = human_states
        # self.building_states = building_states

    def to_tensor(self, add_batchsize_dim=False, device=None, normalize=False):
        uav_tensor = torch.tensor([uav_state.to_tuple(normalize=True if normalize else False) for uav_state in self.uav_states],
                                           dtype=torch.float32)
        car_tensor = torch.tensor([car_state.to_tuple(normalize=True if normalize else False) for car_state in self.car_states],
                                           dtype=torch.float32)
        human_tensor = torch.tensor([human_state.to_tuple(normalize=True if normalize else False) for human_state in self.human_states],
                                           dtype=torch.float32)

        if add_batchsize_dim:
            uav_tensor = uav_tensor.unsqueeze(0)
            car_tensor = car_tensor.unsqueeze(0)
            human_tensor = human_tensor.unsqueeze(0)

        if device is not None:
            uav_tensor = uav_tensor.to(device)
            car_tensor = car_tensor.to(device)
            human_tensor = human_tensor.to(device)

        # uavhumanGNNuavhumantensor
        return uav_tensor, car_tensor, human_tensor
