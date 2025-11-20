#!/usr/bin/python3

import numpy as np
import gym

class RL():
    def __init__(self, config):
        self.config = config
    
    def normalize_observations(self, observation_space, observation):
        for key in observation.keys():
            x = observation[key]
            x_min = observation_space.spaces[key].low
            x_max = observation_space.spaces[key].high
            observation[key] = (x-x_min)/(x_max-x_min)
        return observation
    
    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces.

        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)