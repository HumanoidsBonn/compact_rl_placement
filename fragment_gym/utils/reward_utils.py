#!/usr/bin/python3

import numpy as np

from fragment_gym.utils import math_function_utils

class Reward():
    def __init__(self, config, sim, p_id):
        self.config = config
        self.sim = sim
        self._p = p_id

        # Instatiate classes
        self.math_function_utils = math_function_utils.MathFunction()

        # Initialize Rewards
        self.reward = 0.0
        self.reward_keys = self.config["reward_keys"]
        if self.sim.ablation == "no_ruler":
            if "ruler_closeness" in self.reward_keys:
                self.reward_keys.remove("ruler_closeness")
            if "ruler_distance" in self.reward_keys:
                self.reward_keys.remove("ruler_distance")
        self.rewards = dict.fromkeys(self.reward_keys, 0.0)

        self.tensorboard_reward = self.reward
        self.tensorboard_rewards = self.rewards.copy()

        # Sparse rewards and penalties
        if "corner_closeness" in self.reward_keys:
            self.reward_corner_closeness = float(self.config["reward_corner_closeness"])
        if "ruler_closeness" in self.reward_keys:
            self.reward_ruler_closeness = float(self.config["reward_ruler_closeness"])
        if "fragment_angle_closeness" in self.reward_keys:
            self.reward_fragment_angle_closeness = float(self.config["reward_fragment_angle_closeness"])
        if "plane_contact" in self.reward_keys:
            self.penalty_plane_contact = float(self.config["penalty_plane_contact"])
        if "fragment_contact" in self.reward_keys:
            self.penalty_fragment_contact = float(self.config["penalty_fragment_contact"])

        # Scale factors
        if "corner_distance" in self.reward_keys:
            self.penalty_corner_distance_scale_factor = float(self.config["penalty_corner_distance_scale_factor"])
        if "ruler_distance" in self.reward_keys:
            self.penalty_ruler_distance_scale_factor = float(self.config["penalty_ruler_distance_scale_factor"])
        if "fragment_angle_difference" in self.reward_keys:
            self.penalty_placing_fragment_angle_difference_scale_factor = float(self.config["penalty_placing_fragment_angle_difference_scale_factor"])
        if "corner_closeness" in self.reward_keys:
            self.reward_corner_closeness_scale_factor = float(self.config["reward_corner_closeness_scale_factor"])
        if "ruler_closeness" in self.reward_keys:
            self.reward_ruler_closeness_scale_factor = float(self.config["reward_ruler_closeness_scale_factor"])
        if "fragment_angle_closeness" in self.reward_keys:
            self.reward_fragment_angle_closeness_scale_factor = float(self.config["reward_fragment_angle_closeness_scale_factor"])

        # Functions
        if "corner_closeness" in self.reward_keys:
            self.reward_corner_closeness_function = str(self.config["reward_corner_closeness_function"])
        if "ruler_closeness" in self.reward_keys:
            self.reward_ruler_closeness_function = str(self.config["reward_ruler_closeness_function"])

        # Limits
        self.fragment_to_target_distance_limits = np.array(self.config["fragment_to_target_distance_limits"], dtype=np.float32)
        self.fragment_to_target_yaw_angle_limits = np.deg2rad(np.array(self.config["fragment_to_target_yaw_angle_limits"], dtype=np.float32))
        if "fragment_closeness_limits" in self.config:
            self.fragment_closeness_limits = np.array(self.config["fragment_closeness_limits"], dtype=np.float32)
        else:
            self.fragment_closeness_limits = self.fragment_to_target_distance_limits
        self.gripper_yaw_angle_limits = np.deg2rad(np.array(self.config["gripper_yaw_angle_limits"], dtype=np.float32))

        # Multy target rewards
        self.previous_state = 0
        if "drop_height" in self.reward_keys:
            self.penalty_drop_factor=float(self.config["penalty_drop_factor"])


    def compute_reward(self):
        # Reset all rewards
        self.reward = 0.0
        for key in self.rewards.keys():
            self.rewards[key] = 0.0
        
        # Compute the reward that is defined in the config file
        current_reward = self.compute_reward_components(self.sim.placing_fragment["first_fragment"], self.sim.placing_fragment["ruler_fragment"], self.sim.current_corresponding_corner_distance, self.sim.current_ruler_distance, self.sim.current_target_distance, self.sim.plane_contact, self.sim.frag_contact, self.sim.state, self.sim.current_fragment_to_table_height, self.sim.euclidian_drop_pose_distance, self.sim.drop_pose_angle_difference, self.sim.drop_corner_distance, self.sim.drop_ruler_distance, self.sim.current_placing_fragment_angle)
        
        return current_reward
    
    def compute_reward_components(self, first_fragment, ruler_fragment, current_corresponding_corner_distance, current_ruler_distance, current_target_distance, plane_contact, frag_contact, state, current_fragment_to_table_height=0.0, euclidian_drop_pose_distance=0.0, drop_pose_angle_difference=0.0, drop_corner_distance=0.0, drop_ruler_distance=0.0, current_placing_fragment_angle=0.0):       
        if state != self.previous_state:
            state_transition = True
            self.previous_state = state
        else:
            state_transition = False

        # Sparse rewards
        if "plane_contact" in self.reward_keys and plane_contact:
            self.rewards["plane_contact"] = self.penalty_plane_contact
            
        if "fragment_contact" in self.reward_keys and frag_contact:
            self.rewards["fragment_contact"] = self.penalty_fragment_contact

        if state == 1 and state_transition:
            # Reward corner closeness
            if "corner_closeness" in self.reward_keys and (first_fragment == False or self.sim.ablation == "no_ruler"):
                normed_drop_corner_distance = self.sim.common_utils.normalize_value(self.fragment_closeness_limits[0], self.fragment_closeness_limits[1], drop_corner_distance)
                if self.reward_corner_closeness_function == "tanh":
                    partial_reward_corner_closeness = self.math_function_utils.positive_tanh(x=normed_drop_corner_distance, x_factor=self.reward_corner_closeness_scale_factor, y_factor=self.reward_corner_closeness)
                elif self.reward_corner_closeness_function == "quotient_x":
                    partial_reward_corner_closeness = self.math_function_utils.positive_quotient_x(x=normed_drop_corner_distance, x_factor=self.reward_corner_closeness_scale_factor, y_factor=self.reward_corner_closeness)
                elif self.reward_corner_closeness_function == "linear":
                    partial_reward_corner_closeness = self.math_function_utils.positive_linear(x=normed_drop_corner_distance, x_factor=self.reward_corner_closeness_scale_factor, y_factor=self.reward_corner_closeness)
                self.rewards["corner_closeness"] = partial_reward_corner_closeness

            if "ruler_closeness" in self.reward_keys and ruler_fragment == True:
                # Reward ruler closeness
                normed_drop_ruler_distance = self.sim.common_utils.normalize_value(self.fragment_closeness_limits[0], self.fragment_closeness_limits[1], drop_ruler_distance)
                if self.reward_ruler_closeness_function == "tanh":
                    partial_reward_ruler_closeness = self.math_function_utils.positive_tanh(x=normed_drop_ruler_distance, x_factor=self.reward_ruler_closeness_scale_factor, y_factor=self.reward_ruler_closeness)
                elif self.reward_ruler_closeness_function == "quotient_x":
                    partial_reward_ruler_closeness = self.math_function_utils.positive_quotient_x(x=normed_drop_ruler_distance, x_factor=self.reward_ruler_closeness_scale_factor, y_factor=self.reward_ruler_closeness)
                elif self.reward_ruler_closeness_function == "linear":
                    partial_reward_ruler_closeness = self.math_function_utils.positive_linear(x=normed_drop_ruler_distance, x_factor=self.reward_ruler_closeness_scale_factor, y_factor=self.reward_ruler_closeness)
                self.rewards["ruler_closeness"] = partial_reward_ruler_closeness

            if "fragment_angle_closeness" in self.reward_keys:
                normed_placing_fragment_drop_angle_difference = self.sim.common_utils.normalize_value(0.0, self.fragment_to_target_yaw_angle_limits[1], abs(current_placing_fragment_angle))
                partial_reward_fragment_angle_closeness = self.math_function_utils.positive_tanh(x=normed_placing_fragment_drop_angle_difference, x_factor=self.reward_fragment_angle_closeness_scale_factor, y_factor=self.reward_fragment_angle_closeness)
                self.rewards["fragment_angle_closeness"] = partial_reward_fragment_angle_closeness

        # Penalty drop height
        if "drop_height" in self.reward_keys and state == 1 and state_transition:
            normed_current_fragment_to_table_height = self.sim.common_utils.normalize_value(self.fragment_to_target_distance_limits[0], self.fragment_to_target_distance_limits[1], abs(current_fragment_to_table_height))
            partial_drop_height_reward = self.penalty_drop_factor*normed_current_fragment_to_table_height
            self.rewards["drop_height"] = partial_drop_height_reward

        # Continous reward
        if state == 0 or state == 1:
            if "corner_distance" in self.reward_keys and (first_fragment == False or self.sim.ablation == "no_ruler"):
                normed_current_corresponding_corner_distance = self.sim.common_utils.normalize_value(self.fragment_to_target_distance_limits[0], self.fragment_to_target_distance_limits[1], current_corresponding_corner_distance)
                partial_reward_corner_distance = self.penalty_corner_distance_scale_factor * normed_current_corresponding_corner_distance
                self.rewards["corner_distance"] = partial_reward_corner_distance

            if "ruler_distance" in self.reward_keys and ruler_fragment == True:
                normed_current_ruler_distance = self.sim.common_utils.normalize_value(self.fragment_to_target_distance_limits[0], self.fragment_to_target_distance_limits[1], current_ruler_distance)
                partial_reward_ruler_distance = self.penalty_ruler_distance_scale_factor * normed_current_ruler_distance
                self.rewards["ruler_distance"] = partial_reward_ruler_distance

            if "fragment_angle_difference" in self.reward_keys:
                normed_placing_fragment_angle_difference = self.sim.common_utils.normalize_value(0.0, self.fragment_to_target_yaw_angle_limits[1], abs(current_placing_fragment_angle))
                #debug_degree = np.rad2deg(current_placing_fragment_angle)
                partial_reward_placing_fragment_angle_difference = self.penalty_placing_fragment_angle_difference_scale_factor * normed_placing_fragment_angle_difference
                self.rewards["fragment_angle_difference"] = partial_reward_placing_fragment_angle_difference

        self.reward = sum(self.rewards.values())

        # Tensorboard
        self.tensorboard_reward += self.reward
        for key in self.tensorboard_rewards.keys():
            self.tensorboard_rewards[key] += self.rewards[key]
        
        return self.reward