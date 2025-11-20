#!/usr/bin/python3

from stable_baselines3.common.env_util import make_vec_env
from fragment_gym.env.main_env import MainEnv
import numpy as np
import gym
from collections import OrderedDict, Counter
from fragment_gym.utils import debug_utils

class GraspeAndPlace(MainEnv):
    def __init__(self, mode=None, ablation=None, config=None, real_time=None, debug=None, render=False, shared_memory=False, use_egl=False, fresco_range=[]):
        super().__init__(mode=mode, ablation=ablation, config=config, real_time=real_time, debug=debug, render=render, shared_memory=shared_memory, use_egl=use_egl, fresco_range=fresco_range)
        
        # Config parameters
        # ==============================================================================
        # Task
        self.fresco_no = self.used_frescoes[0]
        self.current_episode = 0
        self.current_iteration = 0
        self.use_curriculum_learning = self.config["use_curriculum_learning"]
        self.current_curriculum_step = 0
        self.start_amount_of_fragments_on_table = 0
        self.place_complete_fresco = False
        self.model_name = ""
        self.drop_height_threshold = self.config["drop_height_threshold"]
        if "place_complete_fresco" in self.config:
            self.place_complete_fresco = self.config["place_complete_fresco"]
        self.gripper_opening_after_grasp = 0.0

        # Termination
        self.terminate_on_plane_contact = self.config.get("terminate_on_plane_contact", True)
        self.terminate_on_fragment_contact = self.config.get("terminate_on_fragment_contact", True)
        self.terminate_on_too_close = self.config.get("terminate_on_too_close", False)

        # Legacy
        self.current_target_distance = 0.0
        self.euclidian_drop_pose_distance = 0.0
        self.drop_pose_angle_difference = 0.0
        
        # Observation space limits
        self.arm_joint_limits = np.deg2rad(np.array(self.config["arm_joint_limits"], dtype=np.float32))
        self.gripper_position_limits = np.array(self.config["gripper_position_limits"], dtype=np.float32)
        self.gripper_yaw_angle_limits = np.deg2rad(np.array(self.config["gripper_yaw_angle_limits"], dtype=np.float32))
        self.fragment_to_target_distance_limits = np.array(self.config["fragment_to_target_distance_limits"], dtype=np.float32)
        self.use_3d_normalization = self.config.get("use_3d_normalization", False)
        if self.use_3d_normalization:
            self.fragment_to_target_3d_distance_limits = np.array(self.config["fragment_to_target_3d_distance_limits"], dtype=np.float32)
        else:
            self.fragment_to_target_3d_distance_limits = self.fragment_to_target_distance_limits
        self.fragment_to_target_yaw_angle_limits = np.deg2rad(np.array(self.config["fragment_to_target_yaw_angle_limits"], dtype=np.float32))
        self.use_min_3d_observations = self.config.get("use_min_3d_observations", False)
        if "use_placing_object_overlap_in_obs" in self.config:
            self.placing_object_overlap_limits = np.array(self.config["placing_object_overlap_limits"], dtype=np.float32)
            self.overlap_inflation = self.config.get("overlap_inflation", 0.0)

        # Max number of neighbours that serve as placing reference
        # Max number of neighbours (5+1 for safety =6) each with 2 corresponding corners and 3 dimensioons (x,y,z) -> e.g. 6x2x3=36
        self.max_no_of_corner_distances = int(self.config["max_expected_placement_neighbours"])*2*3
        # Max number of ruler distances: 2 ruler lines each with 2 corresponding corners and 3 dimensioons (x,y,z) -> e.g. 2x2x3=12
        self.max_no_of_ruler_distances = 2*2*3
        
        self.use_min_placing_object_dist_in_obs = self.config.get("use_min_placing_object_dist_in_obs", False)
        self.use_min_robot_to_table_objects_dist_in_obs = self.config.get("use_min_robot_to_table_objects_dist_in_obs", False)
        self.use_placing_object_overlap_in_obs = self.config.get("use_placing_object_overlap_in_obs", False)
        self.attach_footprint_to_overlap_in_obs = self.config.get("attach_footprint_to_overlap_in_obs", False)
        # if self.use_min_placing_object_dist_in_obs:
        self.min_placing_object_euc_dist = self.fragment_to_target_distance_limits[1] #np.finfo(np.float32).max
        self.min_placing_object_3d_dist = np.array(3*[self.fragment_to_target_3d_distance_limits[1]], dtype=np.float32)
        self.too_close_threshold = self.config.get("too_close_threshold", 0.0)
        # if self.use_min_robot_to_table_objects_dist_in_obs:
        self.min_robot_to_table_objects_euc_dist = self.fragment_to_target_distance_limits[1] #np.finfo(np.float32).max
        self.min_robot_to_table_objects_3d_dist = np.array(3*[self.fragment_to_target_3d_distance_limits[1]], dtype=np.float32)
        
        # Reward and tolerances
        self.current_fragment_to_table_height = 0.0
        self.drop_corner_distance = 0.0
        self.drop_ruler_distance = 0.0
        self.corner_distance_power = self.config.get("corner_distance_power", 1)
        self.corner_distance_root = self.config.get("corner_distance_root", False)
        self.ruler_distance_power = self.config.get("ruler_distance_power", 1)
        self.ruler_distance_root = self.config.get("ruler_distance_root", False)

        self.tensorboard_corresponding_corner_distance = 0.0
        self.tensorboard_ruler_distance = 0.0

        # Set action and observation space
        # ==============================================================================
        self._set_action_space()
        self._set_observation_space()

        # Fragment        
        # ==============================================================================
        self.used_pybullet_frag_ids_sorted = []
        self.used_frag_ids_sorted = []
        self.removed_fix_fragment_to_tool_tip_constraint = False

        # Instantiate debug stuff
        # ==============================================================================
        self.reference_debug_cross = None
        self.fragment_debug_cross = None
        self.ruler_debug_lines = None
        self.corresponding_corners_debug_lines = None
        self.ruler_corners_debug_lines = None     

        # Tensorboard
        # ==============================================================================
        self.tensorboard_fragment_drop_height = 0.0
        self.tensorboard_drop_corner_distance = 0.0
        self.tensorboard_drop_ruler_distance = 0.0
        self.tensorboard_drop_target_angle = 0.0
        
        orange = [150/255, 53/255, 10/255]
        darkorange = [255/255, 140/255, 0/255]
        white = [1, 1, 1]

        self.ruler_color = white
        self.ruler_corner_color = orange
        self.corner_color = darkorange

    def _set_action_space(self):
        self.arm_action_position_increment_value = float(self.config["arm_action_position_increment_value"])
        self.arm_action_yaw_angle_increment_value = np.deg2rad(float(self.config["arm_action_yaw_angle_increment_value"]), dtype=np.float32)
        self.action_space = gym.spaces.box.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
    
    def _set_observation_space(self):       
        observation_space = OrderedDict()

        # 1D VECTOR
        # ==============================================================================
        low_gripper_position = 3*[self.gripper_position_limits[0]]
        high_gripper_position = 3*[self.gripper_position_limits[1]]
       
        low_gripper_yaw_angle = [self.gripper_yaw_angle_limits[0]]
        high_gripper_yaw_angle = [self.gripper_yaw_angle_limits[1]]

        low_corner_distances = self.max_no_of_corner_distances*[self.fragment_to_target_3d_distance_limits[0]]
        high_corner_distances = self.max_no_of_corner_distances*[self.fragment_to_target_3d_distance_limits[1]]

        low_ruler_distances = self.max_no_of_ruler_distances*[self.fragment_to_target_3d_distance_limits[0]]
        high_ruler_distances = self.max_no_of_ruler_distances*[self.fragment_to_target_3d_distance_limits[1]]
        
        if self.use_min_placing_object_dist_in_obs:
            if self.use_min_3d_observations:
                low_min_placing_object_dist = 3*[self.fragment_to_target_3d_distance_limits[0]]
                high_min_placing_object_dist = 3*[self.fragment_to_target_3d_distance_limits[1]]            
            else:
                low_min_placing_object_dist = 1*[self.fragment_to_target_distance_limits[0]]
                high_min_placing_object_dist = 1*[self.fragment_to_target_distance_limits[1]]

        if self.use_min_robot_to_table_objects_dist_in_obs:
            if self.use_min_3d_observations:
                low_min_robot_to_table_objects_dist = 3*[self.fragment_to_target_3d_distance_limits[0]]
                high_min_robot_to_table_objects_dist = 3*[self.fragment_to_target_3d_distance_limits[1]]               
            else:
                low_min_robot_to_table_objects_dist = 1*[self.fragment_to_target_distance_limits[0]]
                high_min_robot_to_table_objects_dist = 1*[self.fragment_to_target_distance_limits[1]]
        
        if self.use_placing_object_overlap_in_obs:
            low_min_placing_object_overlap = 1*[self.placing_object_overlap_limits[0]]
            high_min_placing_object_overlap = 1*[self.placing_object_overlap_limits[1]]

        low_fragment_angle = [self.fragment_to_target_yaw_angle_limits[0]]
        high_fragment_angle = [self.fragment_to_target_yaw_angle_limits[1]]
       
        if self.use_min_placing_object_dist_in_obs and self.use_placing_object_overlap_in_obs and self.use_min_robot_to_table_objects_dist_in_obs:
            if self.ablation == "no_ruler":
                low = np.array([*low_gripper_position, *low_gripper_yaw_angle, *low_min_placing_object_dist, *low_min_robot_to_table_objects_dist, *low_min_placing_object_overlap, *low_corner_distances, *low_fragment_angle])
                high = np.array([*high_gripper_position, *high_gripper_yaw_angle, *high_min_placing_object_dist, *high_min_robot_to_table_objects_dist, *high_min_placing_object_overlap, *high_corner_distances, *high_fragment_angle])             
            else:
                low = np.array([*low_gripper_position, *low_gripper_yaw_angle, *low_min_placing_object_dist, *low_min_robot_to_table_objects_dist, *low_min_placing_object_overlap, *low_corner_distances, *low_ruler_distances, *low_fragment_angle])
                high = np.array([*high_gripper_position, *high_gripper_yaw_angle, *high_min_placing_object_dist, *high_min_robot_to_table_objects_dist, *high_min_placing_object_overlap, *high_corner_distances, *high_ruler_distances, *high_fragment_angle])
        elif self.use_min_placing_object_dist_in_obs and self.use_placing_object_overlap_in_obs:
            if self.ablation == "no_ruler":
                low = np.array([*low_gripper_position, *low_gripper_yaw_angle, *low_min_placing_object_dist, *low_min_placing_object_overlap, *low_corner_distances, *low_fragment_angle])
                high = np.array([*high_gripper_position, *high_gripper_yaw_angle, *high_min_placing_object_dist, *high_min_placing_object_overlap, *high_corner_distances, *high_fragment_angle])             
            else:
                low = np.array([*low_gripper_position, *low_gripper_yaw_angle, *low_min_placing_object_dist, *low_min_placing_object_overlap, *low_corner_distances, *low_ruler_distances, *low_fragment_angle])
                high = np.array([*high_gripper_position, *high_gripper_yaw_angle, *high_min_placing_object_dist, *high_min_placing_object_overlap, *high_corner_distances, *high_ruler_distances, *high_fragment_angle])
        elif self.use_min_placing_object_dist_in_obs:
            if self.ablation == "no_ruler":
                low = np.array([*low_gripper_position, *low_gripper_yaw_angle, *low_min_placing_object_dist, *low_corner_distances, *low_fragment_angle])
                high = np.array([*high_gripper_position, *high_gripper_yaw_angle, *high_min_placing_object_dist, *high_corner_distances, *high_fragment_angle])             
            else:
                low = np.array([*low_gripper_position, *low_gripper_yaw_angle, *low_min_placing_object_dist, *low_corner_distances, *low_ruler_distances, *low_fragment_angle])
                high = np.array([*high_gripper_position, *high_gripper_yaw_angle, *high_min_placing_object_dist, *high_corner_distances, *high_ruler_distances, *high_fragment_angle])
        else:
            if self.ablation == "no_ruler":
                low = np.array([*low_gripper_position, *low_gripper_yaw_angle, *low_corner_distances, *low_fragment_angle])
                high = np.array([*high_gripper_position, *high_gripper_yaw_angle, *high_corner_distances, *high_fragment_angle])          
            else:
                low = np.array([*low_gripper_position, *low_gripper_yaw_angle, *low_corner_distances, *low_ruler_distances, *low_fragment_angle])
                high = np.array([*high_gripper_position, *high_gripper_yaw_angle, *high_corner_distances, *high_ruler_distances, *high_fragment_angle])
        
        observation_space["vector_1d"] = gym.spaces.box.Box(low=low, high=high)

        # FINALIZING
        # ==============================================================================
        self.observation_space = gym.spaces.Dict(observation_space)

    def reset(self):
        if self.mode == "eval" and self.place_complete_fresco and len(self.used_frag_ids_sorted) == len(self.frag_ids_sorted[self.fresco_no])-1:
            if self.mode == "eval":
                self.evaluation_utils.evaluate_rl_agent()
            self.reset_contact_history() # only for evaluation

        if self.debug == True:
            print("RESET")
        if self.render == False:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
        self.reset_task()
        if self.render == False:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        return self.update_observations()

    def switch_state(self):
        try:
            if self.debug and self.render:
                if self.state == 0:
                        del self.corresponding_corners_debug_lines
                        del self.ruler_corners_debug_lines
                
            if self.debug and self.real_time == False and self.render == True:
                del self.reference_debug_cross
        except:
            pass

        if self.state == 0:
            if self.mode == "train" or self.mode == "test":
                self.fresco_no = np.random.default_rng().integers(self.used_frescoes[0], self.used_frescoes[1], endpoint=True)      
            
            # Get neighbour dictionary
            self.gt_data = self.shapely_utils.get_fresco_ground_truth(fresco_no=self.fresco_no, root_path=self.root_path)
            self.no_fragments = self.gt_data["header"]["no_fragments"]
            neighbours = self.gt_data["neighbours"]
            self.fresco_dimensions = self.gt_data["header"].copy()
            self.fresco_dimensions.pop("no_fragments")
            fresco_center_location=self.config["fresco_assembly_center_location"]
            
            # Spawn new fragment
            if self.debug:
                print("Spawn new fragment")

            if self.fragment_format == "urdf":
                self.used_pybullet_frag_ids_sorted, self.fix_fragment_to_tool_tip_constraint, self.layout_placing_fragment_centroid_pose = self.spawn_urdf_fragments_on_table_and_in_gripper(self.fresco_no, scale_factor=float(self.config["fresco_scale_factor"]))
            elif self.fragment_format == "stl" or self.fragment_format == "vhacd":
                if self.place_complete_fresco and len(self.used_frag_ids_sorted) > self.start_amount_of_fragments_on_table:
                    _, _, self.frag_pybullet_id, frag_id, assembly_id, self.fix_fragment_to_tool_tip_constraint, self.layout_placing_fragment_centroid_pose= self.move_fragments_on_table_and_in_gripper_without_retraction(frag_ids_sorted=self.frag_ids_sorted[self.fresco_no], fresco_center_location=self.config["fresco_assembly_center_location"], use_curriculum_learning=self.use_curriculum_learning, current_curriculum_step=self.current_curriculum_step, place_complete_fresco=self.place_complete_fresco, amount_of_fragments_on_table=len(self.used_frag_ids_sorted))
                else:
                    if self.place_complete_fresco == False:
                        self.start_amount_of_fragments_on_table = -1
                    self.used_pybullet_frag_ids_sorted, self.used_frag_ids_sorted, self.frag_pybullet_id, frag_id, assembly_id, self.fix_fragment_to_tool_tip_constraint, self.layout_placing_fragment_centroid_pose = self.move_fragments_on_table_and_in_gripper_without_retraction(frag_ids_sorted=self.frag_ids_sorted[self.fresco_no], fresco_center_location=self.config["fresco_assembly_center_location"], use_curriculum_learning=self.use_curriculum_learning, current_curriculum_step=self.current_curriculum_step, amount_of_fragments_on_table=self.start_amount_of_fragments_on_table)

            self.step_simulation(self.per_step_iterations)
            self.gripper_opening_after_grasp = self.get_current_gripper_opening()

            # Ruler lines
            x_bottom_left = fresco_center_location[0] - self.fresco_dimensions["length"]/2
            y_bottom_left = fresco_center_location[1] - self.fresco_dimensions["width"]/2
            
            x_line = [(x_bottom_left, y_bottom_left, self.fresco_dimensions["height"]/2), (x_bottom_left+1.0, y_bottom_left, self.fresco_dimensions["height"]/2)] # line parallel to x-axis at z of half the fragment height
            y_line = [(x_bottom_left, y_bottom_left, self.fresco_dimensions["height"]/2), (x_bottom_left, y_bottom_left+1.0, self.fresco_dimensions["height"]/2)] # line parallel to y-axis at z of half the fragment height

            self.ruler_lines = [x_line, y_line]

            # frag_ids_sorted = [0]frag_pybullet_id, [1]frag_id, [2]assembly_no, [3]centroid, [4]yaw
            self.create_placing_fragment_dict(self.fresco_no, assembly_id, self.gt_data)
            self.create_table_fragment_dict(self.fresco_no, assembly_id, self.gt_data)

            # Get current coordinates of corresponding corners
            if self.ablation == "no_ruler":
                #Treat special case of first placing fragment
                if len(self.table_fragments) == 0:
                    temp_x_virtual_fragments = {}
                    if str(self.no_fragments) in self.placing_fragment["corresponding_corners"]:
                        temp_x_virtual_fragments = self.placing_fragment["corresponding_corners"][str(self.no_fragments)].copy()
                    temp_y_virtual_fragments = {}
                    if str(self.no_fragments+1) in self.placing_fragment["corresponding_corners"]:
                        temp_y_virtual_fragments = self.placing_fragment["corresponding_corners"][str(self.no_fragments+1)].copy()
                    temp_id = self.placing_fragment["id"]
                    self.table_fragments[temp_id] = {}
                    self.table_fragments[temp_id] = self.placing_fragment.copy()
                    self.table_fragments[temp_id]["centroid"] = self.layout_placing_fragment_centroid_pose[:3]
                    self.table_fragments[temp_id].pop("corresponding_corners")
                    self.table_fragments[temp_id]["corresponding_corners"] = {}
                    self.table_fragments[temp_id]["corresponding_corners"][str(temp_id)] = {}

                    temp_list_fresco_coordinates = []
                    temp_list_fragment_coordinates = []
                    temp_fresco_key_list = []
                    temp_fragment_key_list = []

                    for key in self.placing_fragment["corresponding_corners"]:
                        temp_fresco_values = self.placing_fragment["corresponding_corners"][key]["fresco_coordinates"].values()
                        temp_fresco_keys = self.placing_fragment["corresponding_corners"][key]["fresco_coordinates"].keys()
                        temp_list_fresco_coordinates.extend(temp_fresco_values)
                        temp_fresco_key_list.extend(temp_fresco_keys)

                        temp_fragment_values = self.placing_fragment["corresponding_corners"][key]["fragment_coordinates"].values()
                        temp_fragment_keys = self.placing_fragment["corresponding_corners"][key]["fragment_coordinates"].keys()
                        temp_list_fragment_coordinates.extend(temp_fragment_values)
                        temp_fragment_key_list.extend(temp_fragment_keys)

                    self.table_fragments[temp_id]["corresponding_corners"][str(temp_id)]["fresco_coordinates"] = dict(zip(temp_fresco_key_list, temp_list_fresco_coordinates))
                    self.table_fragments[temp_id]["corresponding_corners"][str(temp_id)]["fragment_coordinates"] = dict(zip(temp_fragment_key_list, temp_list_fragment_coordinates))

                    self.placing_fragment.pop("corresponding_corners")
                    self.placing_fragment["corresponding_corners"] = self.table_fragments[temp_id]["corresponding_corners"]
                    
                    if len(temp_x_virtual_fragments) > 0:
                        self.placing_fragment["corresponding_corners"][str(self.no_fragments)] = temp_x_virtual_fragments
                    if len(temp_y_virtual_fragments) > 0:
                        self.placing_fragment["corresponding_corners"][str(self.no_fragments+1)] = temp_y_virtual_fragments
                

            # Get location of current corresponding corners
            self.current_corners, self.current_ruler_corners, self.current_x_ruler_corners, self.current_y_ruler_corners = self.get_current_corresponding_corner_transforms(self.placing_fragment, self.table_fragments, self.no_fragments, self.ruler_lines)
            self.current_3d_corner_distances, self.current_1d_corner_distances, self.current_corresponding_corner_distance = self.calculate_corner_distances(current_corners=self.current_corners, corner_distance_power=self.corner_distance_power, corner_distance_root=self.corner_distance_root)
            self.current_3d_ruler_distances, self.current_1d_ruler_distances, self.current_ruler_distance = self.calculate_corner_distances(current_corners=self.current_ruler_corners, corner_distance_power=self.ruler_distance_power, corner_distance_root=self.ruler_distance_root)
            self.current_3d_x_ruler_distances, _, _ = self.calculate_corner_distances(current_corners=self.current_x_ruler_corners, corner_distance_power=self.ruler_distance_power, corner_distance_root=self.ruler_distance_root)
            self.current_3d_y_ruler_distances, _, _ = self.calculate_corner_distances(current_corners=self.current_y_ruler_corners, corner_distance_power=self.ruler_distance_power, corner_distance_root=self.ruler_distance_root)

            # Init placing fragment angle
            self.current_placing_fragment_angle = self.get_fragment_pose(self.placing_fragment["pybullet_id"])[3]

            if self.debug and self.real_time == False and self.render == True:
                self.reference_marker_pose = self.get_fragment_pose(self.placing_fragment["pybullet_id"])
                self.reference_debug_cross = debug_utils.DebugCross(config=self.config, sim=self, p_id=self._p, position=self.reference_marker_pose, line_length = 0.05, line_width = 10.0, line_color=[0,1,0])

            if (self.debug and self.real_time == False and self.render == True):
                try:
                    del self.corresponding_corners_debug_lines
                except:
                    pass
                if self.placing_fragment["first_fragment"]  == False or self.ablation == "no_ruler":
                    self.corresponding_corners_debug_lines = debug_utils.DebugLinesCorrespondingCorners(config=self.config, sim=self, p_id=self._p, current_corners=self.current_corners, line_length = 0.05, line_width = 10.0, line_color=self.corner_color)
                if self.placing_fragment["ruler_fragment"] == True and self.ablation != "no_ruler":
                    self.ruler_corners_debug_lines = debug_utils.DebugLinesCorrespondingCorners(config=self.config, sim=self, p_id=self._p, current_corners=self.current_ruler_corners, line_length = 0.05, line_width = 10.0, line_color=self.ruler_corner_color)
            if self.debug and self.real_time == False and self.render == True:
                try:
                    del self.ruler_debug_lines
                except:
                    pass
                self.ruler_debug_lines = debug_utils.DebugLineRuler(config=self.config, sim=self, p_id=self._p, ruler_lines=self.ruler_lines, line_length = 0.05, line_width = 10.0, line_color=self.ruler_color)

        elif self.state == 1:           
            self.current_fragment_to_table_height = self.get_fragment_pose(self.placing_fragment["pybullet_id"])[2]
            self.tensorboard_fragment_drop_height = self.current_fragment_to_table_height

            if self.debug:
                print("Releasing the fragment")

            self._p.removeConstraint(self.fix_fragment_to_tool_tip_constraint, physicsClientId=self.client_id)
            self.removed_fix_fragment_to_tool_tip_constraint = True

            # collision_check_reference is for frag2frag collision
            self.update_contacts(*self.control_gripper(close=False, release=True, release_obj_id=self.placing_fragment["pybullet_id"], coll_check="count", coll_check_ids=self.used_pybullet_frag_ids_sorted, coll_check_ref_id=self.placing_fragment["pybullet_id"]))

    def reset_task(self):
        # State 0: Spawn fragment and place
        # State 1: Open and retract gripper
        # State 2: Retract gripper
        self.state = 0

        if self.current_episode != 0 and self.place_complete_fresco == True and len(self.used_frag_ids_sorted) >= len(self.frag_ids_sorted[self.fresco_no])-1:
            last_fragment = True
        else:
            last_fragment = False

        # reset world
        if len(self.used_pybullet_frag_ids_sorted) > 0 and self.removed_fix_fragment_to_tool_tip_constraint == False:
            self._p.removeConstraint(self.fix_fragment_to_tool_tip_constraint, physicsClientId=self.client_id)
            self.step_simulation(self.per_step_iterations)
        self.removed_fix_fragment_to_tool_tip_constraint = False

        constraint_list_before = self.get_all_constraint_ids()
        self.remove_all_contraints_except(self.robot_constraints)
        constraint_list_after = self.get_all_constraint_ids()

        if Counter(constraint_list_before) != Counter(constraint_list_after):
            print("Unwanted contraint was removed during reset")
            print("constraint ids before reset =\n",constraint_list_before)
            print("constraint ids after reset =\n",constraint_list_after)

        try:
            if self.debug and self.render:
                if self.state == 0:
                        del self.corresponding_corners_debug_lines
                        del self.ruler_corners_debug_lines
                        del self.ruler_debug_lines
            if self.debug and self.real_time == False and self.render == True:
                del self.reference_debug_cross
        except:
            pass

        if self.use_curriculum_learning == False:
            self.reset_robot(self.initial_parameters)

        if self.current_episode != 0 and (self.place_complete_fresco == False or len(self.used_frag_ids_sorted) >= len(self.frag_ids_sorted[self.fresco_no])-1):
            self.remove_mass_from_fragment(self.used_pybullet_frag_ids_sorted)
            self.remove_mass_from_fragment(self.placing_fragment["pybullet_id"])
            self.reset_sim(self.world_id)
            self.step_simulation(self.per_step_iterations)

        if self.place_complete_fresco == True and self.current_episode != 0 and len(self.used_frag_ids_sorted) < len(self.frag_ids_sorted[self.fresco_no])-1:
            self.used_pybullet_frag_ids_sorted.append(self.placing_fragment["pybullet_id"])
            self.used_frag_ids_sorted.append(self.placing_fragment["id"])
        else:
            self.used_pybullet_frag_ids_sorted = []
            self.used_frag_ids_sorted = []

        self.current_fragment_to_table_height = 0.0
        self.drop_corner_distance = 0.0
        self.drop_ruler_distance = 0.0

        if last_fragment and self.mode == "eval" and self.place_complete_fresco:
            self.fresco_no += 1
        
        if self.place_complete_fresco == False or self.fresco_no <= self.used_frescoes[1]:
            self.placing_fragment = {}
            self.table_fragments = {}
            self.switch_state()
        
        # Contacts
        self.reset_contacts()

        self.current_steps = 0
        self.current_episode += 1


    def update_observations(self):
        observation = {}

        ### Gripper pose ###
        temp_pose = self.get_tool_tip_pose()
        # 3D relative gripper position [?,?]
        gripper_position = np.array(temp_pose[:3], dtype=np.float32)

        # 1D relative gripper orientation (Quaternion)
        gripper_yaw_angle = np.array([temp_pose[3]], dtype=np.float32)

        # 1D min placing fragment distance [1]
        if self.use_min_placing_object_dist_in_obs:
            if self.use_min_3d_observations:
                min_placing_object_dist = self.min_placing_object_3d_dist
            else:
                min_placing_object_dist = np.array([self.min_placing_object_euc_dist], dtype=np.float32)
        
        # 1D min robot to table objects distance [1]
        if self.use_min_robot_to_table_objects_dist_in_obs:
            if self.use_min_3d_observations:
                min_robot_to_table_objects_dist = self.min_robot_to_table_objects_3d_dist
            else:
                min_robot_to_table_objects_dist = np.array([self.min_robot_to_table_objects_euc_dist], dtype=np.float32)

        # 1D placing object overlap [1]
        if self.use_placing_object_overlap_in_obs:
            overlap_bool = self.evaluation_utils.get_placing_fragment_overlap(self.overlap_inflation, self.attach_footprint_to_overlap_in_obs)
            overlap = np.array([int(overlap_bool)], dtype=np.float32)

        # 3D corner distances e.g. [0,36]
        if self.placing_fragment["first_fragment"]  == False:
            corner_distances = self.flatten_and_pad_numpy_array(array=self.current_3d_corner_distances, size=self.max_no_of_corner_distances)
        else:
            corner_distances = np.zeros(self.max_no_of_corner_distances, dtype=np.float32)

        # 3D ruler distances e.g. [0,12]
        ruler_distances = np.zeros(self.max_no_of_ruler_distances, dtype=np.float32)
        if self.placing_fragment["ruler_fragment"] == True:
            ruler_distances_per_axis = int(self.max_no_of_ruler_distances/2)
            # x_ruler
            if len(self.current_x_ruler_corners) != 0:
                ruler_distances[:ruler_distances_per_axis] = self.current_3d_x_ruler_distances.flatten()
            # y_ruler
            if len(self.current_y_ruler_corners) != 0:
                ruler_distances[-ruler_distances_per_axis:] = self.current_3d_y_ruler_distances.flatten()

        #1D placing fragment angle
        fragment_angle = np.array([self.get_fragment_pose(self.placing_fragment["pybullet_id"])[3]], dtype=np.float32)

        if self.use_min_placing_object_dist_in_obs and self.use_placing_object_overlap_in_obs and self.use_min_robot_to_table_objects_dist_in_obs:
            if self.ablation == "no_ruler":
                vector_1d = np.concatenate([gripper_position, gripper_yaw_angle, min_placing_object_dist, min_robot_to_table_objects_dist, overlap, corner_distances, fragment_angle], axis=-1)
            else:
                vector_1d = np.concatenate([gripper_position, gripper_yaw_angle, min_placing_object_dist, min_robot_to_table_objects_dist, overlap, corner_distances, ruler_distances, fragment_angle], axis=-1)
        elif self.use_min_placing_object_dist_in_obs and self.use_placing_object_overlap_in_obs:
            if self.ablation == "no_ruler":
                vector_1d = np.concatenate([gripper_position, gripper_yaw_angle, min_placing_object_dist, overlap, corner_distances, fragment_angle], axis=-1)
            else:
                vector_1d = np.concatenate([gripper_position, gripper_yaw_angle, min_placing_object_dist, overlap, corner_distances, ruler_distances, fragment_angle], axis=-1)
        elif self.use_min_placing_object_dist_in_obs:
            if self.ablation == "no_ruler":
                vector_1d = np.concatenate([gripper_position, gripper_yaw_angle, min_placing_object_dist, corner_distances, fragment_angle], axis=-1)
            else:
                vector_1d = np.concatenate([gripper_position, gripper_yaw_angle, min_placing_object_dist, corner_distances, ruler_distances, fragment_angle], axis=-1)
        else:
            if self.ablation == "no_ruler":
                vector_1d = np.concatenate([gripper_position, gripper_yaw_angle, corner_distances, fragment_angle], axis=-1)
            else:
                vector_1d = np.concatenate([gripper_position, gripper_yaw_angle, corner_distances, ruler_distances, fragment_angle], axis=-1)

        observation["vector_1d"] = vector_1d

        if self.config["normalize_obs"]:
            observation = self.rl_utils.normalize_observations(self.observation_space, observation)

        return observation

    def _termination(self):       
        if self.state >= 1 and self.fragment_contacts == 0:
            success = True
        else:
            success = False

        if self.current_steps > int(self.config["timeout_iterations"]):
            self.timeout = True
        else:
            self.timeout = False

        if (success == True
            or self.timeout
            or (self.plane_contact and self.terminate_on_plane_contact and self.mode != "eval")
            or (self.frag_contact and self.terminate_on_fragment_contact and self.mode != "eval")
            or (self.min_placing_object_euc_dist <= self.too_close_threshold and self.terminate_on_too_close and self.mode != "eval")
            or (self.min_robot_to_table_objects_euc_dist <= self.too_close_threshold and self.terminate_on_too_close and self.mode != "eval")
            or self.state >= 1):
            return True, success
        else:
            return False, success

    def _apply_action(self, action):
        # collision_check_reference is for frag2frag collision
        self.update_contacts(*self.mu.move_link_to_xyz_yaw_position(action, self.tool_tip_id, self.debug, collision_check="count", frag_ids=self.used_pybullet_frag_ids_sorted, collision_check_reference=self.placing_fragment["pybullet_id"]))

        if action[4] > 0.0 and self.state == 0:
            self.current_fragment_to_table_height = self.get_fragment_pose(self.placing_fragment["pybullet_id"])[2]
            if (self.drop_height_threshold == -1.0
                or self.current_fragment_to_table_height <= self.drop_height_threshold + (self.gt_data["header"]["height"]/2)):
                if self.placing_fragment["first_fragment"]  == False:
                    self.drop_corner_distance = np.mean(self.current_1d_corner_distances)
                    self.tensorboard_drop_corner_distance = self.drop_corner_distance
                else:
                    self.drop_corner_distance = 0.0
                    self.tensorboard_drop_corner_distance = 0.0             

                if self.placing_fragment["ruler_fragment"] == True:
                    self.drop_ruler_distance = np.mean(self.current_1d_ruler_distances)
                    self.tensorboard_drop_ruler_distance = self.drop_ruler_distance
                else:
                    self.drop_ruler_distance = 0.0
                    self.tensorboard_drop_ruler_distance = 0.0

                drop_target_angle = self.current_placing_fragment_angle
                self.tensorboard_drop_target_angle = np.rad2deg(drop_target_angle)

                self.state += 1 # transit from state 0 to state 1
                self.switch_state()

    def get_observations_from_simulation(self):
        # Get location of current corresponding corners
        self.current_corners, self.current_ruler_corners, self.current_x_ruler_corners, self.current_y_ruler_corners = self.get_current_corresponding_corner_transforms(self.placing_fragment, self.table_fragments, self.no_fragments, self.ruler_lines)
        self.current_3d_corner_distances, self.current_1d_corner_distances, self.current_corresponding_corner_distance = self.calculate_corner_distances(current_corners=self.current_corners, corner_distance_power=self.corner_distance_power, corner_distance_root=self.corner_distance_root)
        self.current_3d_ruler_distances, self.current_1d_ruler_distances, self.current_ruler_distance = self.calculate_corner_distances(current_corners=self.current_ruler_corners, corner_distance_power=self.ruler_distance_power, corner_distance_root=self.ruler_distance_root)
        self.current_3d_x_ruler_distances, _, _ = self.calculate_corner_distances(current_corners=self.current_x_ruler_corners, corner_distance_power=self.ruler_distance_power, corner_distance_root=self.ruler_distance_root)
        self.current_3d_y_ruler_distances, _, _ = self.calculate_corner_distances(current_corners=self.current_y_ruler_corners, corner_distance_power=self.ruler_distance_power, corner_distance_root=self.ruler_distance_root)
        
        if self.debug and self.state == 0:
            if self.placing_fragment["first_fragment"]  == False or self.ablation == "no_ruler":
                self.corresponding_corners_debug_lines = debug_utils.DebugLinesCorrespondingCorners(config=self.config, sim=self, p_id=self._p, current_corners=self.current_corners, line_length = 0.05, line_width = 10.0, line_color=self.corner_color)
            if self.placing_fragment["ruler_fragment"] == True and self.ablation != "no_ruler":
                self.ruler_corners_debug_lines = debug_utils.DebugLinesCorrespondingCorners(config=self.config, sim=self, p_id=self._p, current_corners=self.current_ruler_corners, line_length = 0.05, line_width = 10.0, line_color=self.ruler_corner_color)
            
        self.current_placing_fragment_angle = self.get_fragment_pose(self.placing_fragment["pybullet_id"])[3]
    
        if self.use_min_placing_object_dist_in_obs:
            if self.placing_fragment["first_fragment"]  == False:
                #self.min_placing_object_euc_dist = self.collision_utils.get_min_placing_object_dist()
                self.min_placing_object_euc_dist , self.min_placing_object_3d_dist = self.collision_utils.get_3d_min_placing_object_dist()
            else:
                self.min_placing_object_euc_dist = self.fragment_to_target_distance_limits[1] #np.finfo(np.float32).max
                self.min_placing_object_3d_dist = np.array(3*[self.fragment_to_target_3d_distance_limits[1]], dtype=np.float32)

        if self.use_min_robot_to_table_objects_dist_in_obs:
            if self.placing_fragment["first_fragment"]  == False:
                #self.min_robot_to_table_objects_euc_dist = self.collision_utils.get_min_robot_to_table_frag_dist()
                self.min_robot_to_table_objects_euc_dist, self.min_robot_to_table_objects_3d_dist = self.collision_utils.get_3d_min_robot_to_table_frag_dist()
            else:
                self.min_robot_to_table_objects_euc_dist = self.fragment_to_target_distance_limits[1] #np.finfo(np.float32).max
                self.min_robot_to_table_objects_3d_dist = np.array(3*[self.fragment_to_target_3d_distance_limits[1]], dtype=np.float32)


    def step(self, action):
        if self.debug and self.real_time == False and self.render == True:
            try:
                del self.reference_debug_cross
                if self.state == 0:
                        del self.corresponding_corners_debug_lines
                        del self.ruler_corners_debug_lines
            except:
                pass
        
        self.frag_contact = False
        self.plane_contact = False

        self._apply_action(action)
        
        self.current_steps += 1
        self.current_iteration += 1

        if self.debug and self.real_time == False and self.render == True:
            if self.state == 0:
                self.reference_marker_pose = self.get_fragment_pose(self.placing_fragment["pybullet_id"])
            elif self.state == 1:
                self.reference_marker_pose = self.get_tool_tip_pose()
            self.reference_debug_cross = debug_utils.DebugCross(config=self.config, sim=self, p_id=self._p, position=self.reference_marker_pose, line_length = 0.05, line_width = 10.0, line_color=[0,1,0])

        self.get_observations_from_simulation()
        termination_info, success = self._termination()

        current_reward = self.reward_utils.compute_reward()
        if self.debug:
            print("reward =", current_reward)

        current_observation = self.update_observations()

        if termination_info:
            self.tensorboard_success = success
            self.tensorboard_timeout = self.timeout
            if self.placing_fragment["first_fragment"]  == False:
                self.tensorboard_corresponding_corner_distance = self.current_corresponding_corner_distance
            else:
                self.tensorboard_corresponding_corner_distance = 0.0
            if self.placing_fragment["ruler_fragment"] == True:
                self.tensorboard_ruler_distance = self.current_ruler_distance
            else:
                self.tensorboard_ruler_distance = 0.0

            # Contacts bools
            self.tensorboard_frag_contact = self.frag_contact
            self.tensorboard_plane_contact = self.plane_contact
            # Contact counts
            self.tensorboard_fragment2robot_contacts_count = self.robot_contacts
            self.tensorboard_fragment2fragment_contacts_count = self.fragment_contacts
            self.tensorboard_plane_contacts_count = self.plane_contacts

            if self.mode == "eval":
                self.update_contact_history()

        self.termination_info = termination_info
        self.success = success
        info = {'is_success': self.success}

        return current_observation, current_reward, self.termination_info, info
    
def load_default_task():
    env = make_vec_env('GraspeAndPlaceEnvGUI-v0', n_envs=1)
    return env.envs[0].env.env

def test_env(environment):
    while True:
        environment.reset()
        for i in range(100):
            action = environment.action_space.sample()
            _, r, _, _ = environment.step(action)
            environment.step_simulation(environment.per_step_iterations)

if __name__ == '__main__':
    environment = load_default_task()
    test_env(environment)