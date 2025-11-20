#!/usr/bin/python3

from fragment_gym.env.main_env import MainEnv
import gymnasium as gym
import numpy as np

from utils import relative_placing_utils as relative_placing_utils
from shapely import Point, Polygon, MultiPolygon
from shapely.affinity import translate

class BaselineRelativePlacingEnv(MainEnv):
    def __init__(self, mode=None, ablation=None, config=None, real_time=None, debug=None, render=False, shared_memory=False, use_egl=False, fresco_range=[]):
        super().__init__(mode=mode, ablation=ablation, config=config, real_time=real_time, debug=debug, render=render, shared_memory=shared_memory, use_egl=use_egl, fresco_range=fresco_range)

        # set observation space and action space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        # Initialize additional utils
        self.relative_placing_utils = relative_placing_utils.RelativePlacingUtils(config=self.config, sim=self, p_id=self._p, frescoes_path=self.frescoes_path)

        self.start_fragment = 0

    def task_main(self, baseline_name, fresco_no, assembly_plan_type, visualize, save_plot):
        # Ground truth data
        self.fresco_no = fresco_no
        self.reset_baseline(self.fresco_no)
        self.gt_data = self.shapely_utils.get_fresco_ground_truth(fresco_no=fresco_no, root_path=self.root_path)
        frescos = self.gt_data["fresco"]
        centroids = self.gt_data["centroids"]
        neighbours = self.gt_data["neighbours"]
        no_fragments = self.gt_data["header"]["no_fragments"]

        # Relative placing parameters
        initial_placement_strategy = self.config["initial_placement_strategy"]
        fragment_inflation = self.config["fragment_inflation"]
        gripper_length = self.config["gripper_length"]
        gripper_width = self.config["gripper_width"]
        gripper_release_distance = self.config["gripper_release_distance"]
        fragment_step_increment = self.config["fragment_step_increment"]
        first_fresco_index = 0

        # Manipulation parameters
        safety_release_height = float(self.config["safety_release_height"])
        realistic_pick_and_place = self.config["realistic_pick_and_place"]
        table_fragments_pybullet_ids = []

        # Fresco information
        fresco_center = self.config["fresco_assembly_center_location"]
        fresco_polygons = self.shapely_utils.convert_fresco_array_to_shapely_multi_polygon(self.gt_data["fresco"])
        fresco_centroids = self.shapely_utils.convert_point_array_to_shapely_multi_point(self.gt_data["centroids"])
        table_fragments_pybullet_ids = []
        table_fragments_list = []
        table_fragments_id_list = []
        neighbours_list = []

        # Placing 1 fresco loop
        for frag_iter in range(self.start_fragment, no_fragments):

            # Create dictionary to store all information of current placing fragment
            self.create_placing_fragment_dict(self.fresco_no, frag_iter, self.gt_data)

            if self.debug:
                print(f'----Iteration {frag_iter}: Placing fragment {self.placing_fragment["id"]}----')
            else:
                print(f'{frag_iter}/{no_fragments-1}: Placing fragment with id {self.placing_fragment["id"]}')
        
            # Move ee to 5cm above fresco center
            if self.debug:
                print("Moving to above target")
            waypoint = [fresco_center[0], fresco_center[1], 0.05]

            # Pick fragment
            if self.debug:
                print("Pick fragment")
            picked_fragment_constraint = self.move_fragment_and_pick(fragment_pybullet_id=self.placing_fragment["pybullet_id"], fragment_id=self.placing_fragment["id"], grasp_yaw=self.placing_fragment["yaw"], fragment_spawn_yaw=0.0, pick_from_plane=realistic_pick_and_place)
            centroid_goal = Polygon(frescos[self.placing_fragment["id"]]).centroid
                            
            # Correct yaw assuming that the fragment pose is known
            frag_yaw = self.get_fragment_pose(self.placing_fragment["pybullet_id"])[3]
            tcp_yaw = self.get_current_tcp_yaw_in_euler()
            tcp_place_yaw = tcp_yaw - frag_yaw

            if frag_iter > first_fresco_index:
                if initial_placement_strategy == "relative_centroid":
                    fragment_shifted = translate(Polygon(frescos[self.placing_fragment["id"]]), translation_vector.x, translation_vector.y)
                    fragment_centroid = fragment_shifted.centroid
                    fragment_with_gripper  = self.relative_placing_utils.attach_gripper_footprint(fragment_shifted,tcp_place_yaw, visualize=visualize, 
                                                                                                  gripper_offset = gripper_length, width = gripper_width,
                                                                                                  safe_offset = gripper_release_distance, inflation= fragment_inflation)
                elif initial_placement_strategy == "nearest_neighbour":
                    current_fragment = Polygon(frescos[self.placing_fragment["id"]])
                    current_fragment_centroid = current_fragment.centroid 

                    nearest_fragments = list(neighbours[str(self.placing_fragment["id"])].keys()) # get neighbours (str)
                    nearest_fragments[:] = [x for x in nearest_fragments if int(x) < no_fragments]
                    neighbours_list.append(nearest_fragments)
                    nearest_fragments = [int(element) for element in nearest_fragments] # convert str to int                    
                    distances= []                
                    for fragment_index in nearest_fragments:
                        distances.append(current_fragment_centroid.distance( Polygon(frescos[fragment_index]).centroid))                    
                    sorted_data = sorted(zip(distances, nearest_fragments), key=lambda x: x[0]) # sort neighbours as per centroid distance 
                    distances, nearest_fragments = zip(*sorted_data)

                    for nearest_fragment in nearest_fragments:
                        if nearest_fragment in table_fragments_id_list:
                            nearest_fragment_placed = table_fragments.geoms[table_fragments_id_list.index(nearest_fragment)]
                            break
                        else:
                            if self.debug:
                                print(f"ID {nearest_fragment} is not present in placed_ids.")

                    if frag_iter > first_fresco_index:
                        translation_vector = Point(nearest_fragment_placed.centroid.x - current_fragment_centroid.x, nearest_fragment_placed.centroid.y - current_fragment_centroid.y)
                    else:
                        translation_vector = Point(centroid_goal.x - current_fragment_centroid.x, centroid_goal.y - current_fragment_centroid.y)
                    fragment_shifted = translate(current_fragment, translation_vector.x, translation_vector.y)
                    
                    fragment_in_layout = Polygon(frescos[nearest_fragment])
                    translation_vector = Point(current_fragment_centroid.x - fragment_in_layout.centroid.x, current_fragment_centroid.y - fragment_in_layout.centroid.y)
                    fragment_shifted = translate(fragment_shifted, translation_vector.x, translation_vector.y)
                    
                    fragment_centroid = fragment_shifted.centroid
                    fragment_with_gripper = self.relative_placing_utils.attach_gripper_footprint(fragment_shifted,tcp_place_yaw, visualize=visualize,
                                                                                                 gripper_offset = gripper_length, width = gripper_width,
                                                                                                 safe_offset = gripper_release_distance, inflation= fragment_inflation)
                else:
                    fragment_with_gripper = self.relative_placing_utils.attach_gripper_footprint(frescos[self.placing_fragment["id"]],tcp_place_yaw, visualize=visualize,
                                                                                                 gripper_offset = gripper_length, width = gripper_width,
                                                                                                 safe_offset = gripper_release_distance, inflation= fragment_inflation)
                best_place = self.relative_placing_utils.place_along_centroids(table_fragments,fragment_with_gripper, fragment_centroid, fragment_step_increment=fragment_step_increment, visualize=visualize)

            # Move ee to 5cm above target
            if self.debug:
                print("Moving to above target")
            if frag_iter > first_fresco_index:
                waypoint = [fresco_center[0]+best_place.x, fresco_center[1]+best_place.y, 0.05]
            else:
                waypoint = [fresco_center[0]+centroid_goal.x, fresco_center[1]+centroid_goal.y, 0.05]
            self.mu.move_baseline_to_xyz_yaw_position(position=waypoint, yaw=tcp_place_yaw, link_id=self.tool_tip_id)

            # Move to placement location
            if self.debug:
                print("Moving to target")
            waypoint[2] = self.gt_data["header"]["height"]/2 + safety_release_height
            self.update_contacts(
                *self.mu.move_baseline_to_xyz_yaw_position(
                    position=waypoint,
                    yaw=tcp_place_yaw, 
                    link_id=self.tool_tip_id,
                    collision_check="count",
                    frag_ids=self.get_pybullet_id_list_from_table_fragment_dict(self.table_fragments),
                    collision_check_reference=self.placing_fragment["pybullet_id"]
                )
            )

            # Open the gripper
            if realistic_pick_and_place:
                self._p.removeConstraint(picked_fragment_constraint)
                self.step_simulation(self.per_step_iterations)

            self.update_contacts(
                *self.control_gripper(
                    close=False,
                    release=True,
                    release_obj_id=self.placing_fragment["pybullet_id"],
                    coll_check="count",
                    coll_check_ids=self.get_pybullet_id_list_from_table_fragment_dict(self.table_fragments),
                    coll_check_ref_id=self.placing_fragment["pybullet_id"] # for frag2frag collision
                    )
                )

            if realistic_pick_and_place == False:
                self._p.removeConstraint(picked_fragment_constraint)
                self.step_simulation(self.per_step_iterations)
            
            self.add_table_fragment_to_dict(self.placing_fragment)
            retract_col_check_pybullet_ids = self.get_pybullet_id_list_from_table_fragment_dict(self.table_fragments)
            # Move to 5cm above target
            if self.debug:
                print("Moving above target")
            waypoint[2] = 0.05
            self.update_contacts(
                *self.mu.move_baseline_to_xyz_yaw_position(
                    position=waypoint,
                    yaw=tcp_place_yaw, 
                    link_id=self.tool_tip_id,
                    collision_check="count",
                    frag_ids=retract_col_check_pybullet_ids,
                    collision_check_reference=self.placing_fragment["pybullet_id"] # for frag2frag collision
                )
            )

            self.update_contact_history()
            self.reset_contacts()
            
            if frag_iter > first_fresco_index:
                current_fragment = Polygon(frescos[self.placing_fragment["id"]])
                current_fragment_centroid = current_fragment.centroid                
                translation_vector = Point(best_place.x - current_fragment_centroid.x, best_place.y - current_fragment_centroid.y)
                current_fragment_shifted = translate(current_fragment, translation_vector.x, translation_vector.y)
                table_fragments_id_list.append(self.placing_fragment["id"])
                table_fragments_list.append(current_fragment_shifted)
                table_fragments = MultiPolygon(table_fragments_list) 
            else:
                current_fragment = Polygon(frescos[self.placing_fragment["id"]])
                translation_vector = Point(centroid_goal.x - current_fragment.centroid.x, centroid_goal.y - current_fragment.centroid.y)               
                table_fragments_id_list.append(self.placing_fragment["id"])
                table_fragments_list.append(current_fragment)
                table_fragments = MultiPolygon(table_fragments_list)

        # Save evaluation json and debug image
        self.evaluation_utils.evaluate_baseline_relative_placing(assembly_plan_type=assembly_plan_type)

        self.reset_baseline(self.fresco_no)

        return True