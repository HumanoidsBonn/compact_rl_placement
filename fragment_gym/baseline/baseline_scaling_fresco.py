#!/usr/bin/python3

from fragment_gym.env.main_env import MainEnv
import gymnasium as gym
import numpy as np

class BaselineScalingFrescoEnv(MainEnv):
    def __init__(self, mode=None, ablation=None, config=None, real_time=None, debug=None, render=False, shared_memory=False, use_egl=False, fresco_range=[]):
        super().__init__(mode=mode, ablation=ablation, config=config, real_time=real_time, debug=debug, render=render, shared_memory=shared_memory, use_egl=use_egl, fresco_range=fresco_range)

        # set observation space and action space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.start_fragment = 0
        

    def task_main(self, baseline_name, fresco_no, assembly_plan_type, visualize, save_plot): 
        # Ground truth data
        self.fresco_no = fresco_no
        self.reset_baseline(self.fresco_no)
        self.gt_data = self.shapely_utils.get_fresco_ground_truth(fresco_no=self.fresco_no, root_path=self.root_path)
        no_fragments = self.gt_data["header"]["no_fragments"]

        # Scale factor parameters
        start_scale_factor = float(self.config["start_scale_factor"])
        end_scale_factor = float(self.config["end_scale_factor"])
        scale_increment = float(self.config["scale_increment"])
        round_scale_factor = 2
        scale_factor = start_scale_factor
        scale_factor = round(scale_factor,round_scale_factor)
        max_loop_counter = int(self.config["max_loop_counter"])

        # Manipulation parameters
        safety_release_height = float(self.config["safety_release_height"])
        realistic_pick_and_place = self.config["realistic_pick_and_place"]

        # Fresco information
        fresco_center = self.config["fresco_assembly_center_location"]
        fresco_polygons = self.shapely_utils.convert_fresco_array_to_shapely_multi_polygon(self.gt_data["fresco"])
        
        infinite_loop_counter = 0
        # Scaling fresco loop
        while True:
            infinite_loop_counter += 1

            # Scale up fresco
            print("\nScale fresco with a factor =", scale_factor)
            fresco_polygons_scaled, fresco_centroids_scaled = self.scaling_utils.scale_fresco(fresco_polygons=fresco_polygons, scale_factor=scale_factor)

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
                self.mu.move_baseline_to_xyz_yaw_position(position=waypoint, yaw=self.placing_fragment["yaw"], link_id=self.tool_tip_id)

                # Pick fragment
                if self.debug:
                    print("Pick fragment")
                picked_fragment_constraint = self.move_fragment_and_pick(fragment_pybullet_id=self.placing_fragment["pybullet_id"], fragment_id=self.placing_fragment["id"], grasp_yaw=self.placing_fragment["yaw"], fragment_spawn_yaw=0.0, pick_from_plane=realistic_pick_and_place)
                centroid_goal = list(fresco_centroids_scaled.geoms)[self.placing_fragment["id"]]
                                
                # Correct yaw assuming that the fragment pose is known
                frag_yaw = self.get_fragment_pose(self.placing_fragment["pybullet_id"])[3]
                tcp_yaw = self.get_current_tcp_yaw_in_euler()
                tcp_place_yaw = tcp_yaw - frag_yaw

                # Move ee to 5cm above target
                if self.debug:
                    print("Moving to above target")
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
                        collision_check_reference=self.placing_fragment["pybullet_id"] # for frag2frag collision
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

            # Save evaluation json and debug image
            if infinite_loop_counter == 1:
                first_save_cycle=True
            else:
                first_save_cycle=False
            self.evaluation_utils.evaluate_baseline_scaling_fresco(scale_factor, used_metrics=["all"], assembly_plan_type=assembly_plan_type, first_save_cycle=first_save_cycle)

            if scale_factor == end_scale_factor or (end_scale_factor == -1.0 and max(self.frag_contact_history) == False):
                self.reset_baseline(self.fresco_no)
                break
            self.reset_baseline(self.fresco_no)

            scale_factor += scale_increment
            scale_factor = round(scale_factor,round_scale_factor)

            if infinite_loop_counter > max_loop_counter:
                break

        return True