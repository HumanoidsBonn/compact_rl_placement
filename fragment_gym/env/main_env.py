#!/usr/bin/python3

from fragment_gym.env.robotiq_env import RobotEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from itertools import combinations

class MainEnv(RobotEnv):
    def __init__(self, mode=None, ablation=None, config=None, real_time=None, debug=None, render=False, shared_memory=False, use_egl=False, fresco_range=[]):
        super().__init__(mode=mode, ablation=ablation, config=config, real_time=real_time, debug=debug, render=render, shared_memory=shared_memory, use_egl=use_egl, fresco_range=fresco_range)

        self.fragment_format = self.config["fragment_format"]

        if self.fragment_format == "urdf":
            pass

        elif self.fragment_format == "stl":
            visual_type = "stl"
            collision_type = "stl"
        
        elif self.fragment_format == "vhacd":
            visual_type = "stl"
            collision_type = "vhacd"

        object_parking_position_start = list(self.config["object_parking_position_start"])
        object_parking_distance = float(self.config["object_parking_distance"])
        
        # create a small plane to place the fragment
        self.pick_plane_parking_center = np.asarray(object_parking_position_start)
        self.pick_plane_parking_center[0] = self.pick_plane_parking_center[0] + object_parking_distance
        self.mu.spawn_pick_plane(self.pick_plane_parking_center)

        if len(fresco_range) == 0:
            if self.mode == "train":
                self.used_frescoes = list(self.config["train_frescoes"])
            elif self.mode == "test":
                self.used_frescoes = list(self.config["test_frescoes"])
            elif self.mode == "eval":
                self.used_frescoes = list(self.config["eval_frescoes"])
            else:
                self.used_frescoes = [0,0]
                print("Warning: Only fresco_0 is loaded!!!")
        else:
            self.used_frescoes = fresco_range

        fresco_parking_center = np.asarray(object_parking_position_start)
        self.frag_ids_sorted = {}
        self.fresco_parking_centers = {}
        for fresco_no in range(self.used_frescoes[0], self.used_frescoes[1]+1):
            if fresco_no > 0:
                fresco_parking_center[0] = fresco_parking_center[0] - object_parking_distance
            if (fresco_no % 10) == 0 and fresco_no > 0:
                fresco_parking_center[0] = object_parking_position_start[0]
                fresco_parking_center[1] = fresco_parking_center[1] + object_parking_distance
            self.fresco_parking_center = fresco_parking_center.tolist()
            
            self.fresco_parking_centers[fresco_no] = self.fresco_parking_center
            self.frag_ids_sorted[fresco_no] = self.spawn_fresco_on_table(fresco_no=fresco_no, fresco_center_location=self.fresco_parking_center, visual_type=visual_type, collision_type=collision_type, scale_factor=float(self.config["fresco_scale_factor"]))

        self.world_id = self._p.saveState(physicsClientId=self.client_id)

        self.global_steps = 0

        self.placing_fragment = {}
        self.table_fragments = {}
        self.reset_contacts()
        self.reset_contact_history() # for evaluation

    def initial_reset(self):
        self._p.restoreState(self.world_id)
        self.man.reset_robot(self.initial_parameters)
        return
    
    def create_placing_fragment_dict(self, fresco_no, assembly_id, gt_data):
        no_fragments = gt_data["header"]["no_fragments"]
        neighbours = gt_data["neighbours"]
        self.placing_fragment = {}

        self.placing_fragment["pybullet_id"] = self.frag_ids_sorted[fresco_no][assembly_id][0]
        self.placing_fragment["id"] = self.frag_ids_sorted[fresco_no][assembly_id][1]
        self.placing_fragment["assembly_id"] = self.frag_ids_sorted[fresco_no][assembly_id][2]
        self.placing_fragment["centroid"] = self.frag_ids_sorted[fresco_no][assembly_id][3]
        self.placing_fragment["yaw"] = self.frag_ids_sorted[fresco_no][assembly_id][4]
        self.placing_fragment["corresponding_corners"] = neighbours[str(self.placing_fragment["id"])]

        if (str(no_fragments) in self.placing_fragment["corresponding_corners"] 
            or str(no_fragments+1) in self.placing_fragment["corresponding_corners"]):
            self.placing_fragment["ruler_fragment"] = True
        else:
            self.placing_fragment["ruler_fragment"] = False

        if assembly_id != self.placing_fragment["assembly_id"]:
            print("Placing fragment data is invalid!!!")
            raise ValueError

    def add_table_fragment_to_dict(self, placing_fragment):
        self.table_fragments[placing_fragment["id"]] = placing_fragment

    def get_pybullet_id_list_from_table_fragment_dict(self, table_fragments):
        pybullet_id_list = []
        
        for frag_id in table_fragments.keys():
            pybullet_id_list.append(table_fragments[frag_id]["pybullet_id"])

        return pybullet_id_list

    def create_table_fragment_dict(self, fresco_no, assembly_id, gt_data):
        no_fragments = gt_data["header"]["no_fragments"]
        neighbours = gt_data["neighbours"]
        self.table_fragments = {}

        for i in range(assembly_id):
            temp_id = self.frag_ids_sorted[fresco_no][i][1]
            self.table_fragments[temp_id] = {}
            self.table_fragments[temp_id]["pybullet_id"] = self.frag_ids_sorted[fresco_no][i][0]
            self.table_fragments[temp_id]["id"] = temp_id
            self.table_fragments[temp_id]["assembly_id"] = self.frag_ids_sorted[fresco_no][i][2]
            self.table_fragments[temp_id]["centroid"] = self.frag_ids_sorted[fresco_no][i][3]
            self.table_fragments[temp_id]["yaw"] = self.frag_ids_sorted[fresco_no][i][4]
            self.table_fragments[temp_id]["corresponding_corners"] = neighbours[str(temp_id)]

        if len(self.table_fragments) == 0:
            self.placing_fragment["first_fragment"] = True
        else:
            self.placing_fragment["first_fragment"] = False

    def reset_baseline(self, fresco_no):
        self.reset_robot(self.initial_parameters)
        self.reset_fresco(self.frag_ids_sorted[fresco_no], self.fresco_parking_centers[fresco_no])
        self.reset_contacts()
        self.reset_contact_history()
        self.placing_fragment = {}
        self.table_fragments = {}

    def reset_contacts(self):
        self.frag_contact = False
        self.frag_contact_episode = False
        self.plane_contact = False
        self.plane_contact_episode = False
        self.robot_contacts = 0
        self.fragment_contacts = 0
        self.plane_contacts = 0
    
    def reset_contact_history(self):
        self.plane_contact_history = []
        self.frag_contact_history = []
        self.fragment2robot_contacts_count_history = []
        self.fragment2fragment_contacts_count_history = []
        self.plane_contacts_count_history = []
        # self.collision_pair_list = []

    def update_contacts(self, robot_contacts, fragment_contacts, plane_contacts):
        self.robot_contacts += robot_contacts
        self.fragment_contacts += fragment_contacts
        self.plane_contacts += plane_contacts
        if robot_contacts > 0 or fragment_contacts > 0:
            self.frag_contact = True
            self.frag_contact_episode = True
        if plane_contacts > 0:
            self.plane_contact = True
            self.plane_contact_episode = True

    def update_contact_history(self):
        self.frag_contact_history.append(int(self.frag_contact_episode))
        self.plane_contact_history.append(int(self.plane_contact_episode))

        # Collision counts
        self.fragment2robot_contacts_count_history.append(int(self.robot_contacts))
        self.fragment2fragment_contacts_count_history.append(int(self.fragment_contacts))
        self.plane_contacts_count_history.append(int(self.plane_contacts))

    def find_two_most_distant_points(self, points):
        max_distance, selected_points = max(
            ((np.linalg.norm(np.array(p1) - np.array(p2)), (p1, p2))
            for p1, p2 in combinations(points, 2)),
            key=lambda x: x[0]
        )
        return list(selected_points)

    def get_corner_transform(self, corner_coordinates, fragment_pybullet_id):
        fragment_corner_pos, fragment_corner_orn = self._p.multiplyTransforms(
            self.get_7d_fragment_pose(fragment_pybullet_id)[0],
            self.get_7d_fragment_pose(fragment_pybullet_id)[1],
            [corner_coordinates[0], corner_coordinates[1], 0.0],
            self._p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        )
        return fragment_corner_pos

    def get_current_corresponding_corner_transforms(self, placing_fragment, table_fragments, no_fragments, ruler_lines):
        # Get ruler transforms
        current_ruler_corners = {}
        current_x_ruler_corners = {}
        current_y_ruler_corners = {}
        if self.placing_fragment["ruler_fragment"] == True:
            current_placing_fragment_ruler_corner_positions = []
            current_ruler_corner_positions = []
            current_placing_fragment_x_ruler_corner_positions = []
            current_x_ruler_corner_positions = []
            current_placing_fragment_y_ruler_corner_positions = []
            current_y_ruler_corner_positions = []
            # Ruler x-axis
            if str(no_fragments) in placing_fragment["corresponding_corners"]:
                # Get transform for corners on fragment
                temp_frag_ruler_corners_with_id_sorted = {}
                temp_frag_ruler_corners_with_id_sorted = dict(sorted(placing_fragment["corresponding_corners"][str(no_fragments)]["fragment_coordinates"].items()))
                temp_current_placing_fragment_ruler_corner_positions = []
                for corner_coordinates in temp_frag_ruler_corners_with_id_sorted.values():
                    fragment_ruler_corner_pos = self.get_corner_transform(corner_coordinates, placing_fragment["pybullet_id"])
                    temp_current_placing_fragment_ruler_corner_positions.append(fragment_ruler_corner_pos)
                current_placing_fragment_x_ruler_corner_positions.extend(temp_current_placing_fragment_ruler_corner_positions)
                current_placing_fragment_ruler_corner_positions.extend(temp_current_placing_fragment_ruler_corner_positions)
                
                temp_current_ruler_corner_positions = []
                for corner_coordinates in temp_current_placing_fragment_ruler_corner_positions:
                    line_corner_pos = self.get_closest_point_on_line(corner_coordinates, ruler_lines[0])
                    temp_current_ruler_corner_positions.append(line_corner_pos)
                current_x_ruler_corner_positions.extend(temp_current_ruler_corner_positions)
                current_ruler_corner_positions.extend(temp_current_ruler_corner_positions)

                current_x_ruler_corners = {"fragment_corners":np.asarray(current_placing_fragment_x_ruler_corner_positions, dtype=np.float32), "neighbour_corners":np.asarray(current_x_ruler_corner_positions, dtype=np.float32)}
            
            # Ruler y-axis
            if str(no_fragments+1) in placing_fragment["corresponding_corners"]:
                # Get transform for corners on fragment
                temp_frag_ruler_corners_with_id_sorted = {}
                temp_frag_ruler_corners_with_id_sorted = dict(sorted(placing_fragment["corresponding_corners"][str(no_fragments+1)]["fragment_coordinates"].items()))
                temp_current_placing_fragment_ruler_corner_positions = []
                for corner_coordinates in temp_frag_ruler_corners_with_id_sorted.values():
                    fragment_ruler_corner_pos = self.get_corner_transform(corner_coordinates, placing_fragment["pybullet_id"])
                    temp_current_placing_fragment_ruler_corner_positions.append(fragment_ruler_corner_pos)
                current_placing_fragment_y_ruler_corner_positions.extend(temp_current_placing_fragment_ruler_corner_positions)
                current_placing_fragment_ruler_corner_positions.extend(temp_current_placing_fragment_ruler_corner_positions)

                temp_current_ruler_corner_positions = []
                for corner_coordinates in temp_current_placing_fragment_ruler_corner_positions:
                    line_corner_pos = self.get_closest_point_on_line(corner_coordinates, ruler_lines[1])
                    temp_current_ruler_corner_positions.append(line_corner_pos)
                current_y_ruler_corner_positions.extend(temp_current_ruler_corner_positions)
                current_ruler_corner_positions.extend(temp_current_ruler_corner_positions)

                current_y_ruler_corners = {"fragment_corners":np.asarray(current_placing_fragment_y_ruler_corner_positions, dtype=np.float32), "neighbour_corners":np.asarray(current_y_ruler_corner_positions, dtype=np.float32)}
                
            current_ruler_corners = {"fragment_corners":np.asarray(current_placing_fragment_ruler_corner_positions, dtype=np.float32), "neighbour_corners":np.asarray(current_ruler_corner_positions, dtype=np.float32)}

        # Get corner transforms
        current_corners = {}
        if placing_fragment["first_fragment"]  == False or self.ablation == "no_ruler":
            current_placing_fragment_corner_positions = []
            current_neighbour_fragment_corner_positions = []
            for neighbour_id in table_fragments:
                temp_current_placing_fragment_corner_positions = []
                temp_current_neighbour_fragment_corner_positions = []
                if str(neighbour_id) in placing_fragment["corresponding_corners"].keys():
                    temp_frag_corners_with_id = placing_fragment["corresponding_corners"][str(neighbour_id)]["fragment_coordinates"]
                    temp_frag_corners_with_id_sorted = dict(sorted(temp_frag_corners_with_id.items()))
                    for corner in temp_frag_corners_with_id_sorted.values():
                        fragment_corner_pos = self.get_corner_transform(corner, placing_fragment["pybullet_id"])
                        temp_current_placing_fragment_corner_positions.append(fragment_corner_pos)
                    
                    temp_neighbour_corners_with_id = table_fragments[neighbour_id]["corresponding_corners"][str(placing_fragment["id"])]["fragment_coordinates"]
                    temp_neighbour_corners_with_id_sorted = dict(sorted(temp_neighbour_corners_with_id.items()))
                    for corner in temp_neighbour_corners_with_id_sorted.values():
                        if self.ablation == "no_ruler":
                            # Treat special case of first placing fragment
                            if placing_fragment["id"] == table_fragments[neighbour_id]["id"]:
                                neighbour_corner_pos, neighbour_corner_orn = self._p.multiplyTransforms(
                                    table_fragments[neighbour_id]["centroid"],
                                    self._p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                                    [corner[0], corner[1], 0.0],
                                    self._p.getQuaternionFromEuler([0.0, 0.0, 0.0])
                                )
                            else:
                                neighbour_corner_pos = self.get_corner_transform(corner, table_fragments[neighbour_id]["pybullet_id"])
                        else:
                            neighbour_corner_pos = self.get_corner_transform(corner, table_fragments[neighbour_id]["pybullet_id"])
                        temp_current_neighbour_fragment_corner_positions.append(neighbour_corner_pos)
                    
                    # Keep only 2 corners which have the biggest distance between each other
                    temp_two_current_placing_fragment_corner_positions = self.find_two_most_distant_points(temp_current_placing_fragment_corner_positions)
                    temp_two_current_neighbour_fragment_corner_positions = self.find_two_most_distant_points(temp_current_neighbour_fragment_corner_positions)

                    # Check if corner is shared with ruler
                    if self.placing_fragment["ruler_fragment"] == True:
                        if (temp_two_current_placing_fragment_corner_positions[0] in current_placing_fragment_ruler_corner_positions 
                            or temp_two_current_placing_fragment_corner_positions[1] in current_placing_fragment_ruler_corner_positions):
                            current_placing_fragment_corner_positions.extend(temp_two_current_placing_fragment_corner_positions)
                            current_neighbour_fragment_corner_positions.extend(temp_two_current_neighbour_fragment_corner_positions)
                    else:
                        current_placing_fragment_corner_positions.extend(temp_two_current_placing_fragment_corner_positions)
                        current_neighbour_fragment_corner_positions.extend(temp_two_current_neighbour_fragment_corner_positions)
            if len(current_placing_fragment_corner_positions) > 0 and len(current_neighbour_fragment_corner_positions) > 0:
                current_corners = {"fragment_corners":np.asarray(current_placing_fragment_corner_positions, dtype=np.float32), "neighbour_corners":np.asarray(current_neighbour_fragment_corner_positions, dtype=np.float32)}

        return current_corners, current_ruler_corners, current_x_ruler_corners, current_y_ruler_corners#, amount_of_ruler_distances

    def calculate_corner_distances(self, current_corners, corner_distance_power=1, corner_distance_root=False):
        if len(current_corners) == 0 or len(current_corners["fragment_corners"]) == 0 or len(current_corners["neighbour_corners"]) == 0:
            current_3d_corner_distances = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            current_1d_corner_distances = np.array([0.0], dtype=np.float32)
            current_corresponding_corner_distance = 0.0

        else:
            # Calculate 3d distances of each corresponding corner pair
            current_3d_corner_distances = current_corners["neighbour_corners"] - current_corners["fragment_corners"]
            current_1d_corner_distances = np.linalg.norm(current_3d_corner_distances, axis=1)
            current_corresponding_corner_distance = np.mean(np.power(current_1d_corner_distances, corner_distance_power))
            if corner_distance_root:
                self.current_corresponding_corner_distance = np.sqrt(current_corresponding_corner_distance)
        
        return current_3d_corner_distances, current_1d_corner_distances, current_corresponding_corner_distance

    def flatten_and_pad_numpy_array(self, array, size):
        if array is None:
            temp_array = np.zeros(size, dtype=np.float32)
        
        else:
            array_flat = array.flatten()
            if len(array_flat) > size:
                print("Invalid amount of values in array.")
                raise ValueError
            zeros_to_add = size - len(array_flat)
            temp_array = np.pad(array_flat, (0, zeros_to_add), 'constant', constant_values=(0.0, 0.0))
        
        return temp_array

    def get_closest_point_on_line(self, point_tuple, line_tuples, ensure_solution_on_line=True):
        # Convert input tuples to numpy arrays for easier calculations
        point = np.array(point_tuple)
        line_start = np.array(line_tuples[0])
        line_end = np.array(line_tuples[1])

        # Calculate the direction vector of the line
        line_direction = line_end - line_start

        # Calculate the vector from the line's starting point to the given point
        point_vector = point - line_start

        # Calculate the projection of the point vector onto the line direction
        projection = np.dot(point_vector, line_direction) / np.dot(line_direction, line_direction)

        # Calculate the closest point on the line to the given point
        closest_point = line_start + projection * line_direction

        if ensure_solution_on_line:
            # Ensure the closest point lies within the line segment
            t = np.clip(projection, 0, 1)
            closest_point = line_start + t * line_direction

        return tuple(closest_point)


def load_default_task():
    env = make_vec_env('MainEnv-v0', n_envs=1)
    return env.envs[0].env.env

def test_env(environment):
    environment.reset()
    while True:
        action = environment.action_space.sample()
        _, r, _, _ = environment.step(action)
        environment.step_simulation(environment.per_step_iterations)


if __name__ == '__main__':
    environment = load_default_task()
    test_env(environment)