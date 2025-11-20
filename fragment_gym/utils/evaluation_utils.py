#!/usr/bin/python3

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import translate, rotate
from shapely import overlaps
from collections import OrderedDict
import copy

class EvaluationUtils():
    def __init__(self, config, sim, p_id):
        self.config = config
        self.sim = sim
        self._p = p_id

    def transform_world_to_fresco_coords(self, fresco_polygons_in_world_coords):
        # World to fresco coords
        fresco_polygons_in_world_coords = MultiPolygon(fresco_polygons_in_world_coords)
        move_vector = fresco_polygons_in_world_coords.centroid
        polygon_world_coords = self.sim.shapely_utils.convert_shapely_multi_polygon_to_array_of_shapely_polygons(fresco_polygons_in_world_coords)
        fresco_polygon_array_in_fresco_coords = []
        for polygon in polygon_world_coords:
            translated_polygon = translate(Polygon(polygon), xoff=-move_vector.x, yoff=-move_vector.y)
            fresco_polygon_array_in_fresco_coords.append(translated_polygon)
        fresco_polygons_in_fresco_coords = MultiPolygon(fresco_polygon_array_in_fresco_coords)
        return fresco_polygons_in_fresco_coords
    
    def get_fragment_area(self,multipolygon):
        multipolygon_array = self.sim.shapely_utils.convert_shapely_multi_polygon_to_array_of_shapely_polygons(multipolygon)
        area = 0
        for polygon_array in multipolygon_array:
            area += Polygon(polygon_array).area
        return area

    def get_current_fresco_polygons_via_sim(self, table_fragments, gt_fresco_polygons_in_fresco_coords):
        table_fragments_sorted = OrderedDict(sorted(table_fragments.items())) # Sort table fragments by id
        table_fragments_pybullet_ids = [x['pybullet_id'] for x in table_fragments_sorted.values()] # Get a list of Pybullet IDs
        table_fragments_ids = [x['id'] for x in table_fragments_sorted.values()] # Get a list of IDs
        
        fragment_4D_poses_in_world_coords = self.sim.get_all_fragment_poses(table_fragments_pybullet_ids)

        fresco_polygons = self.sim.shapely_utils.convert_shapely_multi_polygon_to_array_of_shapely_polygons(gt_fresco_polygons_in_fresco_coords)
        fresco_polygons_filtered = [fresco_polygons[i] for i in range(len(fresco_polygons)) if i in table_fragments_ids]
        polygon_fresco_to_world = [] 
        for current_polygon, pose in zip(fresco_polygons_filtered, fragment_4D_poses_in_world_coords):
            current_polygon_frag_coords = translate(Polygon(current_polygon), xoff=-current_polygon.centroid.x, yoff=-current_polygon.centroid.y)
            # Translate the polygon to the new position
            translated_polygon = translate(current_polygon_frag_coords, xoff=pose[0], yoff=pose[1])
            # Apply rotation to the vertices
            rotated_polygon = rotate(translated_polygon, angle=np.rad2deg(pose[3]), origin=translated_polygon.centroid)
            polygon_fresco_to_world.append(rotated_polygon)
        
        current_fresco_polygons_in_world_coords = MultiPolygon(polygon_fresco_to_world)

        return current_fresco_polygons_in_world_coords
    
    def get_current_placing_fragment_polygon_pose_via_sim(self, placing_fragment, gt_fresco_polygons_in_fresco_coords):      
        placing_fragment_4D_pose_in_world_coords = self.sim.get_fragment_pose(placing_fragment["pybullet_id"])
        fresco_polygons = self.sim.shapely_utils.convert_shapely_multi_polygon_to_array_of_shapely_polygons(gt_fresco_polygons_in_fresco_coords)
        placing_fragment_polygon = fresco_polygons[placing_fragment["id"]]
        
        placing_fragment_polygon_frag_coords = translate(Polygon(placing_fragment_polygon), xoff=-placing_fragment_polygon.centroid.x, yoff=-placing_fragment_polygon.centroid.y)
        # Translate the polygon to the new position
        translated_placing_fragment_polygon = translate(placing_fragment_polygon_frag_coords, xoff=placing_fragment_4D_pose_in_world_coords[0], yoff=placing_fragment_4D_pose_in_world_coords[1])
        # Apply rotation to the vertices
        placing_fragment_polygon_world_coords = rotate(translated_placing_fragment_polygon, angle=np.rad2deg(placing_fragment_4D_pose_in_world_coords[3]), origin=translated_placing_fragment_polygon.centroid)
        
        current_placing_fragment_polygons_in_world_coords = MultiPolygon([Polygon(placing_fragment_polygon_world_coords)])

        return current_placing_fragment_polygons_in_world_coords

    def get_distances_between_fragments(self, fragments):
        min_distances = []
        for ref_frag_no in range(len(fragments.geoms)):
            min_dist = float("inf")
            for comp_frag_no in range(len(fragments.geoms)):
                if ref_frag_no != comp_frag_no:
                    min_dist = min(
                        min_dist,
                        fragments.geoms[ref_frag_no].distance(fragments.geoms[comp_frag_no])
                    )
            min_distances.append(min_dist)

        return min_distances
    
    def get_gap_metrics(self, final_fresco_polygons_in_fresco_coords, fresco_dimensions):

        final_fresco_bounding_box = final_fresco_polygons_in_fresco_coords.minimum_rotated_rectangle

        # Base values for metrics
        distances_between_fragments = self.get_distances_between_fragments(final_fresco_polygons_in_fresco_coords)
        area_final_fresco_bbox = final_fresco_bounding_box.area
        area_fragments_of_final_fresco = self.get_fragment_area(final_fresco_polygons_in_fresco_coords)
        area_gt_fresco = fresco_dimensions["length"] * fresco_dimensions["width"]
        area_gap_of_final_fresco = area_final_fresco_bbox-area_fragments_of_final_fresco
        
        # Metrics
        metric_mean_distance_between_fragments = float(np.mean(np.array(distances_between_fragments, dtype=np.float32)))
        metric_min_distance_between_fragments = min(distances_between_fragments)
        metric_max_distance_between_fragments = max(distances_between_fragments)
        metric_ratio_gap_area_over_bbox_area = area_gap_of_final_fresco / area_final_fresco_bbox
        metric_ratio_gap_area_over_gt_area = area_gap_of_final_fresco / area_gt_fresco
        metric_ratio_area_diff = (area_final_fresco_bbox - area_gt_fresco) / area_gt_fresco

        if self.sim.debug:
            print(f"metric_ratio_gap_area_over_bbox_area: {metric_ratio_gap_area_over_bbox_area}")
            print(f"metric_ratio_gap_area_over_gt_area: {metric_ratio_gap_area_over_gt_area}")
      
        gap_metrics = {
            "metric_mean_distance_between_fragments": metric_mean_distance_between_fragments,
            "metric_min_distance_between_fragments": metric_min_distance_between_fragments,
            "metric_max_distance_between_fragments": metric_max_distance_between_fragments,
            "metric_ratio_gap_area_over_bbox_area": metric_ratio_gap_area_over_bbox_area,
            "metric_ratio_gap_area_over_gt_area": metric_ratio_gap_area_over_gt_area,
            "metric_ratio_area_diff": metric_ratio_area_diff
        }
        gap_metrics_base_values = {
            "distances_between_fragments": distances_between_fragments,
            "area_gap_of_final_fresco": area_gap_of_final_fresco,
            "area_fragments_of_final_fresco": area_fragments_of_final_fresco,
            "area_final_fresco_bbox": area_final_fresco_bbox,
            "area_gt_fresco": area_gt_fresco
        }

        return gap_metrics, gap_metrics_base_values
    
    # Tolerance is for matching of fragments
    def get_centroid_metrics(self, ref_multipolygon, comp_multipolygon, tolerance = 1e-3):
        # Lists to store individual distances and angle differences
        distances = []
        angle_differences = []

        # Iterate over polygons in the first MultiPolygon
        for i, comp_polygon in enumerate(comp_multipolygon.geoms):
            # Get corresponding polygon in the second MultiPolygon
            ref_polygon = ref_multipolygon.geoms[i]

            # Find the centroid of each polygon
            ref_centroid = ref_polygon.centroid
            comp_centroid = comp_polygon.centroid

            # Calculate the translation needed to align the centroids
            translation_x = comp_centroid.x - ref_centroid.x
            translation_y = comp_centroid.y - ref_centroid.y

            # Translate ref_polygon to match the position of comp_polygon
            translated_ref_polygon = translate(ref_polygon, translation_x, translation_y)

            # Brute-force rotation to match the angle in both directions
            for angle_increment in np.arange(0, 361, 0.1):
                # Rotate in both positive and negative directions
                rotated_ref_polygon_pos = rotate(translated_ref_polygon, angle_increment)
                rotated_ref_polygon_neg = rotate(translated_ref_polygon, -angle_increment)

                # Check if the rotated polygons almost match comp_polygon
                if rotated_ref_polygon_pos.equals_exact(comp_polygon, tolerance):
                    # Calculate the Euclidean distance between centroids
                    euclidean_distance_pos = ref_centroid.distance(comp_centroid)
                    angle_difference_pos = angle_increment

                    # Append distances and angle differences to the lists
                    distances.append(euclidean_distance_pos)
                    angle_differences.append(angle_difference_pos)

                    
                    if self.sim.debug:
                        print(f"Comparison {i}: Positive Rotation - Euclidean Distance: {euclidean_distance_pos}, Angle Difference: {angle_difference_pos} degrees")
                    
                    # Abort the comparison as soon as a valid solution is found
                    break

                if rotated_ref_polygon_neg.equals_exact(comp_polygon, tolerance):
                    # Calculate the Euclidean distance between centroids
                    euclidean_distance_neg = ref_centroid.distance(comp_centroid)
                    angle_difference_neg = -angle_increment

                    # Append distances and angle differences to the lists
                    distances.append(euclidean_distance_neg)
                    angle_differences.append(angle_difference_neg)

                    if self.sim.debug:
                        print(f"Comparison {i}: Negative Rotation - Euclidean Distance: {euclidean_distance_neg}, Angle Difference: {angle_difference_neg} degrees")
                    
                    # Abort the comparison as soon as a valid solution is found
                    break

        # Calculate the mean of distances and angle differences using numpy
        mean_distance = np.mean(distances)
        mean_angle_difference = np.mean(np.abs(angle_differences))

        # Print the mean distance and mean angle difference in degrees
        if self.sim.debug:
            print(f"\nMean Euclidean Distance: {mean_distance}")
            print(f"Mean Absolute Angle Difference: {mean_angle_difference} degrees")

        centroid_metrics = {
            "metric_mean_centroid_distances": mean_distance,
            "metric_mean_angle_differences": mean_angle_difference
        }
        centroid_metrics_base_values = {
            "centroid_distances_of_final_fresco": distances,
            "centroid_angle_differences_of_final_fresco": angle_differences
        }

        return centroid_metrics, centroid_metrics_base_values
    
    def get_collision_metrics_via_sim(self):
            # Collisions per fragment bool
            fragment_collision = self.sim.frag_contact_history
            plane_collision = self.sim.plane_contact_history

            mean_fragment_collision = np.mean(fragment_collision)
            mean_plane_collision = np.mean(plane_collision)

            # Collision counts
            fragment2robot_collision_count = self.sim.fragment2robot_contacts_count_history
            fragment2fragment_collision_count = self.sim.fragment2fragment_contacts_count_history
            plane_collision_count = self.sim.plane_contacts_count_history

            mean_fragment2robot_collision_count = np.mean(fragment2robot_collision_count)
            mean_fragment2fragment_collision_count = np.mean(fragment2fragment_collision_count)
            mean_plane_collision_count = np.mean(plane_collision_count)

            collision_metrics = {
                "metric_mean_fragment_collision_per_placing_fragment_bool": mean_fragment_collision,
                "metric_mean_plane_collision_per_placing_fragment_bool": mean_plane_collision,
                "metric_mean_fragment2robot_collision_count": mean_fragment2robot_collision_count,
                "metric_mean_fragment2fragment_collision_count": mean_fragment2fragment_collision_count,
                "metric_mean_plane_collision_count": mean_plane_collision_count
            }
            collsion_metrics_base_values = {
                "fragment_collision_per_placing_fragment_bool": fragment_collision,
                "plane_collision_per_placing_fragment_bool": plane_collision,
                "fragment2robot_collision_count": fragment2robot_collision_count,
                "fragment2fragment_collision_count": fragment2fragment_collision_count,
                "plane_collision_count": plane_collision_count
            }

            return collision_metrics, collsion_metrics_base_values
    
    def create_evaluation_json_file(self, gt_data, eval_path, eval_name, assembly_plan_type):
        # Eval file and evaluation parameters
        self.sim.common_utils.create_folder(eval_path)
        eval_json_base_data = copy.deepcopy(gt_data)
        eval_json_base_data[eval_name] = {}
        eval_json_base_data[eval_name][assembly_plan_type] = {}
        return eval_json_base_data
    
    def save_evaluation_json_file(self, eval_path, eval_name, model_name, eval_json_data, metrics, metrics_base_values, placed_fresco_in_fresco_coords, placed_fresco_in_world_coords, assembly_plan_type, scale_factor=None):
        print("Save all metrics in json")
        placed_fresco_array_in_fresco_coords = self.sim.shapely_utils.convert_shapely_multi_polygon_to_fresco_array(placed_fresco_in_fresco_coords)
        placed_fresco_array_in_world_coords = self.sim.shapely_utils.convert_shapely_multi_polygon_to_fresco_array(placed_fresco_in_world_coords)
        
        if model_name != "":
            model_temp_str = model_name.split("@")
            eval_json_data[eval_name][assembly_plan_type]["model_config"] =  model_temp_str[0]
            eval_json_data[eval_name][assembly_plan_type]["model_name"] =  model_temp_str[1]
        
        if scale_factor is None:
            eval_json_data[eval_name][assembly_plan_type]["metrics"] =  metrics
            eval_json_data[eval_name][assembly_plan_type]["metrics_base_values"] =  metrics_base_values
            eval_json_data[eval_name][assembly_plan_type]["placed_fresco_in_fresco_coords"] = placed_fresco_array_in_fresco_coords
            eval_json_data[eval_name][assembly_plan_type]["placed_fresco_in_world_coords"] = placed_fresco_array_in_world_coords
        else:
            scale_factor_key = "sf_"+str(scale_factor)
            eval_json_data[eval_name][assembly_plan_type][scale_factor_key] = {}
            eval_json_data[eval_name][assembly_plan_type][scale_factor_key]["metrics"] =  metrics
            eval_json_data[eval_name][assembly_plan_type][scale_factor_key]["metrics_base_values"] =  metrics_base_values
            eval_json_data[eval_name][assembly_plan_type][scale_factor_key]["placed_fresco_in_fresco_coords"] = placed_fresco_array_in_fresco_coords     
            eval_json_data[eval_name][assembly_plan_type][scale_factor_key]["placed_fresco_in_world_coords"] = placed_fresco_array_in_world_coords        
        
        eval_json_file_name = eval_path+eval_name
        if eval_name == "rl":
            eval_json_file_name += "_"+model_name
        eval_json_file_name += ".json"
        self.sim.common_utils.save_json(path=eval_json_file_name, data=eval_json_data)

    def get_final_fresco_polygons(self):
            print("Get final fresco")
            gt_fresco_polygons_in_fresco_coords = self.sim.shapely_utils.convert_fresco_array_to_shapely_multi_polygon(self.sim.gt_data["fresco"])
            final_fresco_polygons_in_world_coords = self.get_current_fresco_polygons_via_sim(self.sim.table_fragments, gt_fresco_polygons_in_fresco_coords)
            final_fresco_polygons_in_fresco_coords = self.transform_world_to_fresco_coords(final_fresco_polygons_in_world_coords)

            return gt_fresco_polygons_in_fresco_coords, final_fresco_polygons_in_world_coords, final_fresco_polygons_in_fresco_coords

    def get_metrics(self, gt_fresco_polygons_in_fresco_coords, final_fresco_polygons_in_fresco_coords, used_metrics=["all"]):
            print("Calculate all metrics")

            metrics = {}
            metrics_base_values = {}

            if ("all" or "gap") in used_metrics:
                if self.sim.debug:
                    print("Check fresco gaps")
                gap_metrics, gap_metrics_base_values = self.get_gap_metrics(final_fresco_polygons_in_fresco_coords, self.sim.gt_data["header"])
                metrics["gap_metrics"] = gap_metrics
                metrics_base_values["gap_metrics_base_values"] = gap_metrics_base_values

            if ("all" or "centroid") in used_metrics:
                if self.sim.debug:
                    print("Check fresco centroid shift and orientation")
                centroid_metrics, centroid_metrics_base_values = self.get_centroid_metrics(gt_fresco_polygons_in_fresco_coords, final_fresco_polygons_in_fresco_coords)
                metrics["centroid_metrics"] = centroid_metrics
                metrics_base_values["centroid_metrics_base_values"] = centroid_metrics_base_values
            
            if ("all" or "collision") in used_metrics:
                if self.sim.debug:
                    print("Check collisions")
                collision_metrics, collision_metrics_base_values = self.get_collision_metrics_via_sim()
                metrics["collision_metrics"] = collision_metrics
                metrics_base_values["collision_metrics_base_values"] = collision_metrics_base_values
            
            return metrics, metrics_base_values

    def save_debug_images(self, img_path, eval_name, gt_fresco_polygons_in_fresco_coords, final_fresco_polygons_in_fresco_coords, scale_factor=None, visualize=False, save_plot=False):
        print("Save evaluation images")
        if eval_name == "rl":
            plot_name = "Reinforcement Learning\n"
        elif eval_name == "baseline_scaling_fresco":
            plot_name = "Baseline Scaling Fresco\n"
        elif eval_name == "baseline_relative_placing":
            plot_name = "Baseline Relative Placing\n"

        plot_name_comparison = plot_name + "Layout vs. Assembly"
        if scale_factor is not None:
            plot_name_comparison += " (sf="+str(scale_factor)+")"
        self.sim.common_utils.plot_fresco_comparison_image(img_path+"_comparison", [gt_fresco_polygons_in_fresco_coords, final_fresco_polygons_in_fresco_coords], name=plot_name_comparison, visualize=visualize, save_plot=save_plot)

        if self.sim.debug:
            plot_name_assembly = plot_name + "Fresco Assembly"
            if scale_factor is not None:
                plot_name_assembly += " (sf="+str(scale_factor)+")"
            self.sim.common_utils.plot_fresco_image(img_path+"_assembly", final_fresco_polygons_in_fresco_coords, name=plot_name_assembly, visualize=visualize, save_plot=save_plot)

    def evaluate(self, eval_name, used_metrics=["all"], assembly_plan_type="assembly_plan_snake", scale_factor=None, first_save_cycle=True, visualize=False, save_plot=True):
            print("\nEvaluate final fresco")
            if eval_name == "rl":
                model_name = self.sim.model_name
            else:
                model_name = ""
            
            if first_save_cycle:
                self.eval_path = self.sim.root_path + "fragment_evaluation/data/" + "fresco_" + str(self.sim.fresco_no) + "/" + str(eval_name) + "/"
                eval_json_base_data = self.create_evaluation_json_file(self.sim.shapely_utils.get_eval_fresco_ground_truth(self.sim.fresco_no,self.sim.root_path), self.eval_path, eval_name, assembly_plan_type)
                self.json_data = copy.deepcopy(eval_json_base_data)
            
            gt_fresco_polygons_in_fresco_coords, final_fresco_polygons_in_world_coords, final_fresco_polygons_in_fresco_coords =  self.get_final_fresco_polygons()
            metrics, metrics_base_values = self.get_metrics(gt_fresco_polygons_in_fresco_coords, final_fresco_polygons_in_fresco_coords, used_metrics)
            self.save_evaluation_json_file(self.eval_path, eval_name, model_name, self.json_data, metrics, metrics_base_values, final_fresco_polygons_in_fresco_coords, final_fresco_polygons_in_world_coords, assembly_plan_type, scale_factor)
            
            img_path = self.eval_path+eval_name
            if eval_name == "rl":
                img_path += "_"+model_name
            if scale_factor is not None:
                self.sim.common_utils.remove_fresco_image(img_path=img_path+"_sf_"+"*")
                img_path += "_sf_"+str(scale_factor)      
            self.save_debug_images(img_path, eval_name, gt_fresco_polygons_in_fresco_coords, final_fresco_polygons_in_fresco_coords, scale_factor, visualize=visualize, save_plot=save_plot)
            
    def evaluate_rl_agent(self, eval_name = "rl", used_metrics=["all"], assembly_plan_type="assembly_plan_snake"):
        self.sim.table_fragments[self.sim.placing_fragment["id"]] = self.sim.placing_fragment
        self.evaluate(eval_name=eval_name, used_metrics=used_metrics, assembly_plan_type=assembly_plan_type)

    def evaluate_rl_agent_during_training(self, eval_name = "rl", used_metrics=["all"], assembly_plan_type="assembly_plan_snake", save_plot=False):
        self.sim.table_fragments[self.sim.placing_fragment["id"]] = self.sim.placing_fragment
        self.evaluate(eval_name=eval_name, used_metrics=used_metrics, assembly_plan_type=assembly_plan_type, save_plot=save_plot)

    def evaluate_baseline_relative_placing(self, eval_name = "baseline_relative_placing", used_metrics=["all"], assembly_plan_type="assembly_plan_snake"):
        self.evaluate(eval_name=eval_name, used_metrics=used_metrics, assembly_plan_type=assembly_plan_type)

    def evaluate_baseline_scaling_fresco(self, scale_factor, eval_name = "baseline_scaling_fresco", used_metrics=["all"], assembly_plan_type="assembly_plan_snake", first_save_cycle=False):
        self.evaluate(eval_name=eval_name, used_metrics=used_metrics, assembly_plan_type=assembly_plan_type, scale_factor=scale_factor, first_save_cycle=first_save_cycle)

    def get_placing_fragment_overlap(self, inflation=0.0, gripper_footprint=False):
            if self.sim.placing_fragment["first_fragment"] == True:
                overlap_bool = False
            else:
                gt_fresco_polygons_in_fresco_coords = self.sim.shapely_utils.convert_fresco_array_to_shapely_multi_polygon(self.sim.gt_data["fresco"])
                current_table_fragment_polygons_in_world_coords = self.get_current_fresco_polygons_via_sim(self.sim.table_fragments, gt_fresco_polygons_in_fresco_coords)
                # current_fresco_polygons_in_fresco_coords = self.transform_world_to_fresco_coords(current_table_fragment_polygons_in_world_coords)
                current_placing_fragment_polygon_in_world_coords = self.get_current_placing_fragment_polygon_pose_via_sim(self.sim.placing_fragment, gt_fresco_polygons_in_fresco_coords)
                # current_placing_fragment_polygon_in_fresco_coords = self.transform_world_to_fresco_coords(current_placing_fragment_polygon_in_world_coords)

                if gripper_footprint:
                    finger_pad_width = 0.022
                    finger_pad_length = 0.00635
                    gripper_opening = self.sim.get_current_gripper_opening()
                    if gripper_opening > self.sim.gripper_opening_after_grasp:
                        gripper_opening_offset = finger_pad_length + (gripper_opening - self.sim.gripper_opening_after_grasp)/2
                    else:
                        gripper_opening_offset = finger_pad_length
                    gripper_yaw = self.sim.get_current_tcp_yaw_in_euler()
                    #attach_gripper_footprint(self,polygon_coordinates, yaw_radians, gripper_offset = 0.0065, safe_offset = 0.01, inflation = 0.001,length = 0.013, width = 0.022, visualize=False)
                    current_placing_fragment_polygon_in_world_coords = self.sim.relative_placing_utils.attach_gripper_footprint(
                        polygon_coordinates=current_placing_fragment_polygon_in_world_coords.geoms[0],
                        gripper_offset = gripper_opening_offset,
                        yaw_radians = gripper_yaw,
                        safe_offset = 0.0,
                        inflation = 0.0,
                        width = finger_pad_width, # pad width from urdf
                        length = finger_pad_length, # pad length from urdf
                        visualize = False
                        )
                    current_placing_fragment_polygon_in_world_coords = MultiPolygon([current_placing_fragment_polygon_in_world_coords])
                
                if inflation > 0.0:
                    current_placing_fragment_polygon_in_world_coords = MultiPolygon([current_placing_fragment_polygon_in_world_coords.buffer(inflation)])

                if current_table_fragment_polygons_in_world_coords.is_valid == True:
                    overlap_bool = bool(overlaps(current_table_fragment_polygons_in_world_coords, current_placing_fragment_polygon_in_world_coords))
                else:
                    array_of_current_table_fragment_polygons_in_world_coords = self.sim.shapely_utils.convert_shapely_multi_polygon_to_array_of_shapely_polygons(current_table_fragment_polygons_in_world_coords)
                    overlap_bool_list = []
                    for polygon in array_of_current_table_fragment_polygons_in_world_coords:
                        overlap_bool_list.append(bool(overlaps(polygon, Polygon(current_placing_fragment_polygon_in_world_coords.geoms[0]))))
                    if any(overlap_bool_list):
                        overlap_bool = True
                    else:
                        overlap_bool = False
            
            return overlap_bool 