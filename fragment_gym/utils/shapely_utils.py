#!/usr/bin/python3

import os
from shapely import (
    Polygon,
    Point,
    MultiPolygon,
    MultiPoint
)
from fragment_gym.utils import common_utils
from fragment_gym.utils import fresco_scaling_utils as scaling_utils

class ShapelyUtils():
    def __init__(self, config):
        self.config = config
        self.common_utils = common_utils.Common()
        self.scaling_utils = scaling_utils.FrescoScalingUtils(config=self.config)

    def get_fresco_ground_truth(self, fresco_no, root_path):
        gt_path = root_path + "meshes/fragments/fresco_"+ str(fresco_no) + "/ground_truth.json"
        if os.path.isfile(gt_path):
            gt_data = self.common_utils.read_json_file(gt_path)
        else:
            print("Ground truth file of fresco could not be loaded.")
            raise ValueError
        
        scale_factor=float(self.config["fresco_scale_factor"])
        if scale_factor != 0.0:
            fresco_polygons = self.convert_fresco_array_to_shapely_multi_polygon(gt_data["fresco"])
            shapely_fresco_polygons_scaled, shapely_fresco_centroids_scaled = self.scaling_utils.scale_fresco(fresco_polygons=fresco_polygons, scale_factor=scale_factor)

            shapely_fresco_polygons_scaled_bounding_box = shapely_fresco_polygons_scaled.envelope
            shapely_fresco_polygons_scaled_bounding_box_coordinates = shapely_fresco_polygons_scaled_bounding_box.bounds
            length = shapely_fresco_polygons_scaled_bounding_box_coordinates[2] - shapely_fresco_polygons_scaled_bounding_box_coordinates[0]
            width = shapely_fresco_polygons_scaled_bounding_box_coordinates[3] - shapely_fresco_polygons_scaled_bounding_box_coordinates[1]
            gt_data["header"]["length"] = length
            gt_data["header"]["width"] = width
            
            fresco_centroids_scaled = self.convert_shapely_multi_point_to_point_array(shapely_fresco_centroids_scaled)
            gt_data["centroids"] = fresco_centroids_scaled

        return gt_data
    
    def get_eval_fresco_ground_truth(self, fresco_no, root_path):
        gt_path = root_path + "meshes/fragments/fresco_"+ str(fresco_no) + "/ground_truth.json"
        if os.path.isfile(gt_path):
            gt_data = self.common_utils.read_json_file(gt_path)
        else:
            print("Ground truth file of fresco could not be loaded.")
            raise ValueError

        return gt_data

    def convert_fresco_array_to_shapely_multi_polygon(self, fresco_array):
        fresco_polygons = []

        for i in range(len(fresco_array)):
            fragment = Polygon(fresco_array[i])
            fresco_polygons.append(fragment)

        return MultiPolygon(fresco_polygons)   

    def convert_shapely_multi_polygon_to_array_of_shapely_polygons(self, shapely_multi_polygon):
        polygon_array = []

        for i in range(len(shapely_multi_polygon.geoms)):
            polygon = Polygon(shapely_multi_polygon.geoms[i])
            polygon_array.append(polygon)

        return polygon_array
    
    def convert_shapely_multi_polygon_to_fresco_array(self, shapely_multi_polygon):
        polygon_array = []
        for polygon in shapely_multi_polygon.geoms:
            point_array = []
            for point_tuple in polygon.exterior.coords:            
                point_array.append([point_tuple[0], point_tuple[1]])
            polygon_array.append(point_array)

        return polygon_array
       
    def convert_point_array_to_shapely_multi_point(self, point_array):
        points = []

        for point in point_array:
            points.append(Point(point[0],point[1]))

        return MultiPoint(points)

    def convert_shapely_multi_point_to_point_array(self, shapely_multi_point):
        point_array = []

        for i in range(len(shapely_multi_point.geoms)):
            point = Point(shapely_multi_point.geoms[i])

            point_array.append([point.x, point.y])

        return point_array

    def calculate_centroids_of_shapely_multi_polygon(self, polygons):
        points = []
        for polygon in polygons.geoms:
            points.append(polygon.centroid)
        return MultiPoint(points)
    
    def calculate_min_dist_between_polygons(self, polygons):
        min_dist = float("inf")
        for i in range(len(polygons.geoms)):
            for j in range(i + 1, len(polygons.geoms)):
                min_dist = min(
                    min_dist,
                    polygons.geoms[i].distance(
                        polygons.geoms[j]
                    ),
                )

        return min_dist