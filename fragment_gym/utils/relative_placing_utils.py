#!/usr/bin/python3

from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate, rotate
import matplotlib.pyplot as plt
import math

class RelativePlacingUtils():
    def __init__(self, config, sim, p_id, frescoes_path):
        self.config = config
        self.sim = sim
        self._p = p_id
        self.frescoes_path = frescoes_path

    def is_line_touching_polygon(self, line_segment, polygon):
        return line_segment.intersects(polygon)

    def attach_gripper_footprint(self,polygon_coordinates, yaw_radians, gripper_offset = 0.0065, safe_offset = 0.01, inflation = 0.001,
                                  length = 0.013, width = 0.022, visualize=False):
        polygon = Polygon(polygon_coordinates)
        centroid = polygon.centroid

        # Calculate the initial coordinates of the four points (x1, x2, x3, x4) forming the rectangle
        x1 = Point(centroid.x - length / 2, centroid.y - width / 2)
        x2 = Point(centroid.x + length / 2, centroid.y - width / 2)
        x3 = Point(centroid.x + length / 2, centroid.y + width / 2)
        x4 = Point(centroid.x - length / 2, centroid.y + width / 2)        

        # Create three separate polygons with different colors
        polygon_1 = None
        polygon_2 = None
        polygon_3 = None

        curr = 0
        # Rotate and adjust the points to avoid intersection with the main polygon
        while self.is_line_touching_polygon(LineString([x1, x4]), polygon):
            x1 = rotate(x1, angle=math.degrees(curr), origin=centroid)
            x2 = rotate(x2, angle=math.degrees(curr), origin=centroid)    
            x4 = rotate(x4, angle=math.degrees(curr), origin=centroid)
            x3 = rotate(x3, angle=math.degrees(curr), origin=centroid)
            x1 = translate(x1, xoff=-0.0001, yoff=0)
            x4 = translate(x4, xoff=-0.0001, yoff=0)
            curr = yaw_radians
            x1 = rotate(x1, angle=math.degrees(curr), origin=centroid)
            x2 = rotate(x2, angle=math.degrees(curr), origin=centroid)    
            x4 = rotate(x4, angle=math.degrees(curr), origin=centroid)
            x3 = rotate(x3, angle=math.degrees(curr), origin=centroid)
            curr = -yaw_radians

        while self.is_line_touching_polygon(LineString([x2, x3]), polygon):
            x1 = rotate(x1, angle=math.degrees(curr), origin=centroid)
            x2 = rotate(x2, angle=math.degrees(curr), origin=centroid)    
            x4 = rotate(x4, angle=math.degrees(curr), origin=centroid)
            x3 = rotate(x3, angle=math.degrees(curr), origin=centroid)        
            x2 = translate(x2, xoff=0.0001, yoff=0)  
            x3 = translate(x3, xoff=0.0001, yoff=0)
            curr = yaw_radians
            x1 = rotate(x1, angle=math.degrees(curr), origin=centroid)
            x2 = rotate(x2, angle=math.degrees(curr), origin=centroid)    
            x4 = rotate(x4, angle=math.degrees(curr), origin=centroid)
            x3 = rotate(x3, angle=math.degrees(curr), origin=centroid)
            curr = -yaw_radians

        # Create polygon 1
        x1 = rotate(x1, angle=math.degrees(curr), origin=centroid)
        x2 = rotate(x2, angle=math.degrees(curr), origin=centroid)    
        x4 = rotate(x4, angle=math.degrees(curr), origin=centroid)
        x3 = rotate(x3, angle=math.degrees(curr), origin=centroid)
        polygon_1 = Polygon([x1, x2, x3, x4])

        # Translate and create polygon 2
        x1 = translate(x1, xoff=-gripper_offset, yoff=0)
        x4 = translate(x4, xoff=-gripper_offset, yoff=0)
        x2 = translate(x2, xoff=gripper_offset, yoff=0)  
        x3 = translate(x3, xoff=gripper_offset, yoff=0)
        polygon_2 = Polygon([x1, x2, x3, x4])

        # Take union of polygon 1 and polygon 2
        polygon_1_and_2 = polygon_1.union(polygon_2)

        # Translate and create polygon 3
        x1 = translate(x1, xoff=-safe_offset, yoff=0)
        x4 = translate(x4, xoff=-safe_offset, yoff=0)
        x2 = translate(x2, xoff=safe_offset, yoff=0)  
        x3 = translate(x3, xoff=safe_offset, yoff=0)
        polygon_3 = Polygon([x1, x2, x3, x4])

        # Take union of polygon 1, polygon 2, and polygon 3
        result_union = polygon_1_and_2.union(polygon_3)      

        result_union = rotate(result_union, angle=math.degrees(-curr), origin=centroid)
        
        polygon_1 = rotate(polygon_1, angle=math.degrees(-curr), origin=centroid)
        polygon_2 = rotate(polygon_2, angle=math.degrees(-curr), origin=centroid)
        polygon_3 = rotate(polygon_3, angle=math.degrees(-curr), origin=centroid)
        result_union = result_union.union(polygon)

        result_union = result_union.buffer(inflation)

        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(*polygon.exterior.xy, label="Main fragment",color='purple')
            plt.fill(*polygon.exterior.xy, alpha=0.3,color='purple')
            plt.plot(*polygon_1.exterior.xy, label="Dummy strip", color='red')
            plt.fill(*polygon_1.exterior.xy, alpha=0.3, color='red')
            plt.plot(*polygon_2.exterior.xy, label="Gripper offset", color='yellow')
            plt.fill(*polygon_2.exterior.xy, alpha=0.3, color='yellow')
            plt.plot(*polygon_3.exterior.xy, label="Openning offset", color='green')
            plt.fill(*polygon_3.exterior.xy, alpha=0.3, color='green')
            plt.plot(*result_union.exterior.xy, label="Union", color='blue')
            plt.fill(*result_union.exterior.xy, alpha=0.3, color='blue')
            plt.grid(True)
            plt.title("Fragment with gripper and openning offsets")
            plt.axis('equal')  # Fix the aspect ratio
            plt.legend()
            plt.show()

        return result_union
    
    def plot_current_and_multi_polygon(self, current_polygon, multi_polygon, positions):
        fig, ax = plt.subplots()        
        # Plot the MultiPolygon
        for polygon in multi_polygon.geoms:
            x, y = polygon.exterior.xy
            ax.plot(x, y, color='b')
        # Plot the current polygon
        x, y = current_polygon.exterior.xy
        ax.fill(x, y, color='r', alpha=0.7, label='Current Polygon')        
        # Extract the coordinates of positions
        x_positions, y_positions = zip(*[(point.x, point.y) for point in positions])        
        # Plot the positions
        plt.scatter(x_positions, y_positions, color='g', label='Positions')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Current Polygon, MultiPolygon, and Positions at Translation')        
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def place_along_centroids(self, multi_polygon, current_polygon, current_polygon_centroid, fragment_step_increment=0.01, visualize=False):
        if multi_polygon == []:
            return current_polygon.centroid

        # Step 1: Place the current polygon at its centroid
        move_vectors = []
        if visualize:
            self.plot_multi_polygon_circle_current_polygon(multi_polygon, current_polygon)
        # current_centroid = current_polygon.centroid

        # Step 2: Check if it intersects with the multi-polygon
        while current_polygon.intersects(multi_polygon):   
            intersection_detected = False
            
            for i, polygon in enumerate(multi_polygon.geoms):
                if current_polygon.intersects(polygon):                   
                    # intersection_areas.append(current_polygon.intersection(polygon).area)

                    intersection_detected = True
                    last_polygon_centroid = polygon.centroid
                    move_vector = Point(current_polygon_centroid.centroid.x - last_polygon_centroid.x, current_polygon_centroid.centroid.y - last_polygon_centroid.y)

                    # Normalize the move vector to have unit magnitude
                    magnitude = (move_vector.x ** 2 + move_vector.y ** 2) ** 0.5
                    normalized_move_vector = Point(move_vector.x / magnitude, move_vector.y / magnitude)
                    move_vectors.append(normalized_move_vector)

            if intersection_detected:
                # Calculate the resultant move vector                
                resultant_move_vector = Point(sum(move_vector.x for move_vector in move_vectors), sum(move_vector.y for move_vector in move_vectors))

                current_polygon = translate(current_polygon, resultant_move_vector.x * fragment_step_increment, resultant_move_vector.y * fragment_step_increment)          

                # if visualize:
                #     self.plot_multi_polygon_circle_current_polygon(multi_polygon, current_polygon)
            else:
                break

            
        if visualize:
            self.plot_multi_polygon_circle_current_polygon(multi_polygon, current_polygon)
        return current_polygon.centroid
    
    def plot_multi_polygon_circle_current_polygon(self, multi_polygon, current_polygon, best_point=None, move_vector=None):
        fig, ax = plt.subplots()        
        # Plot the MultiPolygon
        for polygon in multi_polygon.geoms:
            x, y = polygon.exterior.xy
            ax.plot(x, y, color='b')
        # Plot the current polygon
        x, y = current_polygon.exterior.xy
        ax.fill(x, y, color='r', alpha=0.7)        
        if not best_point == None:
            # Plot the best point
            ax.plot(best_point.x, best_point.y, 'x', markersize=10)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        if not best_point == None:
            plt.title('Fragments on table, Current Fragment, and Best Point')
        else:
            plt.title('Fragments on table, and Current Fragment')
        plt.axis('equal')  # Fix the aspect ratio
        plt.show()



if __name__ == "__main__":
    # Test code
    pass