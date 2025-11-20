#!/usr/bin/python3

import pybullet as p
import numpy as np
import math

class DebugLineRuler:
    def __init__(self, config, sim, p_id, ruler_lines, line_length, line_width, line_color, line_length_orientation=0.1, line_color_orientation=[0,0,0], reset_all=False):
        self.config = config
        self.p_id = p_id
        self.sim = sim
        self.reset_all = reset_all
        self.ruler_lines = ruler_lines
        self.line_length = line_length
        self.line_width = line_width
        self.line_color = line_color
        self.marker_ids = []

        for line in self.ruler_lines:
            self.draw_debug_line(line[0], line[1], self.line_width, self.line_color)

    def draw_debug_line(self, line_start, line_end, line_width, line_color):
        marker_id = self.p_id.addUserDebugLine(lineFromXYZ=line_start, lineToXYZ=line_end, lineColorRGB=line_color, lineWidth=line_width, physicsClientId=self.sim.client_id)
        self.marker_ids.append(marker_id)
    
    def __del__(self):
        for marker_id in self.marker_ids:
            self.p_id.removeUserDebugItem(marker_id, physicsClientId=self.sim.client_id)
        if self.reset_all:
            self.p_id.removeAllUserDebugItems(physicsClientId=self.sim.client_id) # Temporary solution

class DebugLinesCorrespondingCorners:
    def __init__(self, config, sim, p_id, current_corners, line_length, line_width, line_color, line_length_orientation=0.1, line_color_orientation=[0,0,0], reset_all=False):
        self.config = config
        self.p_id = p_id
        self.sim = sim
        self.reset_all = reset_all
        self.fragment_corners = current_corners["fragment_corners"]
        self.neighbour_corners = current_corners["neighbour_corners"]
        self.line_length = line_length
        self.line_width = line_width
        self.line_color = line_color
        self.marker_ids = []

        for i in range(len(self.fragment_corners)):
            self.draw_debug_line(self.fragment_corners[i], self.neighbour_corners[i], self.line_width, self.line_color)

    def draw_debug_line(self, line_start, line_end, line_width, line_color):
        marker_id = self.p_id.addUserDebugLine(lineFromXYZ=line_start.tolist(), lineToXYZ=line_end.tolist(), lineColorRGB=line_color, lineWidth=line_width, physicsClientId=self.sim.client_id)
        self.marker_ids.append(marker_id)
    
    def __del__(self):
        for marker_id in self.marker_ids:
            self.p_id.removeUserDebugItem(marker_id, physicsClientId=self.sim.client_id)
        if self.reset_all:
            self.p_id.removeAllUserDebugItems(physicsClientId=self.sim.client_id) # Temporary solution

class DebugCross:
    def __init__(self, config, sim, p_id, position, line_length, line_width, line_color, line_length_orientation=0.1, line_color_orientation=[0,0,0], reset_all=False):
        self.config = config
        self.p_id = p_id
        self.sim = sim
        self.reset_all = reset_all
        self.position = position[:3]
        self.yaw = 9999.0
        if len(position) > 3:
            self.yaw = position[3]
        self.line_color_orientation = line_color_orientation
        self.line_length = line_length
        self.line_length_orientation = line_length_orientation
        self.line_width = line_width
        self.line_color = line_color

        self.marker_ids = []
        self.draw_debug_cross(self.position, self.line_length, self.line_width, self.line_color)
        if self.yaw != 9999.0:
            self.draw_debug_orientation(self.position, self.yaw, self.line_length_orientation, self.line_width, self.line_color_orientation)

    def draw_debug_orientation(self, position, yaw, line_length, line_width, line_color):
        # Draw orientation (yaw angle)
        line_end = position + np.array([math.cos(yaw)*line_length, math.sin(yaw)*line_length, 0.0])
        marker_id = self.p_id.addUserDebugLine(lineFromXYZ=position, lineToXYZ=line_end, lineColorRGB=line_color, lineWidth=line_width, physicsClientId=self.sim.client_id)
        self.marker_ids.append(marker_id)

    def draw_debug_cross(self, position, line_length, line_width, line_color):
        line_width = line_width
        line_length = line_length
        line_start = position
        line_end = position

        # Draw position
        for i in range (0,3):
            line_start_temp = line_start.copy()
            line_start_temp[i] = position[i] - line_length/2
            line_end_temp = line_end.copy()
            line_end_temp[i] = position[i] + line_length/2
            marker_id = self.p_id.addUserDebugLine(lineFromXYZ=line_start_temp.tolist(), lineToXYZ=line_end_temp.tolist(), lineColorRGB=line_color, lineWidth=line_width, physicsClientId=self.sim.client_id)
            self.marker_ids.append(marker_id)
    
    def __del__(self):
        for marker_id in self.marker_ids:
            self.p_id.removeUserDebugItem(marker_id, physicsClientId=self.sim.client_id)
        if self.reset_all:
            self.p_id.removeAllUserDebugItems(physicsClientId=self.sim.client_id) # Temporary solution

class SphereMarker:
    def __init__(self, position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0):
        self.p_id = p_id
        position = np.array(position)
        vs_id = p_id.createVisualShape(
            self.p_id.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=self.sim.client_id)

        self.marker_id = self.p_id.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id,
            basePosition=position,
            useMaximalCoordinates=False
        )

        self.debug_item_ids = list()
        if text is not None:
            self.debug_item_ids.append(
                self.p_id.addUserDebugText(text, position + radius, physicsClientId=self.sim.client_id)
            )

        if orientation is not None:
            # x axis
            axis_size = 2 * radius
            rotation_mat = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

            # x axis
            x_end = np.array([[axis_size, 0, 0]]).transpose()
            x_end = np.matmul(rotation_mat, x_end)
            x_end = position + x_end[:, 0]
            self.debug_item_ids.append(
                self.p_id.addUserDebugLine(position, x_end, lineColorRGB=(1, 0, 0), physicsClientId=self.sim.client_id)
            )
            # y axis
            y_end = np.array([[0, axis_size, 0]]).transpose()
            y_end = np.matmul(rotation_mat, y_end)
            y_end = position + y_end[:, 0]
            self.debug_item_ids.append(
                self.p_id.addUserDebugLine(position, y_end, lineColorRGB=(0, 1, 0), physicsClientId=self.sim.client_id)
            )
            # z axis
            z_end = np.array([[0, 0, axis_size]]).transpose()
            z_end = np.matmul(rotation_mat, z_end)
            z_end = position + z_end[:, 0]
            self.debug_item_ids.append(
                self.p_id.addUserDebugLine(position, z_end, lineColorRGB=(0, 0, 1), physicsClientId=self.sim.client_id)
            )

    def __del__(self):
        self.p_id.removeBody(self.marker_id, physicsClientId=self.sim.client_id)
        for debug_item_id in self.debug_item_ids:
            self.p_id.removeUserDebugItem(debug_item_id, physicsClientId=self.sim.client_id)