#!/usr/bin/python3

import numpy as np
from collections import deque
from fragment_gym.utils import debug_utils

class Manipulation():
    def __init__(self, config, sim, p_id):
        self.config = config
        self.sim = sim
        self._p = p_id
        self.debug_pos_diff_list = np.empty((0,3), dtype=np.float32)
        self.debug_ori_diff_list = np.empty((0,3), dtype=np.float32)
        self.debug_yaw_diff_list = []
        self.subgoal_debug_cross = None

    def spawn_pick_plane(self, object_parking_position):
        self.sim.pick_plane_id = self._p.loadURDF(
            self.sim.root_path + "meshes/urdf/pick_plane.urdf",
            basePosition=tuple(object_parking_position),
            baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=self._p.URDF_USE_INERTIA_FROM_FILE,
        )
        self.sim._p.changeVisualShape(self.sim.pick_plane_id, -1, rgbaColor=[210 / 255, 183 / 255, 115 / 255, 0],
                                physicsClientId=self.sim.client_id)

    def move_gripper(self, gripper_delta):
        gripper_opening_length = self.sim.get_current_gripper_joint_config() + gripper_delta
        self.sim.controlGripper(controlMode=self._p.POSITION_CONTROL, positionition=gripper_opening_length)
        return

    def get_ik_joints(self, position, orientation, link, accuracy="medium"):
        if accuracy == "medium":
            max_iter = 500
            thresh = 1e-6
        elif accuracy == "high":
            max_iter = 100000
            thresh = 1e-9
        elif accuracy == "low":
            max_iter = 1000
            thresh = 1e-5
        else:
            print("Invalid accuracy setting in IK solver.")
            raise ValueError

        joint_states = self._p.calculateInverseKinematics(self.sim.robot_id,
                                                       link,
                                                       position,
                                                       self._p.getQuaternionFromEuler(orientation),
                                                       solver = self._p.IK_DLS,
                                                       maxNumIterations = max_iter,
                                                       residualThreshold = thresh)
        
        target_joint_states = list(joint_states)[:6]
        # Check if wrist yaw > 2*pi
        if abs(target_joint_states[5]) > 2*np.pi:
            target_joint_states[5] = target_joint_states[5] % (2*np.pi)
        
        return target_joint_states

    def send_position_command(self, target_joint_states, num_motors=6, position_gain=0.4):
        self._p.setJointMotorControlArray(bodyUniqueId=self.sim.robot_id,
                                          jointIndices=[i for i in range(1, 7)],
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPositions=target_joint_states,
                                          targetVelocities=[0.] * num_motors,
                                          forces=[150, 150, 150, 28, 28, 28],
                                          positionGains=[position_gain] * num_motors,
                                          velocityGains=[1.] * num_motors)
        
    def carry_out_motion(self, target_joint_states, accuracy="medium", collision_check="false", collision_check_bodies=[], collision_check_reference=-1):
        if accuracy=="medium":
            max_it=1000
            atol=1e-4
        elif accuracy=="high":
            max_it=1000000
            atol=1e-5
        elif accuracy=="low":
            max_it=1000
            atol=1e-3
        else:
            print("Invalid accuracy setting in carry out motion.")
            raise ValueError

        if collision_check != "false":
            plane_collisions = 0
            robot_collisions = 0
            fragment_collisions = 0

        #past_joint_pos = deque(maxlen=100)
        past_joint_pos = deque(maxlen=5)
        
        joint_state = self._p.getJointStates(self.sim.robot_id, [i for i in range(1,7)])
        joint_pos = list(zip(*joint_state))[0]
        n_it = 0
        while not np.allclose(joint_pos, target_joint_states, atol) and n_it < max_it:
            self.sim.step_simulation(self.sim.per_step_iterations)
            n_it += 1
            # Check to see if the arm can't move any close to the desired joint position
            #if len(past_joint_pos) == 100 and np.allclose(past_joint_pos[-1], np.mean(past_joint_pos, axis=0), rtol=1e-06, atol=1e-09):
            if len(past_joint_pos) == 5 and np.allclose(past_joint_pos[-1], past_joint_pos, atol):
                break
            # if np.allclose(joint_pos, target_joint_states, atol):
            #     pass
            # Check to see if the arm is in collision
            if collision_check != "false":
                plane_collisions += self.sim.collision_utils.count_robot_collisions([self.sim.planeID])
                robot_collisions += self.sim.collision_utils.count_robot_collisions(collision_check_bodies)
                if collision_check_reference < 0:
                    fragment_collisions += self.sim.collision_utils.count_n_to_n_collisions(collision_check_bodies)
                else:
                    fragment_collisions += self.sim.collision_utils.count_1_to_n_collisions(collision_check_reference, collision_check_bodies)

            past_joint_pos.append(joint_pos)
            joint_state = self._p.getJointStates(self.sim.robot_id, [i for i in range(1,7)])
            joint_pos = list(zip(*joint_state))[0]
        if collision_check != "false":
            if collision_check == "count":
                return robot_collisions, fragment_collisions, plane_collisions
            else:
                return True if robot_collisions > 0 else False, True if fragment_collisions > 0 else False, True if plane_collisions > 0 else False
        else:
            return None, None, None

    def _move_to_xyz_yaw_position(self, action, link_id, debug, collision_check="false", frag_ids=[], collision_check_reference=-1):
        state = self._p.getLinkState(self.sim.robot_id, link_id, physicsClientId=self.sim.client_id)
        pos_action = np.multiply(action[0:3], self.sim.arm_action_position_increment_value) # Calculate x,y,z increment
        yaw_action = np.multiply(action[3], self.sim.arm_action_yaw_angle_increment_value) # Calculate yaw increment
        goalEndEffectorPos = np.asarray(state[0]) + [pos_action[0], pos_action[1], pos_action[2]] # Add x,y,z increment
        goalEndEffectorOri = np.array(self._p.getEulerFromQuaternion(state[1])) + np.array([0, 0, yaw_action]) # Add yaw increment
        goalEndEffectorOri =  np.array([-np.pi,0,goalEndEffectorOri[2]]) # Restrict TCP orientation to yaw

        if debug:
            goalEndEffectorPose = np.concatenate((goalEndEffectorPos, goalEndEffectorOri), dtype=np.float32)
            goalEndEffectorPose_4d = np.append(goalEndEffectorPose[:3], goalEndEffectorPose[5])
            del self.subgoal_debug_cross
            self.subgoal_debug_cross = debug_utils.DebugCross(config=self.config, sim=self.sim, position=goalEndEffectorPose_4d, line_length = 0.05, line_width = 10.0, line_color=[1,0,0], p_id=self._p, reset_all=False)

        target_joint_states = np.array(self.get_ik_joints(goalEndEffectorPos, goalEndEffectorOri, link_id, accuracy="medium"))
        self.send_position_command(target_joint_states=target_joint_states, position_gain=float(self.config["position_gain"]))
        robot_collisions, fragment_collisions, plane_collisions = self.carry_out_motion(target_joint_states, accuracy="medium", collision_check=collision_check, collision_check_bodies=frag_ids, collision_check_reference=collision_check_reference)

        return robot_collisions, fragment_collisions, plane_collisions

    def move_link_to_xyz_yaw_position(self, pose, link, debug, collision_check="false", frag_ids=[], collision_check_reference=-1):
        return self._move_to_xyz_yaw_position(pose, link, debug, collision_check=collision_check, frag_ids=frag_ids, collision_check_reference=collision_check_reference)

    def move_baseline_to_xyz_yaw_position(self, position, yaw, link_id, collision_check="false", frag_ids=[], collision_check_reference=-1, debug=False):
        '''
        Used in baseline approach to move to target position and yaw
        '''
        goalEndEffectorOri =  np.array([-np.pi,0,yaw]) # Restrict TCP orientation to yaw
        target_joint_states = np.array(self.get_ik_joints(position, goalEndEffectorOri, link_id, accuracy="medium"))
        self.send_position_command(target_joint_states=target_joint_states, position_gain=float(self.config["position_gain"]))
        robot_collisions, fragment_collisions, plane_collisions = self.carry_out_motion(target_joint_states, accuracy="medium", collision_check=collision_check, collision_check_bodies=frag_ids, collision_check_reference=collision_check_reference)
        
        return robot_collisions, fragment_collisions, plane_collisions