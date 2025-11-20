#!/usr/bin/python3

from fragment_gym.env.base_env import BasePybulletEnv
import pybullet as p
import numpy as np
import os
from stable_baselines3.common.env_util import make_vec_env
import math
from collections import deque
import json

# Utils
from fragment_gym.utils import collision_utils
from fragment_gym.utils import common_utils
from fragment_gym.utils import reward_utils
from fragment_gym.utils import rl_utils
from fragment_gym.utils import robot_utils
from fragment_gym.utils import manipulation_utils
from fragment_gym.utils import fresco_scaling_utils as scaling_utils
from utils import relative_placing_utils as relative_placing_utils
from fragment_gym.utils import shapely_utils as shapely_utils
from fragment_gym.utils import evaluation_utils as evaluation_utils

class RobotEnv(BasePybulletEnv):
    
    def __init__(self, mode=None, ablation=None, config=None, real_time=None, debug=None, render=False, shared_memory=False, use_egl=False, fresco_range=[]):
        super().__init__(mode=mode, ablation=ablation, config=config, real_time=real_time, debug=debug, render=render, shared_memory=shared_memory, use_egl=use_egl, fresco_range=fresco_range)

        self.robot_reach = float(self.config["robot_reach"])

        self.gripper_length_limit = (0.0, 0.085)
        self.gripper_angle_limit = (self.convert_gripper_length_to_angle(self.gripper_length_limit[0]), self.convert_gripper_length_to_angle(self.gripper_length_limit[1]))
        
        self.initial_parameters = (0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0., self.gripper_angle_limit[1])

        self.planeID = self._p.loadURDF("plane.urdf")

        # create robot entity in environment
        self.robot_constraints = []
        self.load_robot()

        # Instantiate utils
        # ==============================================================================
        self.collision_utils = collision_utils.Collision(config=self.config, sim=self, p_id=self._p)
        self.common_utils = common_utils.Common()
        self.rl_utils = rl_utils.RL(config=self.config)
        self.reward_utils = reward_utils.Reward(config=self.config, sim=self, p_id=self._p)
        self.robot_utils = robot_utils.RobotSetup(config=self.config)
        self.mu = manipulation_utils.Manipulation(config=self.config, sim=self, p_id=self._p)
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName = self.robot_utils.setup_sisbot(p, self.robot_id)

        # Baseline utils
        self.scaling_utils = scaling_utils.FrescoScalingUtils(config=self.config)
        self.relative_placing_utils = relative_placing_utils.RelativePlacingUtils(config=self.config, sim=self, p_id=self._p, frescoes_path=self.frescoes_path)
        self.evaluation_utils = evaluation_utils.EvaluationUtils(config=self.config, sim=self, p_id=self._p)
        self.shapely_utils = shapely_utils.ShapelyUtils(config=self.config)
        
        self.arm_joint_indices = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]

        self.initialize_gripper()
        self.eef_id = self.joints['ee_fixed_joint'][0]  # ee_link
        self.tool_tip_id = self.joints['tool0_fixed_joint-tool_tip'][0]  # ee_link
        self.camera_link = self.joints['dummy_camera_joint'][0]

        # set robot to intial position
        self.reset_robot(self.initial_parameters)
        self.init_pos = list(self._p.getLinkState(self.robot_id, self.tool_tip_id)[0])
        self.init_ori = np.array(self._p.getEulerFromQuaternion(self._p.getLinkState(self.robot_id, self.tool_tip_id)[1])) 


    def initialize_gripper(self):
        # Add force sensors
        self._p.enableJointForceTorqueSensor(
            self.robot_id, self.joints['left_inner_finger_pad_joint'].id)
        self._p.enableJointForceTorqueSensor(
            self.robot_id, self.joints['right_inner_finger_pad_joint'].id)

        # Change the friction of the gripper
        self._p.changeDynamics(
            self.robot_id, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=1, physicsClientId=self.client_id)
        self._p.changeDynamics(
            self.robot_id, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=1, physicsClientId=self.client_id)

        self.gid_left = self._p.createConstraint(self.robot_id, self.joints['left_inner_knuckle_joint'].id, self.robot_id, self.joints['left_inner_finger_joint'].id,
                           p.JOINT_POINT2POINT, [0, 0, 0], [0, -0.014, 0.043], [0, -0.034, 0.021], physicsClientId=self.client_id)
        self.robot_constraints.append(self.gid_left)
        self._p.changeConstraint(self.gid_left, physicsClientId=self.client_id)

        self.gid_right = self._p.createConstraint(self.robot_id, self.joints['right_inner_knuckle_joint'].id, self.robot_id, self.joints['right_inner_finger_joint'].id,
                           p.JOINT_POINT2POINT, [0, 0, 0], [0, -0.014, 0.043], [0, -0.034, 0.021], physicsClientId=self.client_id)
        self.robot_constraints.append(self.gid_right)
        self._p.changeConstraint(self.gid_right, physicsClientId=self.client_id)
        
    def move_gripper(self, gripper_opening_length):
        gripper_opening_angle = self.convert_gripper_length_to_angle(gripper_opening_length)
        self.controlGripper(controlMode=self._p.POSITION_CONTROL,targetPosition=gripper_opening_angle)
        return

    def load_robot(self):
        self.robot_id = self._p.loadURDF(self.root_path+'meshes/urdf/ur5_robotiq_85_new.urdf' ,[0, 0, 0],
                                         self._p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

    def get_current_tcp(self):
        return self._p.getLinkState(self.robot_id, self.tool_tip_id, physicsClientId=self.client_id)

    def get_current_tcp_yaw_in_euler(self):
        tcp_state = self.get_current_tcp()   
        tcp_pos = tcp_state[0]
        tcp_orn = tcp_state[1]
        tcp_rpy = self._p.getEulerFromQuaternion(tcp_orn)
        tcp_yaw = tcp_rpy[2]
        return tcp_yaw

    def get_current_joint_config(self):
        return [self._p.getJointState(self.robot_id, i)[0] for i in range(1, 7)]

    def convert_gripper_length_to_angle(self, gripper_length):
        gripper_angle = 0.715 - math.asin((gripper_length - 0.010) / 0.1143)
        return gripper_angle
    
    def convert_gripper_angle_to_length(self, gripper_angle):
        gripper_length = 0.01 + (0.1143 * math.sin(0.715 - gripper_angle))
        return gripper_length

    def get_current_gripper_opening_angle(self):
        return self._p.getJointState(self.robot_id, self.joints[self.mimicParentName].id)[0]
    
    def get_current_gripper_opening(self):
        gripper_opening_angle = self.get_current_gripper_opening_angle()
        gripper_opening_length = self.convert_gripper_angle_to_length(gripper_opening_angle)
        return gripper_opening_length

    def reset_robot(self, parameters):
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            if i == 6:
                targetPosition = parameters[i]
                self.controlGripper(controlMode=self._p.POSITION_CONTROL, targetPosition=targetPosition)
                continue
            self._p.resetJointState(self.robot_id, joint.id, parameters[i])
            self._p.setJointMotorControl2(self.robot_id, joint.id, self._p.POSITION_CONTROL,
                                          targetPosition=parameters[i], force=joint.maxForce,
                                          maxVelocity=joint.maxVelocity)
        self.step_simulation(self.per_step_iterations)

    def load_urdf_fragment(self, fresco_no, fragment_no, translation, orientation):
        fragment_path = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) +"/fragment_"+ str(fragment_no) +".urdf"
        frag_id = self._p.loadURDF(fragment_path)
        color = np.array([165, 113, 78, 255])/255
        self._p.resetBasePositionAndOrientation(frag_id, translation, self._p.getQuaternionFromEuler(orientation))
        self._p.changeVisualShape(frag_id, -1, rgbaColor=color)

        return frag_id

    def control_gripper(self, close: bool = False, release: bool = False, release_obj_id: int = None, coll_check: str = "false", coll_check_ids: list = None, coll_check_ref_id: int = -1):
        release_dist = float(self.config["gripper_release_distance"])
        open_config = float(self.config["gripper_open"])
        close_config = float(self.config["gripper_closed"])
        
        # Release fragment
        if release:
            pos_delta = float(self.config["gripper_position_delta_high_precision"])
            queue_size = int(self.config["gripper_queue_size_high_precision"])
        # Close to grasp fragment
        elif close:
            pos_delta = float(self.config["gripper_position_delta_high_precision"])
            queue_size = int(self.config["gripper_queue_size_low_precision"])
        # Standard open without fragment
        else:
            pos_delta = float(self.config["gripper_position_delta_low_precision"])
            queue_size = int(self.config["gripper_queue_size_low_precision"])
        gripper_target_threshold = float(self.config["gripper_target_threshold"])
        curr_config = self.get_current_gripper_opening()

        target_config = close_config if close else open_config
        direction = -pos_delta if close else pos_delta

        config_queue = deque(maxlen=queue_size)

        if release:
            start_config = self.get_current_gripper_opening()
            target_config = start_config + release_dist

        if coll_check != "false":
            plane_collisions = 0
            robot_collisions = 0
            fragment_collisions = 0

        while True:
            new_config = curr_config + direction
            self.move_gripper(gripper_opening_length=new_config)
            self.step_simulation(self.per_step_iterations)
            curr_config = self.get_current_gripper_opening()
            config_queue.append(curr_config)

            # check for collisions
            if coll_check != "false":
                plane_collisions += self.collision_utils.count_robot_collisions([self.planeID])
                robot_collisions += self.collision_utils.count_robot_collisions(coll_check_ids)
                if coll_check_ref_id < 0:
                    fragment_collisions += self.collision_utils.count_n_to_n_collisions(coll_check_ids)
                else:
                    fragment_collisions += self.collision_utils.count_1_to_n_collisions(coll_check_ref_id, coll_check_ids)

            # break when gripper can no longer close when grasping
            if len(config_queue) == queue_size and np.std(config_queue) < pos_delta:
                    if coll_check != "false":
                        if coll_check == "count":
                            return robot_collisions, fragment_collisions, plane_collisions
                        else:
                            return True if robot_collisions > 0 else False, True if fragment_collisions > 0 else False, True if plane_collisions > 0 else False
                    else:
                        return 0

            # break when gripper reaches target position
            if abs(curr_config - target_config) <= gripper_target_threshold:
                    if coll_check != "false":
                        if coll_check == "count":
                            return robot_collisions, fragment_collisions, plane_collisions
                        else:
                            return True if robot_collisions > 0 else False, True if fragment_collisions > 0 else False, True if plane_collisions > 0 else False
                    else:
                        return 0

    def load_stl_fragment(self, fresco_no, fragment_no, fragment_translation, orientation):
        fragment_vis_mesh = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) +"/fragment_"+ str(fragment_no) +".stl"
        fragment_col_mesh = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) +"/fragment_"+ str(fragment_no) +"_vhacd.obj"

        gt_data = self.shapely_utils.get_fresco_ground_truth(fresco_no=fresco_no, root_path=self.root_path)
        fragment_mass = gt_data["masses"][fragment_no]
        
        col_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=fragment_col_mesh,
            meshScale=[1.0, 1.0, 1.0]
        )

        color = np.array([165, 113, 78, 255])/255
        viz_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=fragment_vis_mesh,
            rgbaColor=color
        )

        body_id = p.createMultiBody(
            baseMass = fragment_mass,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=viz_shape_id,
            basePosition=fragment_translation,
            baseOrientation=orientation
        )

        return body_id
    
    def load_fragment(self, fresco_no, fragment_no, fragment_translation, orientation, visual_type="stl", collision_type="vhacd", collision_mesh_scale=1.0):
        if visual_type == "stl":
            fragment_vis_mesh = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) +"/fragment_"+ str(fragment_no) +".stl"
        elif visual_type == "obj":
            fragment_vis_mesh = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) +"/fragment_"+ str(fragment_no) +".obj"
        elif visual_type == "vhacd":
            fragment_vis_mesh = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) +"/fragment_"+ str(fragment_no) +"_vhacd.obj"
        else:
            print("Unknown visual mesh type")
            raise ValueError
        
        if collision_type == "stl":
            fragment_col_mesh = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) +"/fragment_"+ str(fragment_no) +".stl"
        elif collision_type == "obj":
            fragment_col_mesh = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) +"/fragment_"+ str(fragment_no) +".obj"
        elif collision_type == "vhacd":
            fragment_col_mesh = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) +"/fragment_"+ str(fragment_no) +"_vhacd.obj"
        else:
            print("Unknown visual mesh type")
            raise ValueError

        gt_data = self.shapely_utils.get_fresco_ground_truth(fresco_no=fresco_no, root_path=self.root_path)
        fragment_mass = gt_data["masses"][fragment_no]
        
        col_shape_id = self._p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=fragment_col_mesh,
            meshScale=[collision_mesh_scale, collision_mesh_scale, collision_mesh_scale]
        )

        color = np.array([165, 113, 78, 255])/255
        viz_shape_id = self._p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=fragment_vis_mesh,
            rgbaColor=color
        )

        body_id = self._p.createMultiBody(
            baseMass = 0,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=viz_shape_id,
            basePosition=fragment_translation,
            baseOrientation=orientation
        )

        self._p.changeDynamics(
            bodyUniqueId = body_id,
            linkIndex = -1,
            collisionMargin = 0.0,
            physicsClientId=self.client_id
        )

        return body_id

    def get_fragments_at_scale_factor(self, fresco_no, assembly_plan_type="assembly_plan_snake", scale_factor=0.0):
        gt_data = self.shapely_utils.get_fresco_ground_truth(fresco_no=fresco_no, root_path=self.root_path)
        assembly_plan = gt_data["assembly_plan"][assembly_plan_type]
        frag_height = gt_data["header"]["height"]
        fresco_centroids = gt_data["centroids"]

        assembly_id = -1
        frag_info_sorted = []
        for frag_id in assembly_plan:
            assembly_id += 1
            yaw = assembly_plan[frag_id]["yaw"]
            yaw = np.deg2rad(yaw)

            frag_centroid = np.array(fresco_centroids[int(frag_id)])
            frag_centroid = np.append(frag_centroid, frag_height/2) # add z coordinate

            frag_info_sorted.append([int(frag_id), assembly_id, frag_centroid, yaw])

            pass

        return frag_info_sorted

    def get_fragments_at_scale_factor_from_baseline_evaluation(self, fresco_no, assembly_plan_type="assembly_plan_snake", scale_factor=0.0):
        gt_data = self.shapely_utils.get_fresco_ground_truth(fresco_no=fresco_no, root_path=self.root_path)

        # Get the highest scale factor in gt_data
        if scale_factor == -1:
            scale_factor = gt_data["header"]["scale_factor"][assembly_plan_type]

        scale_factor = str(scale_factor)
        assembly_plan = gt_data["assembly_plan"][assembly_plan_type]
        frag_height = gt_data["header"]["height"]

        assembly_id = -1
        frag_info_sorted = []
        for frag_id in assembly_plan:
            assembly_id += 1
            yaw = assembly_plan[frag_id]["yaw"]
            yaw = np.deg2rad(yaw)

            if "baseline_scaling" in gt_data:
                frag_centroid = np.array(gt_data["baseline_scaling"][assembly_plan_type]["sf_"+scale_factor]["centroids"][int(frag_id)])
            else:
                frag_centroid = np.array(gt_data["centroids"][int(frag_id)])

            # add z coordinate
            frag_centroid = np.append(frag_centroid, frag_height/2)

            frag_info_sorted.append([int(frag_id), assembly_id, frag_centroid, yaw])

        return frag_info_sorted

    def spawn_stl_fragment_and_pick(self, fresco_no, fragment_no, yaw=-1, pick_from_plane=True):
        current_tool_tip_state = self._p.getLinkState(self.robot_id, self.tool_tip_id, physicsClientId=self.client_id)
        current_tool_tip_translation = current_tool_tip_state[0]

        if yaw == -1.0:
            yaw = np.random.uniform(0, 2*np.pi)


        if pick_from_plane:

            plane_pos = list(current_tool_tip_translation)
            plane_pos[2] -= 0.005

            # create a small plane to place the fragment
            self.pick_plane_id = self._p.loadURDF(
                self.root_path + "meshes/urdf/pick_plane.urdf",
                plane_pos,
                self._p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
                flags=self._p.URDF_USE_INERTIA_FROM_FILE,
            )

            fragment_translation = list(current_tool_tip_translation)
            fragment_translation[2] += 0.03

            # convert euler to quaternion
            orientation = self._p.getQuaternionFromEuler([0, 0, yaw])

            frag_id = self.load_stl_fragment(fresco_no, fragment_no, fragment_translation, orientation)

            self.control_gripper(close=True)

            orientation = self._p.getQuaternionFromEuler([np.pi, 0, -np.pi/2 - yaw])

            # get fragment translation and orientation
            frag_temp = self._p.getBasePositionAndOrientation(frag_id, physicsClientId=self.client_id)
            frag_translation = frag_temp[0]
            frag_orientation_quaternion = frag_temp[1]

            # get the relative position of the fragment to the tool tip
            relative_pos, relative_orn = self._p.multiplyTransforms(
                current_tool_tip_translation,
                orientation,
                frag_translation,
                frag_orientation_quaternion
            )

        else:
            relative_pos = [0, 0, 0.0]
            orientation = self._p.getQuaternionFromEuler([np.pi, 0, -np.pi/2 - yaw])
            fragment_translation = list(current_tool_tip_translation)
            frag_id = self.load_stl_fragment(fresco_no, fragment_no, fragment_translation, yaw)

        # create constraint to fixate fragment to tool tip
        cid = self._p.createConstraint(
            self.robot_id, self.tool_tip_id, frag_id, -1, p.JOINT_FIXED, [0, 0, 0],
            parentFramePosition=[0, 0, 0], parentFrameOrientation=[0, 0, 0], 
            childFramePosition=[0,0,relative_pos[2]], childFrameOrientation=orientation, physicsClientId=self.client_id)
        
        # get constraint info
        self.step_simulation(self.per_step_iterations)

        if not pick_from_plane:
            self.control_gripper(close=True)
        else:
            self._p.removeBody(self.pick_plane_id)

        return frag_id, cid

    def spawn_stl_fragments_on_table_and_in_gripper(self, fresco_no, fresco_center_location=[0.45,0.0,0.0], amount_of_fragments_on_table=-1, scale_factor=-1):
        frag_ids = []

        fresco_path = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) + "/"
        if os.path.isfile(fresco_path + "ground_truth_modified.json"):
            gt_data = json.load(
                open(fresco_path + "ground_truth_modified.json")
            )
        else:
            gt_data = json.load(open(fresco_path + "ground_truth.json"))

        no_fragments = gt_data["header"]["no_fragments"]
        frag_height = gt_data["header"]["height"]

        fresco_center_location[2] = frag_height/2
        frag_gen = self.get_fragments_at_scale_factor(fresco_no=fresco_no, fresco_center=fresco_center_location, scale_factor=scale_factor)

        if amount_of_fragments_on_table == -1:
            amount_of_fragments_on_table = np.random.randint(0, no_fragments-2)

        # Spawn the fragment on the table
        count = 0
        for _, assembly_no, frag_centroid, _ in frag_gen:
            if count == amount_of_fragments_on_table:
                frag_id, fix_fragment_to_tool_tip_constraint = self.spawn_stl_fragment_and_pick(fresco_no, assembly_no)
                frag_ids.append(frag_id)
                target_pose = np.asarray(frag_centroid)
                target_pose = np.append(target_pose, 0.0)
                break
            else:
                frag_id = self.load_stl_fragment(fresco_no, assembly_no, tuple(frag_centroid), orientation=self._p.getQuaternionFromEuler([0,0,0]))
                frag_ids.append(frag_id)
            count += 1
        
        return frag_ids, fix_fragment_to_tool_tip_constraint, target_pose
    
    def spawn_stl_fresco_on_table(self, fresco_no, fresco_center_location, scale_factor=-1):
        frag_info_sorted = self.get_fragments_at_scale_factor(fresco_no=fresco_no, scale_factor=scale_factor)

        frag_ids_sorted = []
        # Spawn the fragment on the table
        for frag_no, assembly_no, frag_centroid, yaw in frag_info_sorted:
            translated_frag_centroid = np.asarray(frag_centroid) + np.asarray(fresco_center_location)
            frag_pybullet_id = self.load_stl_fragment(fresco_no, frag_no, tuple(translated_frag_centroid), orientation=self._p.getQuaternionFromEuler([0,0,0]))
            frag_ids_sorted.append([frag_pybullet_id, frag_no, assembly_no, frag_centroid, yaw])

        return frag_ids_sorted # frag_pybullet_id, frag_id, assembly_no, centroid, yaw
    
    def reset_fresco(self, frag_ids_sorted, fresco_center_location):
        for frag_pybullet_id, frag_id, assembly_no, centroid, yaw in frag_ids_sorted:
            translated_frag_centroid = np.asarray(centroid) + np.asarray(fresco_center_location)
            self._p.resetBasePositionAndOrientation(frag_pybullet_id, posObj=tuple(translated_frag_centroid), ornObj=self._p.getQuaternionFromEuler([0,0,0]))
            self.remove_mass_from_fragment(frag_pybullet_id)
        self.step_simulation(self.per_step_iterations)


    
    def spawn_fresco_on_table(self, fresco_no, fresco_center_location, visual_type, collision_type, collision_mesh_scale=1.0, scale_factor=0.0):
        frag_info_sorted = self.get_fragments_at_scale_factor(fresco_no=fresco_no, scale_factor=scale_factor)
        frag_ids_sorted = []
        # Spawn the fragment on the table
        for frag_no, assembly_no, frag_centroid, yaw in frag_info_sorted:
            translated_frag_centroid = np.asarray(frag_centroid) + np.asarray(fresco_center_location)
            frag_pybullet_id = self.load_fragment(fresco_no, frag_no, tuple(translated_frag_centroid), orientation=self._p.getQuaternionFromEuler([0,0,0]), visual_type=visual_type, collision_type=collision_type, collision_mesh_scale=collision_mesh_scale)
            frag_ids_sorted.append([frag_pybullet_id, frag_no, assembly_no, frag_centroid, yaw])

        return frag_ids_sorted

    def move_fragments_on_table_and_in_gripper(self, frag_ids_sorted, fresco_center_location=[0.45,0.0], pick=True, use_curriculum_learning=False, current_curriculum_step=0, place_complete_fresco=False, amount_of_fragments_on_table=-1):

        if amount_of_fragments_on_table == -1:
            amount_of_fragments_on_table = np.random.randint(0, len(frag_ids_sorted)-2)
       
        count = 0
        # Move the fragment on the table
        for frag_pybulllet_no, frag_no, assembly_no, frag_centroid, grasp_yaw in frag_ids_sorted:
            temp_fresco_center_location = np.asarray([*fresco_center_location, 0.0], dtype=np.float32)
            translated_frag_centroid = np.asarray(frag_centroid) + np.asarray(temp_fresco_center_location, dtype=np.float32)
            if count == amount_of_fragments_on_table and pick:
                target_pose = np.asarray(translated_frag_centroid, dtype=np.float32)
                target_pose = np.append(target_pose, 0.0) # Add placement yaw
                # Curriculum learning
                if use_curriculum_learning:
                    current_curriculum_retract_height = self.config.get("initial_curriculum_height", self.config["gripper_retract_height"])    
                    
                    # Curriculum robot arm reset configuration
                    self.current_ee_pose = self.get_tool_tip_pose()
                    current_gripper_opening_angle = self.get_current_gripper_opening_angle()

                    curriculum_initial_pose = np.asarray(target_pose, dtype=np.float32).copy()[:3]
                    curriculum_initial_pose[2] += current_curriculum_retract_height

                    curriculum_final_pose = np.asarray(temp_fresco_center_location, dtype=np.float32).copy()
                    curriculum_final_pose[2] += self.config.get("final_curriculum_height", self.robot_reach/4)
                    
                    curriculum_poses = np.linspace(curriculum_initial_pose, curriculum_final_pose, int(self.config["curriculum_steps"]), dtype=np.float32)
                    current_curriculum_pose = curriculum_poses[current_curriculum_step]
                    
                    curriculum_robot_joint_angles = self.mu.get_ik_joints(position=current_curriculum_pose, orientation=[-np.pi,0,self.current_ee_pose[3]], link=self.tool_tip_id, accuracy="medium")

                    curriculum_joint_angles = (*curriculum_robot_joint_angles, current_gripper_opening_angle)
                    self.reset_robot(curriculum_joint_angles)
                else:
                    current_curriculum_retract_height = float(self.config["gripper_retract_height"])
                
                if self.mode == "eval":
                    fix_fragment_to_tool_tip_constraint = self.move_fragment_and_pick(fragment_pybullet_id=frag_pybulllet_no, fragment_id=frag_no, grasp_yaw=grasp_yaw, fragment_spawn_yaw=0.0) # for evaluation
                else:
                    grasp_yaw = float(self.config.get("grasp_yaw", -1.0))
                    if grasp_yaw != -1.0:
                        grasp_yaw = np.deg2rad(grasp_yaw)
                    fragment_spawn_yaw = float(self.config.get("fragment_spawn_yaw", -1.0))
                    if fragment_spawn_yaw != -1.0:
                        fragment_spawn_yaw = np.deg2rad(fragment_spawn_yaw)
                    fix_fragment_to_tool_tip_constraint = self.move_fragment_and_pick(fragment_pybullet_id=frag_pybulllet_no, fragment_id=frag_no, grasp_yaw=grasp_yaw, fragment_spawn_yaw=fragment_spawn_yaw) # for training and testing
                
                used_pybullet_frag_ids_sorted = [row[0] for row in frag_ids_sorted[0:count]]
                used_frag_ids_sorted = [row[1] for row in frag_ids_sorted[0:count]]
                return used_pybullet_frag_ids_sorted, used_frag_ids_sorted, frag_pybulllet_no, frag_no, assembly_no, fix_fragment_to_tool_tip_constraint, target_pose, current_curriculum_retract_height
            else:
                if place_complete_fresco == False:
                    self.add_mass_to_fragment(frag_pybulllet_no, frag_no)
                    self._p.resetBasePositionAndOrientation(frag_pybulllet_no, posObj=tuple(translated_frag_centroid), ornObj=self._p.getQuaternionFromEuler([0,0,0]))
                count += 1
                
        used_pybullet_frag_ids_sorted = [row[0] for row in frag_ids_sorted[0:count]]

        return used_pybullet_frag_ids_sorted
    

    def move_fragments_on_table_and_in_gripper_without_retraction(self, frag_ids_sorted, fresco_center_location=[0.45,0.0], pick=True, use_curriculum_learning=False, current_curriculum_step=0, place_complete_fresco=False, amount_of_fragments_on_table=-1):
        if amount_of_fragments_on_table == -1:
            amount_of_fragments_on_table = np.random.randint(0, len(frag_ids_sorted)-2)
       
        count = 0
        # Move the fragment on the table
        for frag_pybulllet_no, frag_no, assembly_no, frag_centroid, grasp_yaw in frag_ids_sorted:
            temp_fresco_center_location = np.asarray([*fresco_center_location, 0.0], dtype=np.float32)
            translated_frag_centroid = np.asarray(frag_centroid) + np.asarray(temp_fresco_center_location, dtype=np.float32)
            if count == amount_of_fragments_on_table and pick:
                target_pose = np.asarray(translated_frag_centroid, dtype=np.float32)
                target_pose = np.append(target_pose, 0.0) # Add placement yaw
                
                # Curriculum learning
                self.current_ee_pose = self.get_tool_tip_pose()
                current_gripper_opening_angle = self.get_current_gripper_opening_angle()
                curriculum_start_type = self.config.get("curriculum_start_type", "fragment_center")
                curriculum_end_type = self.config.get("curriculum_end_type", "fresco_center")
                if use_curriculum_learning:              
                    # Curriculum robot arm reset configuration
                    if curriculum_start_type == "fragment_center":
                        curriculum_initial_pose = np.asarray(target_pose, dtype=np.float32).copy()[:3]
                    elif curriculum_start_type == "fresco_center":
                        curriculum_initial_pose = np.asarray(temp_fresco_center_location, dtype=np.float32).copy()[:3]
                    curriculum_initial_pose[2] += self.config.get("initial_curriculum_height", 0.05)

                if curriculum_end_type == "fragment_center":
                    curriculum_final_pose = np.asarray(target_pose, dtype=np.float32).copy()[:3]
                elif curriculum_end_type == "fresco_center":
                    curriculum_final_pose = np.asarray(temp_fresco_center_location, dtype=np.float32).copy()[:3]
                curriculum_final_pose[2] += self.config.get("final_curriculum_height", self.robot_reach/4)
                
                if use_curriculum_learning:
                    curriculum_poses = np.linspace(curriculum_initial_pose, curriculum_final_pose, int(self.config["curriculum_steps"]), dtype=np.float32)
                    current_curriculum_pose = curriculum_poses[current_curriculum_step]
                else:
                    current_curriculum_pose = curriculum_final_pose
                curriculum_robot_joint_angles = self.mu.get_ik_joints(position=current_curriculum_pose, orientation=[-np.pi,0,self.current_ee_pose[3]], link=self.tool_tip_id, accuracy="medium")

                curriculum_joint_angles = (*curriculum_robot_joint_angles, current_gripper_opening_angle)
                #curriculum_joint_angles = np.array([*curriculum_robot_joint_angles, current_gripper_opening_angle], dtype=np.float32)
                self.reset_robot(curriculum_joint_angles)
                
                if self.mode == "eval": 
                    fix_fragment_to_tool_tip_constraint = self.move_fragment_and_pick(fragment_pybullet_id=frag_pybulllet_no, fragment_id=frag_no, grasp_yaw=grasp_yaw, fragment_spawn_yaw=0.0) # for evaluation
                else:
                    grasp_yaw = float(self.config.get("grasp_yaw", -1.0))
                    if grasp_yaw != -1.0:
                        grasp_yaw = np.deg2rad(grasp_yaw)
                    fragment_spawn_yaw = float(self.config.get("fragment_spawn_yaw", -1.0))
                    if fragment_spawn_yaw != -1.0:
                        fragment_spawn_yaw = np.deg2rad(fragment_spawn_yaw)
                    fix_fragment_to_tool_tip_constraint = self.move_fragment_and_pick(fragment_pybullet_id=frag_pybulllet_no, fragment_id=frag_no, grasp_yaw=grasp_yaw, fragment_spawn_yaw=fragment_spawn_yaw) # for training and testing

                used_pybullet_frag_ids_sorted = [row[0] for row in frag_ids_sorted[0:count]]
                used_frag_ids_sorted = [row[1] for row in frag_ids_sorted[0:count]]
                return used_pybullet_frag_ids_sorted, used_frag_ids_sorted, frag_pybulllet_no, frag_no, assembly_no, fix_fragment_to_tool_tip_constraint, target_pose
            else:
                if place_complete_fresco == False:
                    self.add_mass_to_fragment(frag_pybulllet_no, frag_no)
                    self._p.resetBasePositionAndOrientation(frag_pybulllet_no, posObj=tuple(translated_frag_centroid), ornObj=self._p.getQuaternionFromEuler([0,0,0]))
                count += 1
                
        used_pybullet_frag_ids_sorted = [row[0] for row in frag_ids_sorted[0:count]]

        return used_pybullet_frag_ids_sorted

    def add_mass_to_fragment(self, fragment_pybullet_id, fragment_id):
        # Add mass to the fragment
        fragment_mass = self.gt_data["masses"][fragment_id]
        self._p.changeDynamics(
            bodyUniqueId = fragment_pybullet_id,
            linkIndex = -1,
            mass = fragment_mass,
            collisionMargin = 0.0,
            physicsClientId=self.client_id
        )

    def remove_mass_from_fragment(self, fragment_pybullet_id_list):
        if type(fragment_pybullet_id_list) == int:
            fragment_pybullet_id_list = [fragment_pybullet_id_list]
        # Remove mass to the fragment
        for pybullet_id in fragment_pybullet_id_list:
            self._p.changeDynamics(
                bodyUniqueId = pybullet_id,
                linkIndex = -1,
                mass = 0.0,
                collisionMargin = 0.0,
                physicsClientId=self.client_id
            )

    def move_fragment_and_pick(self, fragment_pybullet_id, fragment_id, grasp_yaw=-1.0, fragment_spawn_yaw=-1.0, pick_from_plane=True):
        if grasp_yaw == -1.0:
            grasp_yaw = np.random.uniform(-np.pi/2, np.pi/2)

        if fragment_spawn_yaw == -1.0:
            fragment_spawn_yaw = np.random.uniform(-np.pi/2, np.pi/2)

        self.control_gripper()

        tcp_pose = self.get_tool_tip_pose()

        self.add_mass_to_fragment(fragment_pybullet_id, fragment_id)
        
        if pick_from_plane:

            plane_pos = tcp_pose[:3]
            plane_pos[2] -= 0.005

            self._p.resetBasePositionAndOrientation(self.pick_plane_id, posObj=tuple(plane_pos), ornObj=self._p.getQuaternionFromEuler([0, 0, 0]))


            fragment_translation = plane_pos
            fragment_translation[2] += 0.01

            self._p.resetBasePositionAndOrientation(fragment_pybullet_id, posObj=tuple(fragment_translation), ornObj=self._p.getQuaternionFromEuler([0, 0, fragment_spawn_yaw]))
            self.step_simulation(self.per_step_iterations)

            self.mu.move_baseline_to_xyz_yaw_position(tcp_pose[:3], grasp_yaw, self.tool_tip_id)
            self.control_gripper(close=True)
            self.step_simulation(self.per_step_iterations)

        else:
            self.mu.move_baseline_to_xyz_yaw_position(tcp_pose[:3], 0.0, self.tool_tip_id)
            self.step_simulation(self.per_step_iterations)

            relative_pos = [0.0, 0.0, 0.0]
            relative_orn = self._p.getQuaternionFromEuler([np.pi, 0, grasp_yaw])
            fragment_translation = tcp_pose[:3]
            self._p.resetBasePositionAndOrientation(fragment_pybullet_id, posObj=tuple(fragment_translation), ornObj=self._p.getQuaternionFromEuler([0, 0, grasp_yaw]))

        # create constraint to fixate fragment to tool tip
        # get fragment translation and orientation
        frag_temp = self._p.getBasePositionAndOrientation(fragment_pybullet_id, physicsClientId=self.client_id)
        frag_translation = frag_temp[0]
        frag_orientation_quaternion = frag_temp[1]

        # get tcp translation and orientation
        tcp_temp = self._p.getLinkState(self.robot_id, self.tool_tip_id, physicsClientId=self.client_id)
        tcp_translation = tcp_temp[0]
        tcp_orientation_quaternion = tcp_temp[1]

        tcp_translation_inverse, tcp_orientation_quaternion_inverse = self._p.invertTransform(tcp_translation, tcp_orientation_quaternion)

        # get the relative position of the fragment to the tool tip
        relative_pos, relative_orn = self._p.multiplyTransforms(
            tcp_translation_inverse,
            tcp_orientation_quaternion_inverse,
            frag_translation,
            frag_orientation_quaternion
        )

        cid = self._p.createConstraint(
            self.robot_id, self.tool_tip_id, fragment_pybullet_id, -1, p.JOINT_FIXED, [0, 0, 0],
            parentFramePosition=[0, 0, 0], parentFrameOrientation=self._p.getQuaternionFromEuler([0, 0, 0]), 
            childFramePosition=relative_pos, childFrameOrientation=relative_orn, physicsClientId=self.client_id)
        
        # get constraint info
        self.step_simulation(self.per_step_iterations)

        if not pick_from_plane:
            self.control_gripper(close=True)
        else:
            self._p.resetBasePositionAndOrientation(self.pick_plane_id, posObj=tuple(self.pick_plane_parking_center), ornObj=self._p.getQuaternionFromEuler([0, 0, 0]))
        self.step_simulation(self.per_step_iterations)

        return cid

    def spawn_urdf_fragment_in_gripper(self, fresco_no, fragment_no, yaw=-1.0):
        current_tool_tip_state = self._p.getLinkState(self.robot_id, self.tool_tip_id, physicsClientId=self.client_id)
        current_tool_tip_translation = current_tool_tip_state[0]

        z_offset = 0.010
        adjusted_tool_tip_translation = list(current_tool_tip_translation)
        adjusted_tool_tip_translation[2] = current_tool_tip_translation[2] + z_offset
        adjusted_tool_tip_translation = tuple(adjusted_tool_tip_translation)

        if yaw == -1.0:
            yaw = np.random.uniform(0, 2*np.pi)
        orientation = self._p.getQuaternionFromEuler([np.pi, 0, -np.pi/2 - yaw])

        # Spawn the fragment into the scene
        frag_id = self.load_urdf_fragment(fresco_no, fragment_no, adjusted_tool_tip_translation, [0, 0, yaw])

        # Fixate the fragment to the tool tip
        # The translation and orientation is adjusted so that it matches the spawned own
        fix_fragment_to_tool_tip_constraint = self._p.createConstraint(self.robot_id, self.tool_tip_id, frag_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], childFramePosition=[0,0,-z_offset], childFrameOrientation=orientation, physicsClientId=self.client_id) #JOINT_POINT2POINT JOINT_FIXED
        self._p.changeConstraint(fix_fragment_to_tool_tip_constraint, physicsClientId=self.client_id)

        return frag_id, fix_fragment_to_tool_tip_constraint
    
    def spawn_urdf_fragments_on_table_and_in_gripper(self, fresco_no, fresco_center_location=[0.45,0.0], amount_of_fragments_on_table=-1, scale_factor=-1):
        frag_ids = []

        fresco_path = self.root_path + "meshes/fragments/fresco_"+ str(fresco_no) + "/"
        if os.path.isfile(fresco_path + "ground_truth_modified.json"):
            gt_data = json.load(
                open(fresco_path + "ground_truth_modified.json")
            )
        else:
            gt_data = json.load(open(fresco_path + "ground_truth.json"))
        
        no_fragments = gt_data["header"]["no_fragments"]
        frag_height = gt_data["header"]["height"]
        assembly_plan = gt_data["assembly_plan"]

        # Get the highest scale factor in gt_data
        if scale_factor == -1:
            scale_factor = max(list(map(float,(list(gt_data["scale_factors"].keys())))))
        scale_factor = str(scale_factor)

        if amount_of_fragments_on_table == -1:
            amount_of_fragments_on_table = np.random.randint(0, no_fragments-2)

        # Spawn the fragment on the table
        for frag_no in range(0,amount_of_fragments_on_table+1):
            assembly_no = int(list(assembly_plan.keys())[frag_no])
            frag_centroid = gt_data["scale_factors"][scale_factor]["centroids"][assembly_no]
            frag_placement_location = np.add(np.asarray(fresco_center_location), np.asarray(frag_centroid))
            frag_placement_location = np.append(frag_placement_location, frag_height/2)
            
            if frag_no == amount_of_fragments_on_table:
                frag_id, fix_fragment_to_tool_tip_constraint = self.spawn_urdf_fragment_in_gripper(fresco_no,amount_of_fragments_on_table)
                target_pose = np.append(frag_placement_location, 0.0)
            else:
                frag_id = self.load_urdf_fragment(fresco_no, frag_no, tuple(frag_placement_location.tolist()), [0, 0, 0])

            frag_ids.append(frag_id)
        
        return frag_ids, fix_fragment_to_tool_tip_constraint, target_pose

    def get_fragment_pose(self, frag_id):
        frag_temp = self._p.getBasePositionAndOrientation(frag_id, physicsClientId=self.client_id)
        frag_position = frag_temp[0]
        frag_orientation_quaternion = frag_temp[1]
        frag_orientation_euler = self._p.getEulerFromQuaternion(frag_orientation_quaternion)
        frag_yaw = frag_orientation_euler[2]
        fragment_pose = np.array([*frag_position, frag_yaw], dtype=np.float32)
        return fragment_pose
    
    def get_7d_fragment_pose(self, frag_id):
        return self._p.getBasePositionAndOrientation(frag_id, physicsClientId=self.client_id)
    
    def get_all_fragment_poses(self, frag_ids):
        fragment_poses = []
        for frag in frag_ids:
            fragment_poses.append(self.get_fragment_pose(frag))
        return fragment_poses
    
    def get_tool_tip_pose(self):
        temp_pose = self._p.getLinkState(self.robot_id, self.tool_tip_id, physicsClientId=self.client_id)
        position = temp_pose[0]
        orientation_quaternion = temp_pose[1]
        orientation_euler = self._p.getEulerFromQuaternion(orientation_quaternion)
        yaw = orientation_euler[2]
        pose = np.array([*position, yaw], dtype=np.float32)
        return pose
    
    def get_hand_cam_pose(self):
        temp_pose = self._p.getLinkState(self.robot_id, self.camera_link, physicsClientId=self.client_id)
        position = temp_pose[0]
        orientation_quaternion = temp_pose[1]
        orientation_euler = self._p.getEulerFromQuaternion(orientation_quaternion)
        yaw = orientation_euler[2]
        pose = np.array([*position, yaw], dtype=np.float32)
        return pose

def load_default_task():
    env = make_vec_env('RobotiqEnv-v0', n_envs=1)
    return env.envs[0].env.env

def test_load_fragment_in_gripper_env(environment):
    
    fragment_spawned = False

    frag_id = -1

    while True:
        if fragment_spawned == False and environment.get_current_gripper_opening() >= 0.084:
            frag_id = environment.spawn_urdf_fragment_in_gripper(2,6)
            fragment_spawned = True
            print("Fragment was spawned and it has the id =",frag_id)

        if fragment_spawned == True:
            joint_change = 0.001
            current_opening = environment.get_current_gripper_opening()
            goal_opening = current_opening - joint_change
            environment.move_gripper(gripper_opening_length=goal_opening)

        environment.step_simulation(environment.per_step_iterations)

# Load a fragment and open and close the gripper
def test_open_close_env(environment):
    environment.load_urdf_fragment(2, 6, [0.4, 0., 0.01], [0, 0, 0])

    open, close, joint_change = False, True, 0.001

    while True: 
        environment.step_simulation(environment.per_step_iterations)

        current_opening = environment.get_current_gripper_opening()

        if current_opening > 0.084 and close and not open:
            open, close, joint_change = True, False, -0.001
        if current_opening < 0.001 and open and not close:
            open, close, joint_change = False, True, 0.001

        goal_opening = current_opening + joint_change
        environment.move_gripper(gripper_opening_length=goal_opening)

def test_load_fragment_env(environment):
    environment.load_urdf_fragment(2, 6, [0.4, 0., 0.01], [0,0,np.pi/2])
    while True:
        environment.step_simulation(environment.per_step_iterations)

def test_robot_start_state_env(environment):

    while True:
        state = environment._p.getLinkState(environment.robot_id, environment.tool_tip_id, physicsClientId=environment.client_id)
        goalEndEffectorOri = np.array(environment._p.getEulerFromQuaternion(state[1]))
        print("goalEndEffectorOri =",goalEndEffectorOri)
        environment.step_simulation(environment.per_step_iterations)

if __name__ == '__main__':
    environment = load_default_task()
    test_robot_start_state_env(environment)