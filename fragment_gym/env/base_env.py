#!/usr/bin/python3

import pybullet as p
from pybullet_utils import bullet_client
import gymnasium as gym
from gymnasium.utils import EzPickle
import pybullet_data
import os

class BasePybulletEnv(gym.Env):
    def __init__(self, mode=None, ablation=None, config=None, real_time=None, debug=None, render=False, shared_memory=True, use_egl=False, fresco_range=[]):
        EzPickle.__init__(**locals())
        self.mode = mode
        self.ablation = ablation
        self.config = config
        self.root_path = os.path.join(os.path.dirname(__file__), '../')
        self.frescoes_path = self.root_path + "meshes/fragments/"
        self.real_time = real_time
        self.debug = debug
        self._p = None
        self.step_size_fraction = 240.0 # Warning: DO NOT CHANGE. See Pybullet Quickstart Guide setTimeStep
        self._urdfRoot = pybullet_data.getDataPath()
        self.egl = use_egl
        self.shared_memory = shared_memory
        self.render = render
        self.pybullet_init()

    def _reset_base_simulation(self):
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -9.81)

    def pybullet_init(self):
        self.p_gui_res = [1920, 1080]
        render_option = p.DIRECT
        if self.render:
            render_option = p.GUI
        self._p = bullet_client.BulletClient(connection_mode=render_option, options=f"--width={self.p_gui_res[0]} --height={self.p_gui_res[1]}")
        self._urdfRoot = pybullet_data.getDataPath()
        self._p.setAdditionalSearchPath(self._urdfRoot)
        # self._egl_plugin = None
        # if not self.render and "feature_extractor" in self.config:
        #     print("I will use the alternative renderer")
        #     assert sys.platform == 'linux', ('EGL rendering is only supported on ''Linux.')
        #     egl = pkgutil.get_loader('eglRenderer')
        #     if egl:
        #         self._egl_plugin = self._p.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
        #     else:
        #         self._egl_plugin = self._p.loadPlugin('eglRendererPlugin')
        #     print('EGL renderering enabled.')
        
        if self.real_time:
            self._p.setRealTimeSimulation(1)

        
        self._p.setPhysicsEngineParameter(enableFileCaching=0)
        self._p.setPhysicsEngineParameter(numSolverIterations=200)
        self._p.setTimeStep(1. / self.step_size_fraction) # Warning: DO NOT CHANGE. See Pybullet Quickstart Guide setTimeStep

        self.p_gui_pos = [0,0]
        self.p_gui_name = "Bullet Physics ExampleBrowser using OpenGL3+ [btgl] Release build"

        self._p.resetDebugVisualizerCamera(
            cameraDistance=1.2000000476837158,
            cameraYaw=24.79998207092285,
            cameraPitch=-25.799999237060547,
            cameraTargetPosition=[0.13571400940418243, 0.30781200528144836, -0.005765209440141916]
        )
        
        self.client_id = self._p._client
        self._reset_base_simulation()
        control_frequency = int(self.config["control_frequency"])
        self.per_step_iterations = int(self.step_size_fraction / control_frequency)
        print("pybullet initialized")

    def print_cam_settings(self):
            cam_settings = self._p.getDebugVisualizerCamera()
            #print("cam_settings =",cam_settings)
            print("cameraDistance =",cam_settings[10])
            print("cameraYaw =",cam_settings[8])
            print("cameraPitch =",cam_settings[9])
            print("cameraTargetPosition =",cam_settings[11])
            pass

    def step_simulation(self, num_steps):
        if self.real_time:
            pass
        else:
            for _ in range(int(num_steps)):
                self._p.stepSimulation(physicsClientId=self.client_id)

    def close(self):
        # if self._egl_plugin is not None:
        #     p.unloadPlugin(self._egl_plugin)
        self._p.disconnect()

    def step(self, action):
        return None, None, None, None

    def reset(self):
        pass

    def reset_sim(self, id):
        self._p.restoreState(id, physicsClientId=self.client_id)

    def get_complete_body_list(self):
        self.step_simulation(self.per_step_iterations) # Ensure that contraints are up-to-date
        body_list = [[],[]]
        num_bodies = self._p.getNumBodies()
        for i in range (num_bodies):
            body_id = self._p.getBodyUniqueId(i)
            body_list[0].append(body_id)
            body_list[1].append(self._p.getBodyInfo(body_id))

        return body_list
       

    def get_complete_constraint_list(self):
        self.step_simulation(self.per_step_iterations) # Ensure that contraints are up-to-date
        num_constraints = self._p.getNumConstraints()
        constraint_list = [[],[],[]]
        for i in range (num_constraints):
            constraint_id = self._p.getConstraintUniqueId(i)
            constraint_list[0].append(constraint_id)
            constraint_list[1].append(self._p.getConstraintInfo(constraint_id))
            constraint_list[2].append(self._p.getConstraintState(constraint_id))
        return constraint_list
    
    def get_all_constraint_ids(self):
        self.step_simulation(self.per_step_iterations) # Ensure that contraints are up-to-date
        num_constraints = self._p.getNumConstraints()
        constraint_list = []
        for i in range (num_constraints):
            constraint_id = self._p.getConstraintUniqueId(i)
            constraint_list.append(constraint_id)
        return constraint_list
    
    def remove_contraint(self, constraint_id):
        self.step_simulation(self.per_step_iterations) # Ensure that contraints are up-to-date
        self._p.removeConstraint(constraint_id, physicsClientId=self.client_id)
        self.step_simulation(self.per_step_iterations) # Update all constraints

    def remove_all_contraints(self):
        self.step_simulation(self.per_step_iterations) # Ensure that contraints are up-to-date
        constraint_list = self.get_all_constraint_ids()
        for constraint_id in constraint_list:
            self._p.removeConstraint(constraint_id, physicsClientId=self.client_id)
        self.step_simulation(self.per_step_iterations) # Update all constraints

    def remove_all_contraints_except(self, exception_list):
        self.step_simulation(self.per_step_iterations) # Ensure that contraints are up-to-date
        constraint_list = self.get_all_constraint_ids()
        for exception_id in exception_list:
            constraint_list.remove(exception_id)
        for constraint_id in constraint_list:
            self._p.removeConstraint(constraint_id, physicsClientId=self.client_id)
        self.step_simulation(self.per_step_iterations) # Update all constraints