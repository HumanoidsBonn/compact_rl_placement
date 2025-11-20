#!/usr/bin/python3

import copy
import sys
import numpy as np

class Collision():
    def __init__(self, config, sim, p_id):
        self.config = config
        self.sim = sim
        self._p = p_id

        self.inflate_object_collision_for_training = self.config.get("inflate_object_collision_for_training", False)
        if self.sim.mode == "train" and self.inflate_object_collision_for_training:
            self.inflation_collision_distance = self.config.get("inflation_collision_distance", 0.001)
        else:
            self.inflation_collision_distance = 0.001

    def collision_checks(self, collision, closest_check, body_A, body_B, link_A=-1, link_B=-1, distance=0.001):
        distance = self.inflation_collision_distance
        collision += self._p.getContactPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                              physicsClientId=self.sim.client_id)
        closest_check += self._p.getClosestPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                                  distance=distance, physicsClientId=self.sim.client_id)
        return collision, closest_check

    def collision_check_base_to_base(self, collision, closest_check, body_A, body_B, link_A=-1, link_B=-1, distance=0.001):
        distance = self.inflation_collision_distance
        collision = self._p.getContactPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                              physicsClientId=self.sim.client_id)
        closest_check = self._p.getClosestPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                                  distance=distance, physicsClientId=self.sim.client_id)
        return collision, closest_check
    
    def closest_check(self, body_A, body_B, distance=0.001):
        closest_check = self._p.getClosestPoints(bodyA=body_A, bodyB=body_B,
                                                  distance=distance, physicsClientId=self.sim.client_id)
        if closest_check:
            return True
        else:
            return False

    def object_contact(self, obj_id):
        collision = []
        closest_check = []
        collision += self._p.getContactPoints(bodyA=self.sim.robot_id, bodyB=obj_id,
                                              physicsClientId=self.sim.client_id)
        closest_check += self._p.getClosestPoints(bodyA=self.sim.robot_id, bodyB=obj_id, distance=0.001,
                                                  physicsClientId=self.sim.client_id)
        return True if collision or closest_check else False

    def check_for_base_to_base_contact(self, obj_id_1, obj_id_2):
        contact = close = False
        contact, close = self.collision_check_base_to_base(contact, close, obj_id_1, obj_id_2)
        return True if contact or close else False
    
    # Added modification for inflate_object_collision_for_training
    def check_for_base_to_base_collision(self, obj_id_1, obj_id_2):
        contact = close = False
        collision, closest_check = self.collision_check_base_to_base(contact, close, obj_id_1, obj_id_2)
        # if collision:
        #     self.sim.collision_pair_list.append([obj_id_1, obj_id_2])
        if self.sim.mode == "train" and self.inflate_object_collision_for_training:
            return True if collision or closest_check else False
        else:
            return True if collision else False
    
    def check_for_robot_contact(self, obj_id):
        contact, close = [], []
        all_contacts = []
        for j in range(1, 20):
            contact, close = self.collision_checks(contact, close, self.sim.robot_id, obj_id, link_A=j)
            all_contacts.extend([True if contact or close else False])
        return any(all_contacts)
    
    # Added modification for inflate_object_collision_for_training
    def check_for_robot_collision(self, obj_id):
        contact, close = [], []
        all_contacts = []
        for j in range(1, 20):
            contact, closest_check = self.collision_checks(contact, close, self.sim.robot_id, obj_id, link_A=j)
            if self.sim.mode == "train" and self.inflate_object_collision_for_training:
                all_contacts.extend([True if contact or closest_check else False])
            else:
                all_contacts.extend([True if contact else False])
                # if any(all_contacts):
                #     pass
        return any(all_contacts)
    
    def count_1_to_n_contacts(self, pybullet_ref_id, pybullet_ids):
        total_fragment_contacts = 0
        if len(pybullet_ids) > 0:
            for id in pybullet_ids:
                if id != pybullet_ref_id:
                    frag_cont = self.check_for_base_to_base_contact(pybullet_ref_id, id)
                    if frag_cont:
                        total_fragment_contacts += 1
                    if frag_cont and self.sim.debug:
                        # print in red
                        print("\033[91m"+f"Fragment {pybullet_ref_id} in contact with fragment {id}"+"\033[0m")
        return total_fragment_contacts

    def check_for_1_to_n_contact(self, pybullet_ref_id, pybullet_ids):
        frag_cont = self.count_1_to_n_contacts(pybullet_ref_id, pybullet_ids)
        return True if frag_cont > 0 else False

    def count_1_to_n_collisions(self, pybullet_ref_id, pybullet_ids):
        total_fragment_collisions = 0
        if len(pybullet_ids) > 0:
            for id in pybullet_ids:
                if id != pybullet_ref_id:
                    frag_col = self.check_for_base_to_base_collision(pybullet_ref_id, id)
                    if frag_col:
                        total_fragment_collisions += 1
                    if frag_col and self.sim.debug:
                        # print in red
                        print("\033[91m"+f"Fragment {pybullet_ref_id} in collision with fragment {id}"+"\033[0m")
        return total_fragment_collisions
    
    def check_for_1_to_n_collision(self, pybullet_ref_id, pybullet_ids):
        frag_col = self.count_1_to_n_collisions(pybullet_ref_id, pybullet_ids)
        return True if frag_col > 0 else False

    def count_n_to_n_contacts(self, pybullet_ids):
        total_fragment_contacts = 0
        if len(pybullet_ids) > 0:
            temp_fragments = copy.deepcopy(pybullet_ids)
            temp_ref_fragments = copy.deepcopy(pybullet_ids)
            for current_ref_frag in temp_ref_fragments:
                temp_fragments.pop(0)
                for frag in temp_fragments:
                    frag_cont = self.check_for_base_to_base_contact(current_ref_frag, frag) 
                    if frag_cont:
                        total_fragment_contacts += 1
                    if frag_cont and self.sim.debug:
                        # print in red
                        print("\033[91m"+f"Fragment {current_ref_frag} in contact with fragment {frag}"+"\033[0m")
        return total_fragment_contacts
    
    def check_for_n_to_n_contact(self, pybullet_ids):
        frag_cont = self.count_n_to_n_contacts(pybullet_ids)
        return True if frag_cont > 0 else False
    
    def count_n_to_n_collisions(self, pybullet_ids):
        total_fragment_collisions = 0
        if len(pybullet_ids) > 0:
            temp_fragments = copy.deepcopy(pybullet_ids)
            temp_ref_fragments = copy.deepcopy(pybullet_ids)
            for current_ref_frag in temp_ref_fragments:
                temp_fragments.pop(0)
                for frag in temp_fragments:
                    frag_col = self.check_for_base_to_base_collision(current_ref_frag, frag) 
                    if frag_col:
                        total_fragment_collisions += 1
                    if frag_col and self.sim.debug:
                        #print in red
                        print("\033[91m"+f"Fragment {current_ref_frag} in collision with fragment {frag}"+"\033[0m")
        return total_fragment_collisions
    
    def check_for_n_to_n_collision(self, pybullet_ids):
        frag_col = self.count_n_to_n_collisions(pybullet_ids)
        return True if frag_col > 0 else False
    
    def count_robot_contacts(self, pybullet_ids):
        total_robot_contacts = 0
        if len(pybullet_ids) > 0:
            temp_pybullet_ids = copy.deepcopy(pybullet_ids)
            for id in temp_pybullet_ids:
                robo_cont = self.check_for_robot_contact(id) 
                if robo_cont:
                    total_robot_contacts +=1
                if robo_cont and self.sim.debug:
                    # print in red
                    print("\033[91m"+f"Robot in contact with body {id}"+"\033[0m")
        return total_robot_contacts
    
    def count_robot_collisions(self, pybullet_ids):
        total_robot_collisions = 0
        if len(pybullet_ids) > 0:
            temp_pybullet_ids = copy.deepcopy(pybullet_ids)
            for id in temp_pybullet_ids:
                robo_col = self.check_for_robot_collision(id) 
                if robo_col:
                    total_robot_collisions +=1
                if robo_col and self.sim.debug:
                    # print in red
                    print("\033[91m"+f"Robot in collision with body {id}"+"\033[0m")
        return total_robot_collisions

    def get_min_dist_between_objects(self, body_A, body_B, link_A=-1, link_B=-1, distance_treshold=sys.float_info.max):
        closest_points = self._p.getClosestPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                                  distance=distance_treshold, physicsClientId=self.sim.client_id)
        if closest_points:
            min_distance = min(contact[8] for contact in closest_points)
            return min_distance
        else:
            return float('inf')
        
    def get_3d_min_dist_between_objects(self, body_A, body_B, link_A=-1, link_B=-1, distance_treshold=sys.float_info.max):
        closest_points = self._p.getClosestPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                                  distance=distance_treshold, physicsClientId=self.sim.client_id)
        if closest_points:
            distance_1d_list = []
            pos_A_list = []
            pos_B_list = []
            for contact in closest_points:
                distance_1d_list.append(contact[8])
                pos_A_list.append(contact[5])
                pos_B_list.append(contact[6])
            min_distance_index = distance_1d_list.index(min(distance_1d_list))
            distance_3d = np.array(pos_B_list[min_distance_index], dtype=np.float32) - np.array(pos_A_list[min_distance_index], dtype=np.float32)
            return min(distance_1d_list), distance_3d
        else:
            return float('inf'), np.full((3), float('inf'), dtype=np.float32)
        
    def get_min_placing_object_dist(self, distance_treshold=sys.float_info.max):
        table_fragments_pybullet_ids = self.sim.get_pybullet_id_list_from_table_fragment_dict(self.sim.table_fragments)
        euc_dists = []
        for temp_id in table_fragments_pybullet_ids:
            euc_dist = self.get_min_dist_between_objects(self.sim.placing_fragment["pybullet_id"], temp_id, distance_treshold=distance_treshold)
            euc_dists.append(euc_dist)
        min_euc_dist = min(euc_dists)
        # Clip if there is penetration
        if min_euc_dist < 0.0:
            min_euc_dist = 0.0
        return min_euc_dist
    
    def get_3d_min_placing_object_dist(self, distance_treshold=sys.float_info.max):
        table_fragments_pybullet_ids = self.sim.get_pybullet_id_list_from_table_fragment_dict(self.sim.table_fragments)
        dists_1d = []
        dists_3d = []
        for temp_id in table_fragments_pybullet_ids:
            dist_1d, dist_3d = self.get_3d_min_dist_between_objects(self.sim.placing_fragment["pybullet_id"], temp_id, distance_treshold=distance_treshold)
            dists_1d.append(dist_1d)
            dists_3d.append(dist_3d)
        min_dist_1d = min(dists_1d)
        index_min_dist_1d = dists_1d.index(min(dists_1d))
        min_dist_3d = dists_3d[index_min_dist_1d]
        # Clip if there is penetration
        if min_dist_1d < 0.0:
            min_dist_1d = 0.0
        return min_dist_1d, min_dist_3d
    
    def get_min_robot_to_table_frag_dist(self, distance_treshold=sys.float_info.max):
        table_fragments_pybullet_ids = self.sim.get_pybullet_id_list_from_table_fragment_dict(self.sim.table_fragments)
        euc_dists = []
        for frag_id in table_fragments_pybullet_ids:
            for robot_link in range(1, 20):
                euc_dist = self.get_min_dist_between_objects(self.sim.robot_id, frag_id, robot_link, distance_treshold=distance_treshold)
                euc_dists.append(euc_dist)
        min_euc_dist = min(euc_dists)
        # Clip if there is penetration
        if min_euc_dist < 0.0:
            min_euc_dist = 0.0
        return min_euc_dist

    def get_3d_min_robot_to_table_frag_dist(self, distance_treshold=sys.float_info.max):
        table_fragments_pybullet_ids = self.sim.get_pybullet_id_list_from_table_fragment_dict(self.sim.table_fragments)
        dists_1d = []
        dists_3d = []
        for frag_id in table_fragments_pybullet_ids:
            for robot_link in range(1, 20):
                dist_1d, dist_3d = self.get_3d_min_dist_between_objects(self.sim.robot_id, frag_id, robot_link, distance_treshold=distance_treshold)
                dists_1d.append(dist_1d)
                dists_3d.append(dist_3d)
        min_dist_1d = min(dists_1d)
        index_min_dist_1d = dists_1d.index(min(dists_1d))
        min_dist_3d = dists_3d[index_min_dist_1d]
        # Clip if there is penetration
        if min_dist_1d < 0.0:
            min_dist_1d = 0.0
        return min_dist_1d, min_dist_3d