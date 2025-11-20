#!/usr/bin/python3

import numpy as np
import os
import json
from matplotlib import pyplot as plt
from shapely import plotting
from pathlib import Path
import shutil
import glob
import yaml
import torch
import random

class Common():
    def __init__(self):
        pass

    def create_folder(self, folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    def remove_folder(self, folder_path):
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            pass
    
    def plot_fresco_comparison_image(self, img_path, fresco_polygons_list, ref="fresco", name="", visualize=False, save_plot=False):
        ax, fig = self.plot_fresco_image(img_path, fresco_polygons_list[0], ref=ref, name=name, color="black")
        self.plot_fresco_image(img_path, fresco_polygons_list[1], ref=ref, name=name, color="C0", ax=ax, fig=fig, visualize=visualize, save_plot=save_plot)

    def plot_fresco_image(self, img_path, fresco_polygons, ref="fresco", name="", color="C0", line_width=2.0, plot_grid=False, plot_axis_labels=True, plot_centroids=False, ax=None, fig=None, visualize=False, save_plot=False):
        if name == "":
            name = Path(img_path).stem
        if ax is None:
            if ref == "fresco":
                fresco_length = 0.12
                fresco_width = 0.18
                worst_case_sf = 2.0
                worst_case_length = fresco_length * (1.0+worst_case_sf)
                worst_case_width = fresco_width * (1.0+worst_case_sf)
           
            fig, ax = plt.subplots()
            ax.set_aspect('equal', 'box')
            if ref == "fresco":
                ax.set_xlim(-worst_case_length/2, worst_case_length/2)
                ax.set_ylim(-worst_case_width/2, worst_case_width/2)
        
        # Plot fresco
        plotting.plot_polygon(
            fresco_polygons,
            ax,
            alpha=0.5,
            add_points=False,
            color=color
        )
        # Plot centroids
        if plot_centroids:
            for fragment in fresco_polygons.geoms:
                plotting.plot_points(
                    fragment.centroid,
                    color="black"
                )
        
        if plot_axis_labels:
            ax.set_xlabel(r"$x$ [m]")
            ax.set_ylabel("$y$ [m]")
        else:
            ax.xticks([])
            ax.yticks([]) 
        ax.grid(plot_grid)
        ax.set_title(name)

        if visualize:
            fig.show()
        if save_plot:
            fig.savefig(img_path+".png", dpi=300,bbox_inches='tight')
        
        return ax, fig

    def remove_fresco_image(self, img_path):
        # Remove old shifted fresco
        fresco_image_list = []
        fresco_image_list.extend(glob.glob(img_path))
        for file in fresco_image_list:
            print("Deleting shifted fresco image")
            if os.path.isfile(file):
                os.remove(file)
            else:
                print("Error: %s file was not found and could not be deleted." % img_path)

    def read_yaml(self, yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        return data

    def save_yaml(self, path, data):
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False, width=float("inf"))

    def print_dict(self, dict, header):
        print(header)
        for key, value in dict.items():
            entry = str(key) + " = " + str(value)
            print(entry)
        print("")

    def read_json_file(self, json_path) -> dict:
        file = open(json_path)
        json_data = json.load(file)
        return json_data

    def save_json(self, path, data):
        with open(path, "w") as outfile:
            json.dump(data, outfile, indent=4)

    def load_ground_truth(self, root_path, fresco_no):
        gt_path = root_path + "meshes/fragments/fresco_"+ str(fresco_no) + "/ground_truth.json"
        if os.path.isfile(gt_path):
            gt_data = self.read_json_file(gt_path)
        else:
            print("Ground truth file of fresco could not be loaded.")
            raise ValueError
        return gt_data

    def normalize_value(self, min, max, value):
        normed = (value - min) / (max - min)
        return normed

    def normalize_value_to_min_max(self, c_min, c_max, dist, normed_min, normed_max):
        #if (dist - c_min) == 0 or (c_max - c_min) == 0:
        #    print("dist: ", dist, "c_min: ", c_min, "c_max: ", c_max)
        x_normed = (dist - c_min) / (c_max - c_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        #return round(x_normed, 4)
        return x_normed
    
    # Calculate the angle difference considering the circular nature
    def circular_angle_difference(self, angle1, angle2):
        diff = abs(angle1 - angle2) % (2 * np.pi)
        return min(diff, 2 * np.pi - diff)
    
    def torch_settings(self, hardware):
        print("Execute code on hardware:", hardware)
        if hardware == "cuda":
            print("Cuda is available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                torch.device("cuda")
        else:
            torch.device("cpu")

        torch.set_num_threads(4)
        torch.autograd.set_detect_anomaly(True)

    def set_random_seed(self, hardware, random_seed):
        """
        Generalizing random seed
        """
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if hardware == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            # Important for reproducability
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False     