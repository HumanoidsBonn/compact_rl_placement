#!/usr/bin/python3

from stable_baselines3.common.env_util import make_vec_env
import argparse
from fragment_gym.utils import common_utils

class RunBaseline():
    def __init__(self):
        self.common_utils = common_utils.Common()

    def get_terminal_params(self):
        parser = argparse.ArgumentParser(description="Train and test fragment gym RL agents using Pybullet.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-f", "--fresco_range", type=str, default="-1", help="Frescoes that are loaded, e.g. 0,1. -1 uses the ones defined in the config file.")
        parser.add_argument("-p", "--path", type=str, default="./baseline/", help="Path where YAML configuration files are stored.")
        parser.add_argument("-c", "--config", type=str, default="baseline_scaling_fresco", choices=["baseline_scaling_fresco", "baseline_relative_placing"], help="Name of the YAML configuration file which defines the evaluation name.")
        parser.add_argument("-m", "--mode", type=str, default="eval", choices=["eval"], help="Evaluation mode.")
        parser.add_argument("-g", "--gui", type=str, default="n", choices=["y", "n"], help="Show GUI or not.")
        parser.add_argument("-r", "--realtime", type=str, default="n", choices=["y", "n"], help="Show output in real time (affects test mode only).")
        parser.add_argument("-d", "--debug", type=str, default="n", choices=["y", "n"], help="Show debug prints and visual elements.")
        parser.add_argument("-v", "--visualize", type=str, default="n", choices=["y", "n"], help="Show debug prints and visual elements.")
        parser.add_argument("-s", "--save_plot", type=str, default="n", choices=["y", "n"], help="Show debug prints and visual elements.")
        parser.add_argument("-e", "--execute_on", type=str, default="cpu", choices=["cpu", "cuda"], help="Specify if yout want to run the code on the CPU or GPU (cuda for nvidia GPUs)")
            
        args = parser.parse_args()
        params = vars(args)

        return params

    def run_baseline(self, env_name, baseline_name, config, mode, fresco_range, real_time, debug, visualize, save_plot):       
        assembly_plan_type = "assembly_plan_snake"
        if len(fresco_range) == 0:
            eval_frescoes = list(config["eval_frescoes"])
        else:
            eval_frescoes = fresco_range
        
        # Initialize environments
        baseline_env = make_vec_env(env_name, n_envs=1, env_kwargs=dict(mode=mode, config=config, real_time=real_time, debug=debug, fresco_range=eval_frescoes, ablation=""))
        print("\nEvaluation frescoes", eval_frescoes)
        
        # Runs until True is returned from main_task()
        for fresco_no in range(eval_frescoes[0], eval_frescoes[1]+1):
            print("\nStart evaluating fresco ", fresco_no)
            while not baseline_env.envs[0].env.env.task_main(baseline_name=baseline_name, fresco_no=fresco_no, assembly_plan_type=assembly_plan_type, visualize = visualize, save_plot = save_plot):
                continue
        print("Task completed")

        baseline_env.envs[0].env.env.close()


    def main(self):
        # LOAD PARAMETERS FROM TERMINAL
        # ===========================================================
        params = self.get_terminal_params()
        self.common_utils.print_dict(params, "Parameters (Terminal):")

        # LOAD EXPERIMENT CONFIG FROM YAML FILE
        # ===========================================================
        fresco_range = list(map(int, str(params["fresco_range"]).split(",")))
        if len(fresco_range) != 2:
            if fresco_range[0] == -1:
                fresco_range = [] # Fresco ranfe from json is used
            else:
                print("Invalid fresco range:", fresco_range)
                raise ValueError
        base_path = str(params["path"])
        config_file_name = str(params["config"])
        mode = str(params["mode"])
        gui = str(params["gui"])
        real_time = str(params["realtime"])
        debug = str(params["debug"])
        visualize = str(params["visualize"])
        save_plot = str(params["save_plot"])
        hardware = str(params["execute_on"])

        yaml_path = base_path + "config/"+ config_file_name + ".yaml"
        config = self.common_utils.read_yaml(yaml_path)
        #print_dict(config, "Configuration (YAML):")

        task_name = config["task_name"]
        baseline_name = config_file_name
        if gui == "y":
            task_name = task_name + "GUI" + "-v0"
        elif gui == "n":
            task_name = task_name + "-v0"
        else:
            print("GUi is not defined.")
            raise ValueError
        
        if real_time == "y":
            real_time_bool = True
        elif real_time == "n":
            real_time_bool = False
        else:
            print("Real time option is not defined.")
            raise ValueError
        
        if debug == "y":
            debug_bool = True
        elif debug == "n":
            debug_bool = False
        else:
            print("Debug option is not defined.")
            raise ValueError

        if visualize == "y":
            visualize_bool = True
        elif visualize == "n":
            visualize_bool = False
        else:
            print("Visualize option is not defined.")
            raise ValueError

        if save_plot == "y":
            save_plot_bool = True
        elif save_plot == "n":
            save_plot_bool = False
        else:
            print("Saving plot option is not defined.")
            raise ValueError

        if hardware == "cpu" or hardware == "cuda":
            pass
        else:
            print("Please specify execute_on: cpu/cuda")
            raise ValueError
        
        # Pytorch settings
        self.common_utils.torch_settings(hardware)

        # Seed settings
        if "seed" in config:
            seed = int(config["seed"])
            print("seed =",seed)
            self.common_utils.set_random_seed(hardware, random_seed=seed)
        else:
            print("seed = random")

        self.run_baseline(env_name=task_name, baseline_name=baseline_name, config=config, mode=mode, fresco_range=fresco_range, real_time=real_time_bool, debug=debug_bool, visualize=visualize_bool, save_plot=save_plot_bool)

if __name__ == '__main__':
    run_baseline = RunBaseline()
    run_baseline.main()
