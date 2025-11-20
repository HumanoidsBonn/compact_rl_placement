#!/usr/bin/python3

import os
from sb3_contrib import TQC
from stable_baselines3.common.noise import NormalActionNoise
from fragment_gym.utils.custom_callbacks import SavingCallback, ProgressBarManager, CustomEvalCallback, CompleteFrescoEvalCallback, TensorboardCallback, CurriculumLearningCallback
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
import torch
import numpy as np
import argparse
import glob
from pathlib import Path
from fragment_gym.utils import common_utils, hugging_face_utils
import wandb
from wandb.integration.sb3 import WandbCallback

class TrainRL():
    def __init__(self):
        self.common_utils = common_utils.Common()
        self.hugging_face_utils = hugging_face_utils.HuggingFaceUtils()
        os.environ["TRUST_REMOTE_CODE"] = "True"
            
    def get_terminal_params(self):
        parser = argparse.ArgumentParser(description="Train and test fragment gym RL agents using Pybullet.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-f", "--fresco_range", type=str, default="0,0", help="Frescoes that are loaded, e.g. 0,1. -1 uses the ones defined in the config file.")
        parser.add_argument("-p", "--path", type=str, default="./networks/rl/", help="Path where YAML configuration files are stored.")
        parser.add_argument("-c", "--config", type=str, default="TQC_grasp_and_place", help="Name of the YAML configuration file.")
        parser.add_argument("-m", "--mode", type=str, default="test", choices=["train", "test", "eval"], help="Train, test or evaluation mode.")
        parser.add_argument("-abl", "--ablation", type=str, default="", choices=["", "no_ruler"], help="Ablation study type.")
        parser.add_argument("-a", "--agent", type=str, default="fresco", help="Agent type. Options: final=final_model, best=best_model, fresco=best_fresco_model, inter_model_x=inter_model_x")
        parser.add_argument("-g", "--gui", type=str, default="n", choices=["y", "n"], help="Show GUI or not.")
        parser.add_argument("-r", "--realtime", type=str, default="n", choices=["y", "n"], help="Show output in real time (affects test mode only).")
        parser.add_argument("-d", "--debug", type=str, default="n", choices=["y", "n"], help="Show debug prints and visual elements.")
        parser.add_argument("-e", "--execute_on", type=str, default="cpu", choices=["cpu", "cuda"], help="Specify if yout want to run the code on the CPU or GPU (cuda for nvidia GPUs)")
        parser.add_argument("-o", "--online_model", type=str, default="y", choices=["y", "n"], help="Allow using an online model stored on Hugging Face.")
        parser.add_argument("-w", "--wandb", type=str, default="y", choices=["y", "n"], help="Use weights and biases for online tensorboard logs.")
            
        args = parser.parse_args()
        params = vars(args)

        return params

    # If a model exists it is loaded else a new one is created
    def create_model(self, hardware, algo, env, config, tensorboard_log_dir):

        policy_kwargs = dict(n_critics=2, activation_fn=torch.nn.LeakyReLU, net_arch=[128, 128, 128])
        sigma = float(config.get("action_noise", 0.1))
        action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                        sigma=sigma * np.ones(env.action_space.shape[-1]))
        replay_buffer_class = None
        replay_buffer_kwargs = None

        print("choosing algorithm: ", algo)
        if algo == "TQC":
            policy = "MultiInputPolicy"
            training_starts_at = float(config["fill_buffer_at_start"])
            env.envs[0].env.env.training_starts_at = training_starts_at
            model = TQC(policy, env, tensorboard_log=tensorboard_log_dir, buffer_size=int(config["buffer_size"]), batch_size=128, tau=0.05, learning_rate=float(config["learning_rate"]),
                    gamma=0.95, train_freq=(10, 'step'), learning_starts=training_starts_at,
                    top_quantiles_to_drop_per_net=2, gradient_steps=-1,  policy_kwargs=policy_kwargs, action_noise=action_noise,
                    replay_buffer_class=replay_buffer_class, replay_buffer_kwargs=replay_buffer_kwargs,
                    verbose=1, device=hardware)
        return model
    
    def load_train(self, hardware, log_dir_name, tensorboard_log_dir, tensorboard_log_name, env, config_file_name, config, wandb_conf, model_name, use_curriculum_learning, online_model_bool, wandb_bool):
        print("LOAD saved model")

        start_step = 0
        model_path = log_dir_name + model_name + ".zip"
        
        # Load or create model
        if os.path.exists(model_path):
            model = TQC.load(model_path, tensorboard_log=tensorboard_log_dir, device=hardware, learning_rate=float(config["learning_rate"]))
            model.set_env(env)
            if os.path.exists(log_dir_name + model_name + "_replay_buffer.pkl"):
                model.load_replay_buffer(log_dir_name + model_name + "_replay_buffer")
                start_step = int(model_name.split("_")[2])
            additional_learning_parameters_path = log_dir_name + model_name + "_additional_training_parameters.yaml"
            if os.path.exists(additional_learning_parameters_path):
                additional_learning_parameters = self.common_utils.read_yaml(yaml_path=additional_learning_parameters_path)
                env.envs[0].env.env.current_curriculum_step = int(additional_learning_parameters["current_curriculum_step"])
        else:
            model = self.create_model(hardware, "TQC", env, config, tensorboard_log_dir)

        # Init WandB
        if wandb_bool:
            run = wandb.init(
                project=wandb_conf["wandb_project_name"],
                group=tensorboard_log_name,
                config=config,
                sync_tensorboard=True
            )
            wandb_run_id = run.id
            wandb_run_name = run.name
        else:
            wandb_run_id = 1
            wandb_run_name = "without WandB"
        
        online_model_params = {
            "repo_id": "rl-fragment-placing/"+config_file_name,
            "commit_msg": "Run "+ wandb_run_name + " (ID=" + str(wandb_run_id) + ")"
        }
        # Init callbacks
        max_iterations = int(config["max_iterations"])
        saving_cb = SavingCallback(
            log_dir_name,
            start_step,
            save_freq=int(config["saving_cb_frequency"]),
            online_model_bool=online_model_bool,
            online_model_params=online_model_params,
            number_of_inter_models_to_keep=int(config["number_of_inter_models_to_keep"]),
            save_inter_replay_buffer=config["save_inter_replay_buffer"],
            wait_to_remove_newer_models=200)
        tensorboard_cb = TensorboardCallback(
            reward_keys=config["reward_keys"],
            stats_window_size = int(config["tensorboard_cb_amount_of_episodes_for_mean"]), 
            use_curriculum_learning = use_curriculum_learning)
        fresco_eval_cb = CompleteFrescoEvalCallback(
            start_step=start_step,
            eval_env=env, 
            log_path=log_dir_name,
            eval_freq=config.get("eval_best_fresco_model_cb_frequency", config["eval_cb_frequency"]),
            best_model_save_path=log_dir_name,
            save_inter_replay_buffer=config["save_inter_replay_buffer"], 
            eval_best_fresco_model_success_threshold=config.get("eval_best_fresco_model_success_threshold", 0.0))

        eval_best_model_success_threshold = config.get("eval_best_model_success_threshold", 0.0)
        if use_curriculum_learning:
            curriculum_transition_trigger = str(config["curriculum_transition_trigger"])
            curriculum_cb = CurriculumLearningCallback(config=config)
            if curriculum_transition_trigger == "episodes":
                eval_cb = CustomEvalCallback(
                    start_step=start_step, 
                    eval_env=env, 
                    best_model_save_path=log_dir_name, 
                    log_path=log_dir_name, 
                    eval_freq=config["eval_cb_frequency"], 
                    n_eval_episodes=config["eval_episodes"], 
                    save_inter_replay_buffer=config["save_inter_replay_buffer"], 
                    eval_best_model_success_threshold=eval_best_model_success_threshold)
                callbacks = [saving_cb, eval_cb, tensorboard_cb, curriculum_cb]
            elif curriculum_transition_trigger == "eval_success":
                eval_cb = CustomEvalCallback(
                    callback_after_eval=curriculum_cb, 
                    start_step=start_step, 
                    eval_env=env, 
                    best_model_save_path=log_dir_name, 
                    log_path=log_dir_name, 
                    eval_freq=config["eval_cb_frequency"], 
                    n_eval_episodes=config["eval_episodes"], 
                    save_inter_replay_buffer=config["save_inter_replay_buffer"], 
                    eval_best_model_success_threshold=eval_best_model_success_threshold)
                callbacks = [saving_cb, eval_cb, tensorboard_cb]
        else:
            eval_cb = CustomEvalCallback(
                start_step=start_step, 
                eval_env=env, 
                best_model_save_path=log_dir_name, 
                log_path=log_dir_name, 
                eval_freq=config["eval_cb_frequency"], 
                n_eval_episodes=config["eval_episodes"], 
                save_inter_replay_buffer=config["save_inter_replay_buffer"], 
                eval_best_model_success_threshold=eval_best_model_success_threshold)
            callbacks = [saving_cb, eval_cb, tensorboard_cb]

        callbacks.append(fresco_eval_cb)

        if wandb_bool:
            wandb_cb = WandbCallback(model_save_path=f"./wandb/{wandb_run_id}",verbose=2)
            callbacks.append(wandb_cb)

        try:
            with ProgressBarManager(max_iterations) as prog_cb:
                callbacks.append(prog_cb)
                model.learn(total_timesteps=max_iterations, tb_log_name=tensorboard_log_name,
                            callback=callbacks, reset_num_timesteps=config["reset_tensorboard_iterations"])
        except KeyboardInterrupt:
            pass
        
        if wandb_bool:
            run.finish()

        # Remove old final model
        print("Deleting old final model data")
        for name in glob.glob(log_dir_name+'final_model_*'):
            if os.path.isfile(name):
                print("Deleting",name)
                os.remove(name)
            else:
                print("Error: %s file was not found and could not be deleted." % name)

        current_step = saving_cb.n_calls
        end_step = start_step + current_step
        model.save(log_dir_name + "final_model_"+str(end_step))
        print("saved final model")
        model.save_replay_buffer(log_dir_name + "final_model_"+str(end_step)+"_replay_buffer")
        print("saved final replay buffer")
        env.envs[0].env.env.common_utils.save_yaml(path=(log_dir_name + "final_model_"+str(end_step)+"_additional_training_parameters.yaml"), data={"current_curriculum_step": env.envs[0].env.env.current_curriculum_step})
        print("saved final additional training parameters")

        # Delete inter model replay buffer
        if config["save_inter_replay_buffer"] == False:
            old_buffer_list = []
            old_buffer_list.extend(glob.glob(log_dir_name+'inter_model_*.pkl'))
            if len(old_buffer_list) > 0:
                path_of_replay_buffer_to_delete = old_buffer_list[0]
                print("Deleting intermediate replay buffer")
                # Delete replay buffer
                if os.path.isfile(path_of_replay_buffer_to_delete):
                    print("Deleting",path_of_replay_buffer_to_delete)
                    os.remove(path_of_replay_buffer_to_delete)
                else:
                    print("Error: %s file was not found and could not be deleted." % path_of_replay_buffer_to_delete)

        # Upload models to Hugging Face
        if online_model_bool:
            self.hugging_face_utils.push_to_hub(
                repo_id=online_model_params["repo_id"],
                filename=log_dir_name,
                commit_message=online_model_params["commit_msg"]
                )


    def load_test(self, hardware, log_dir_name, env, config_file_name, config, real_time, model_name, online_model_bool, use_curriculum_learning, n_eval_episodes = 200, debug=False):
        model_path = log_dir_name + model_name + ".zip"

        if os.path.exists(model_path):  
            model = TQC.load(model_path, device=hardware)
        else: 
            print("No model was found for testing.")
            raise ValueError
        
        if use_curriculum_learning:
            additional_learning_parameters_path = log_dir_name + model_name + "_additional_training_parameters.yaml"
            if os.path.exists(additional_learning_parameters_path):
                additional_learning_parameters = self.common_utils.read_yaml(yaml_path=additional_learning_parameters_path)
                env.envs[0].env.env.current_curriculum_step = int(additional_learning_parameters["current_curriculum_step"])
        obs = env.reset()
        for i in tqdm(range(n_eval_episodes)):
            Done = False
            count = -1
            while not Done:
                count += 1
                if debug:
                    print("Iteration:", count)
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, Done, info = env.step(action)

    def load_eval(self, hardware, log_dir_name, env, config_file_name, config, real_time, model_name, online_model_bool, use_curriculum_learning, fresco_range, ablation, n_eval_episodes = 200, debug=False):
        fresco_no = 0
        model_path = log_dir_name + model_name + ".zip"

        if os.path.exists(model_path):    
            model = TQC.load(model_path, device=hardware)
        else: 
            print("No model was found for testing.")
            raise ValueError
        
        env.envs[0].env.env.place_complete_fresco = True
        env.envs[0].env.env.model_name = config_file_name + "@" +model_name
        # env.envs[0].env.env.fresco_no = fresco_no
        if ablation == "no_ruler":
            env.envs[0].env.env.start_amount_of_fragments_on_table = 1

        
        if len(fresco_range) == 0:
            eval_frescoes = list(config["eval_frescoes"])
        else:
            eval_frescoes = fresco_range
        n_eval_episodes = 0
        for fresco_no in range(eval_frescoes[0], eval_frescoes[1]+1):
            gt_data = self.common_utils.load_ground_truth(root_path="./", fresco_no=fresco_no)
            n_eval_episodes += gt_data["header"]["no_fragments"]
        if ablation == "no_ruler":
            n_eval_episodes -= 1
        
        if use_curriculum_learning:
            additional_learning_parameters_path = log_dir_name + model_name + "_additional_training_parameters.yaml"
            if os.path.exists(additional_learning_parameters_path):
                additional_learning_parameters = self.common_utils.read_yaml(yaml_path=additional_learning_parameters_path)
                env.envs[0].env.env.current_curriculum_step = int(additional_learning_parameters["current_curriculum_step"])
        obs = env.reset()
        for i in tqdm(range(n_eval_episodes)):
            Done = False
            count = -1
            while not Done:
                count += 1
                if debug:
                    print("Iteration:", count)
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, Done, info = env.step(action)

    def rl_fragment(self, env_name, base_path, config, config_file_name, mode, ablation, fresco_range, real_time, debug, model_name, hardware, online_model_bool, wandb_conf, wandb_bool):

        log_dir_name = base_path + "saved_models/" + config_file_name + "/"
        tensorboard_log_dir = base_path  + "tensorboard_logs/"
        tensorboard_log_name = config_file_name

        os.makedirs(log_dir_name, exist_ok=True)

        if "use_curriculum_learning" in config:
            use_curriculum_learning = config["use_curriculum_learning"]
        else:
            use_curriculum_learning = False

        if mode == "train":
            # Initialize environments
            train_env = make_vec_env(env_id=env_name, n_envs=1, env_kwargs=dict(mode=mode, config=config, real_time=False, debug=debug, fresco_range=fresco_range, ablation=ablation))
            self.load_train(hardware, log_dir_name, tensorboard_log_dir, tensorboard_log_name, train_env, config_file_name, config, wandb_conf, model_name, use_curriculum_learning, online_model_bool, wandb_bool)
            train_env.envs[0].env.env.close()
        elif mode == "test":
            # Initialize environments
            test_env = make_vec_env(env_id=env_name, n_envs=1, env_kwargs=dict(mode=mode, config=config, real_time=real_time, debug=debug, fresco_range=fresco_range, ablation=ablation))
            self.load_test(hardware, log_dir_name, test_env, config_file_name, config, real_time, model_name, online_model_bool, use_curriculum_learning, debug=debug)
            test_env.envs[0].env.env.close()
        elif mode == "eval":
            # Initialize environments
            test_env = make_vec_env(env_id=env_name, n_envs=1, env_kwargs=dict(mode=mode, config=config, real_time=real_time, debug=debug, fresco_range=fresco_range, ablation=ablation))
            self.load_eval(hardware, log_dir_name, test_env, config_file_name, config, real_time, model_name, online_model_bool, use_curriculum_learning, fresco_range=fresco_range, ablation=ablation, debug=debug)
            test_env.envs[0].env.env.close()
        else:
            print("Please specify the mode: train/test/eval")
            raise ValueError  

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
        ablation = str(params["ablation"])
        agent = str(params["agent"])
        gui = str(params["gui"])
        real_time = str(params["realtime"])
        debug = str(params["debug"])
        hardware = str(params["execute_on"])
        online_model = str(params["online_model"])
        wandb = str(params["wandb"])

        yaml_path = base_path + "config/"+ config_file_name + ".yaml"
        config = self.common_utils.read_yaml(yaml_path)

        task_name = config["task_name"]
        if gui == "y":
            task_name = task_name + "GUI" + "-v0"
        elif gui == "n":
            task_name = task_name + "-v0"
        else:
            print("GUi is not defined.")
            raise ValueError

        if mode != "train" and mode != "test" and mode != "eval":
            print("Please specify the mode")
            raise ValueError

        if online_model == "y":
            online_model_bool = True
        elif online_model == "n":
            online_model_bool = False
        else:
            print("Online model is not defined.")
            raise ValueError
        
        file_handling_conf = config["file_handling"]
        wandb_conf = file_handling_conf["wandb"]
        huggingface_conf = file_handling_conf["huggingface"]

        model_path = base_path+"saved_models/"+config_file_name+"/"  
        model_name = ""
        file_name = ""
        if agent == "best":
            # Check if local or online model exist
            model_list = glob.glob(model_path+'best_model_*.zip')
            if len(model_list) == 0 and online_model_bool == True:
                online_model_exists, online_model = self.hugging_face_utils.load_online_model(huggingface_conf["huggingface_project_name"], config_file_name, model_path)
                if online_model_exists:
                    model_list = glob.glob(model_path+'best_model_*.zip')
            for name in model_list:
                file_name = Path(name).stem
                if file_name != "":
                    model_name = file_name
        elif agent == "fresco":
            # Check if local or online model exist
            model_list = glob.glob(model_path+'best_fresco_model_*.zip')
            if len(model_list) == 0 and online_model_bool == True:
                online_model_exists, online_model = self.hugging_face_utils.load_online_model(huggingface_conf["huggingface_project_name"], config_file_name, model_path)
                if online_model_exists:
                    model_list = glob.glob(model_path+'best_fresco_model_*.zip')
            for name in model_list:
                file_name = Path(name).stem
                if file_name != "":
                    model_name = file_name
        elif agent == "final":
            # Check if local or online model exist
            model_list = glob.glob(model_path+'final_model_*.zip')
            if len(model_list) == 0 and online_model_bool == True:
                online_model_exists, online_model = self.hugging_face_utils.load_online_model(huggingface_conf["huggingface_project_name"], config_file_name, model_path)
                if online_model_exists:
                    model_list = glob.glob(model_path+'final_model_*.zip')
            for name in model_list:
                file_name = Path(name).stem
                if file_name != "":
                    model_name = file_name
        elif agent[:5] == "inter":
            if os.path.exists(model_path+agent+".zip"):
                model_name = agent
            elif online_model_bool == True:
                online_model_exists, online_model = self.hugging_face_utils.load_online_model(huggingface_conf["huggingface_project_name"], config_file_name, model_path)
                if os.path.exists(model_path+agent+".zip"):
                    model_name = agent
            else:
                if len(glob.glob(model_path+'inter_model_*.zip')) == 0 and online_model_bool == True:
                    online_model_exists, online_model = self.hugging_face_utils.load_online_model(huggingface_conf["huggingface_project_name"], config_file_name, model_path)
                inter_model_list = []
                number_list = []
                for name in glob.glob(model_path+'inter_model_*.zip'):
                    file_name = Path(name).stem
                    inter_model_list.append(file_name)
                    number_list.append(int(file_name.split("_")[2]))
                if len(inter_model_list) > 0:
                    max_index = number_list.index(max(number_list))
                    model_name = inter_model_list[max_index]
        else:
            print("Undefined agent model.")
            raise ValueError
        
        print("Let's",mode, model_name,"\n")

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
            print("Real time option is not defined.")
            raise ValueError
        
        if hardware == "cpu" or hardware == "cuda":
            pass
        else:
            print("Please specify execute_on: cpu/cuda")
            raise ValueError

        if wandb == "y":
            wandb_bool = True
        elif wandb == "n":
            wandb_bool = False
        else:
            print("Weights and biases option is not defined.")
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

        self.rl_fragment(task_name, base_path, config, config_file_name, mode, ablation, fresco_range, real_time_bool, debug_bool, model_name, hardware, online_model_bool, wandb_conf, wandb_bool)

if __name__ == '__main__':
    train_rl = TrainRL()
    train_rl.main()