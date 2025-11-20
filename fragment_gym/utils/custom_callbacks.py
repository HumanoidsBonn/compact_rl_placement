#!/usr/bin/python3

from tqdm.auto import tqdm
import os
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
import numpy as np
from collections import deque
import gymnasium as gym
import warnings
from typing import Any, Dict, Optional, Union
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import glob
from pathlib import Path
from fragment_gym.utils import hugging_face_utils

class SavingCallback(BaseCallback):
    def __init__(self, log_dir, start_step, save_freq, online_model_bool, online_model_params={}, number_of_inter_models_to_keep=-1, save_inter_replay_buffer=True, wait_to_remove_newer_models=200, verbose=0):
        super(SavingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.start_step = start_step
        self.wait_to_remove_newer_models = wait_to_remove_newer_models
        self.number_of_inter_models_to_keep = number_of_inter_models_to_keep
        self.save_inter_replay_buffer = save_inter_replay_buffer
        self.hugging_face_utils = hugging_face_utils.HuggingFaceUtils()
        self.online_model_bool = online_model_bool
        self.online_model_params = online_model_params

    def _on_step(self) -> bool:
        if self.n_calls == self.wait_to_remove_newer_models:
            # Deleting newer models
            # It is done in the callback and not at the programm start
            # to give the user time to stop the training
            # if the selected model to continue training was wrong.
            newer_models_list = []
            newer_models_list.extend(glob.glob(self.log_dir+'inter_model_*'))
            newer_models_list.extend(glob.glob(self.log_dir+'best_model_*'))
            newer_models_list.extend(glob.glob(self.log_dir+'final_model_*'))
            for name in newer_models_list:
                file_name = Path(name).stem
                number = int(file_name.split("_")[2])
                if number > self.start_step:
                    print("Deleting newer model data")
                    if os.path.isfile(name):
                        print("Deleting",name)
                        os.remove(name)
                    else:
                        print("Error: %s file was not found and could not be deleted." % name)

        if self.n_calls % self.save_freq == 0:
            # Check if old models have to be deleted
            old_models_list = []
            old_models_list.extend(glob.glob(self.log_dir+'inter_model_*.zip'))

            old_buffer_list = []
            old_buffer_list.extend(glob.glob(self.log_dir+'inter_model_*.pkl'))

            if self.save_inter_replay_buffer == False and len(old_buffer_list) >= 1:
                path_of_replay_buffer_to_delete = old_buffer_list[0]
                print("Deleting intermediate replay buffer")
                # Delete replay buffer
                if os.path.isfile(path_of_replay_buffer_to_delete):
                    print("Deleting",path_of_replay_buffer_to_delete)
                    os.remove(path_of_replay_buffer_to_delete)
                else:
                    print("Error: %s file was not found and could not be deleted." % path_of_replay_buffer_to_delete)


            if self.number_of_inter_models_to_keep > 0:
                if len(old_models_list) >= self.number_of_inter_models_to_keep:
                    number_list = []
                    for name in old_models_list:
                        file_name = Path(name).stem
                        number_list.append(int(file_name.split("_")[2]))
                    min_index = number_list.index(min(number_list))
                    path_of_model_to_delete = old_models_list[min_index]
                    name_of_model_to_delete = Path(path_of_model_to_delete).stem
                    path_of_replay_buffer_to_delete = self.log_dir+name_of_model_to_delete+"_replay_buffer.pkl"
                    path_of_additional_training_parameters_to_delete = self.log_dir+name_of_model_to_delete+"_additional_training_parameters.yaml"

                    print("Deleting data of oldest intermediate model")
                    # Delete model
                    if os.path.isfile(path_of_model_to_delete):
                        print("Deleting",path_of_model_to_delete)
                        os.remove(path_of_model_to_delete)
                    else:
                        print("Error: %s file was not found and could not be deleted." % path_of_model_to_delete)

                    # Delete replay buffer
                    if os.path.isfile(path_of_replay_buffer_to_delete):
                        print("Deleting",path_of_replay_buffer_to_delete)
                        os.remove(path_of_replay_buffer_to_delete)
                    else:
                        print("Error: %s file was not found and could not be deleted." % path_of_replay_buffer_to_delete)

                    # Delete additional training parameters
                    if os.path.isfile(path_of_additional_training_parameters_to_delete):
                        print("Deleting",path_of_additional_training_parameters_to_delete)
                        os.remove(path_of_additional_training_parameters_to_delete)
                    else:
                        print("Error: %s file was not found and could not be deleted." % path_of_additional_training_parameters_to_delete)


            # Storing new intermediate model
            save_model_name = "inter_model_"+str(self.start_step+self.n_calls)
            save_additional_training_parameters_name = save_model_name+"_additional_training_parameters"
            save_replay_buffer_name = save_model_name+"_replay_buffer"
            print("Save intermediate model:", save_model_name)
            self.model.save(os.path.join(self.log_dir, save_model_name))
            print("Save replay buffer:", save_replay_buffer_name)
            self.model.save_replay_buffer(os.path.join(self.log_dir, save_replay_buffer_name))
            print("Save additional training parameters:", save_additional_training_parameters_name)
            self.training_env.envs[0].env.env.common_utils.save_yaml(path=os.path.join(self.log_dir, save_additional_training_parameters_name+".yaml"), data={"current_curriculum_step": self.training_env.envs[0].env.env.current_curriculum_step})

            # Upload models to Hugging Face
            if self.online_model_bool:
                # Save intermediate model
                self.hugging_face_utils.push_file_to_hub(
                    repo_id=self.online_model_params["repo_id"],
                    filename=self.log_dir+save_model_name+".zip",
                    commit_message=self.online_model_params["commit_msg"]
                    )
                # Save additional training parameters
                self.hugging_face_utils.push_file_to_hub(
                    repo_id=self.online_model_params["repo_id"],
                    filename=self.log_dir+save_additional_training_parameters_name+".yaml",
                    commit_message=self.online_model_params["commit_msg"]
                    )
                
        return True

class CustomEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        start_step: int,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        save_inter_replay_buffer: bool = True,
        eval_best_model_success_threshold: float = 1.0
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.save_inter_replay_buffer = save_inter_replay_buffer
        self.eval_best_model_success_threshold = eval_best_model_success_threshold

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        self.start_step = start_step
        self.old_save_model_name = ""
        self.old_save_replay_buffer_name = ""
        self.old_save_evaluations_name = ""

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

        self.current_curriculum_step = self.training_env.envs[0].env.env.current_curriculum_step
        self.previous_curriculum_step = self.current_curriculum_step

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Check current curriculum step
            self.current_curriculum_step = self.training_env.envs[0].env.env.current_curriculum_step

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )


            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # Reset best_mean_reward when the curriculum step is increased
            if self.current_curriculum_step > self.previous_curriculum_step:
                self.best_mean_reward = -np.inf

            if mean_reward > self.best_mean_reward and len(self._is_success_buffer) > 0:
                if success_rate >= self.eval_best_model_success_threshold:
                    if self.verbose >= 1:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        # Remove old best model
                        print("Deleting old best model data")
                        for name in glob.glob(self.best_model_save_path+'best_model_*'):
                            if os.path.isfile(name):
                                print("Deleting",name)
                                os.remove(name)
                            else:
                                print("Error: %s file was not found and could not be deleted." % name)

                        if self.best_model_save_path is not None:
                            save_model_name = "best_model_"+str(self.start_step+self.n_calls)
                            save_replay_buffer_name = save_model_name+"_replay_buffer"
                            save_additional_training_parameters_name = save_model_name+"_additional_training_parameters.yaml"
                        
                        # Logs will be written in ``file_name.npz``
                        if self.log_path is not None:
                            save_evaluations_name = save_model_name+"_evaluations"
                            log_path = os.path.join(self.log_path, save_evaluations_name)
                        if log_path is not None:
                            os.makedirs(os.path.dirname(log_path), exist_ok=True)

                        if log_path is not None:
                            self.evaluations_timesteps.append(self.num_timesteps)
                            self.evaluations_results.append(episode_rewards)
                            self.evaluations_length.append(episode_lengths)

                            kwargs = {}
                            # Save success log if present
                            if len(self._is_success_buffer) > 0:
                                self.evaluations_successes.append(self._is_success_buffer)
                                kwargs = dict(successes=self.evaluations_successes)

                            np.savez(
                                log_path,
                                timesteps=self.evaluations_timesteps,
                                results=self.evaluations_results,
                                ep_lengths=self.evaluations_length,
                                **kwargs,
                            )

                        print("Save best model:",save_model_name)
                        self.model.save(os.path.join(self.best_model_save_path, save_model_name))
                        self.training_env.envs[0].env.env.common_utils.save_yaml(path=os.path.join(self.best_model_save_path, save_additional_training_parameters_name), data={"current_curriculum_step": self.training_env.envs[0].env.env.current_curriculum_step})

                        if self.save_inter_replay_buffer:
                            print("Save replay buffer: ",save_replay_buffer_name)
                            self.model.save_replay_buffer(os.path.join(self.best_model_save_path, save_replay_buffer_name))

                        self.best_mean_reward = mean_reward
                        # Trigger callback on new best model, if needed
                        if self.callback_on_new_best is not None:
                            continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            # Update previous curriculum step
            self.previous_curriculum_step = self.current_curriculum_step
        return continue_training

class CompleteFrescoEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        start_step: int,
        eval_env: Union[gym.Env, VecEnv],
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        save_inter_replay_buffer: bool = True,
        eval_best_fresco_model_success_threshold: float = 0.0
    ):
        super().__init__(verbose=verbose)
        self.eval_freq = eval_freq

        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.best_success_rate = -np.inf
        self.last_success_rate = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.save_inter_replay_buffer = save_inter_replay_buffer
        self.eval_best_fresco_model_success_threshold = eval_best_fresco_model_success_threshold

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        self.start_step = start_step
        self.old_save_model_name = ""
        self.old_save_replay_buffer_name = ""
        self.old_save_evaluations_name = ""

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        self.current_curriculum_step = self.training_env.envs[0].env.env.current_curriculum_step
        self.previous_curriculum_step = self.current_curriculum_step

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Check current curriculum step
            self.current_curriculum_step = self.training_env.envs[0].env.env.current_curriculum_step

            self.eval_env.envs[0].env.env.mode = "train_eval"
            self.eval_env.envs[0].env.env.place_complete_fresco = True
            n_eval_episodes = self.training_env.envs[0].env.env.no_fragments

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Complete fresco eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval_fresco/mean_reward", float(mean_reward))
            self.logger.record("eval_fresco/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval_fresco/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # Reset best_mean_reward when the curriculum step is increased
            if self.current_curriculum_step > self.previous_curriculum_step:
                self.best_mean_reward = -np.inf
                self.best_success_rate = -np.inf

            if len(self._is_success_buffer) > 0:
                if ((success_rate > self.best_success_rate and success_rate >= self.eval_best_fresco_model_success_threshold)
                    or success_rate == 1.0 and mean_reward > self.best_mean_reward):
                        if self.verbose >= 1:
                            print("New best fresco evaluation model!")
                        if self.best_model_save_path is not None:
                            # Remove old best model
                            print("Deleting old best model data")
                            for name in glob.glob(self.best_model_save_path+'best_fresco_model_*'):
                                if os.path.isfile(name):
                                    print("Deleting",name)
                                    os.remove(name)
                                else:
                                    print("Error: %s file was not found and could not be deleted." % name)

                            if self.best_model_save_path is not None:
                                save_model_name = "best_fresco_model_"+str(self.start_step+self.n_calls)
                                save_replay_buffer_name = save_model_name+"_replay_buffer"
                                save_additional_training_parameters_name = save_model_name+"_additional_training_parameters.yaml"
                            
                            # Logs will be written in ``file_name.npz``
                            if self.log_path is not None:
                                save_evaluations_name = save_model_name+"_evaluations"
                                log_path = os.path.join(self.log_path, save_evaluations_name)
                            if log_path is not None:
                                os.makedirs(os.path.dirname(log_path), exist_ok=True)

                            if log_path is not None:
                                self.evaluations_timesteps.append(self.num_timesteps)
                                self.evaluations_results.append(episode_rewards)
                                self.evaluations_length.append(episode_lengths)

                                kwargs = {}
                                # Save success log if present
                                if len(self._is_success_buffer) > 0:
                                    self.evaluations_successes.append(self._is_success_buffer)
                                    kwargs = dict(successes=self.evaluations_successes)

                                np.savez(
                                    log_path,
                                    timesteps=self.evaluations_timesteps,
                                    results=self.evaluations_results,
                                    ep_lengths=self.evaluations_length,
                                    **kwargs,
                                )

                            print("Save best fresco model:",save_model_name)
                            self.model.save(os.path.join(self.best_model_save_path, save_model_name))
                            self.training_env.envs[0].env.env.common_utils.save_yaml(path=os.path.join(self.best_model_save_path, save_additional_training_parameters_name), data={"current_curriculum_step": self.training_env.envs[0].env.env.current_curriculum_step})

                            if self.save_inter_replay_buffer:
                                print("Save replay buffer: ",save_replay_buffer_name)
                                self.model.save_replay_buffer(os.path.join(self.best_model_save_path, save_replay_buffer_name))

                            self.best_mean_reward = mean_reward
                            self.best_success_rate = success_rate
            
            # Update previous curriculum step
            self.previous_curriculum_step = self.current_curriculum_step

            # Reset place_complete_fresco
            self.eval_env.envs[0].env.env.mode = "train"
            self.eval_env.envs[0].env.env.place_complete_fresco = False

        return continue_training
    

class NewBestCompleteFrescoEvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        save_inter_replay_buffer: bool = True,
        eval_best_fresco_model_success_threshold: float = 0.0
    ):
        super().__init__(verbose=verbose)

        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.best_success_rate = -np.inf
        self.last_success_rate = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.save_inter_replay_buffer = save_inter_replay_buffer
        self.eval_best_fresco_model_success_threshold = eval_best_fresco_model_success_threshold

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        self.old_save_model_name = ""
        self.old_save_replay_buffer_name = ""
        self.old_save_evaluations_name = ""

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        self.current_curriculum_step = self.training_env.envs[0].env.env.current_curriculum_step

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from e

        # Check current curriculum step
        self.current_curriculum_step = self.training_env.envs[0].env.env.current_curriculum_step

        self.eval_env.envs[0].env.env.mode = "train_eval"
        self.eval_env.envs[0].env.env.place_complete_fresco = True
        n_eval_episodes = self.training_env.envs[0].env.env.no_fragments

        # Reset success rate buffer
        self._is_success_buffer = []

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
        )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = mean_reward

        if self.verbose >= 1:
            print(f"Complete fresco eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        self.logger.record("eval_fresco/mean_reward", float(mean_reward))
        self.logger.record("eval_fresco/mean_ep_length", mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose >= 1:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record("eval_fresco/success_rate", success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        # Reset best_mean_reward when the curriculum step is increased
        if self.current_curriculum_step > self.previous_curriculum_step:
            self.best_mean_reward = -np.inf
            self.best_success_rate = -np.inf

        if len(self._is_success_buffer) > 0:
            if ((success_rate > self.best_success_rate and success_rate >= self.eval_best_fresco_model_success_threshold)
                or success_rate == 1.0 and mean_reward > self.best_mean_reward):
                    if self.verbose >= 1:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        # Remove old best model
                        print("Deleting old best model data")
                        for name in glob.glob(self.best_model_save_path+'best_fresco_model_*'):
                            if os.path.isfile(name):
                                print("Deleting",name)
                                os.remove(name)
                            else:
                                print("Error: %s file was not found and could not be deleted." % name)

                        if self.best_model_save_path is not None:
                            save_model_name = "best_fresco_model_"+str(self.num_timesteps)
                            save_replay_buffer_name = save_model_name+"_replay_buffer"
                            save_additional_training_parameters_name = save_model_name+"_additional_training_parameters.yaml"
                        
                        # Logs will be written in ``file_name.npz``
                        if self.log_path is not None:
                            save_evaluations_name = save_model_name+"_evaluations"
                            log_path = os.path.join(self.log_path, save_evaluations_name)
                        if log_path is not None:
                            os.makedirs(os.path.dirname(log_path), exist_ok=True)

                        if log_path is not None:
                            self.evaluations_timesteps.append(self.num_timesteps)
                            self.evaluations_results.append(episode_rewards)
                            self.evaluations_length.append(episode_lengths)

                            kwargs = {}
                            # Save success log if present
                            if len(self._is_success_buffer) > 0:
                                self.evaluations_successes.append(self._is_success_buffer)
                                kwargs = dict(successes=self.evaluations_successes)

                            np.savez(
                                log_path,
                                timesteps=self.evaluations_timesteps,
                                results=self.evaluations_results,
                                ep_lengths=self.evaluations_length,
                                **kwargs,
                            )

                        print("Save best fresco model:",save_model_name)
                        self.model.save(os.path.join(self.best_model_save_path, save_model_name))
                        self.training_env.envs[0].env.env.common_utils.save_yaml(path=os.path.join(self.best_model_save_path, save_additional_training_parameters_name), data={"current_curriculum_step": self.training_env.envs[0].env.env.current_curriculum_step})

                        if self.save_inter_replay_buffer:
                            print("Save replay buffer: ",save_replay_buffer_name)
                            self.model.save_replay_buffer(os.path.join(self.best_model_save_path, save_replay_buffer_name))

                        self.best_mean_reward = mean_reward
                        self.best_success_rate = success_rate

        self.eval_env.envs[0].env.env.mode = "train"
        self.eval_env.envs[0].env.env.place_complete_fresco = False
        return continue_training


    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, reward_keys, stats_window_size = 100, log_interval=4, use_curriculum_learning=False, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.reward_keys = reward_keys
        self.n_episodes = 0
        self.stats_window_size = stats_window_size
        self.log_interval = log_interval
        
        self.use_curriculum_learning = use_curriculum_learning
        
        # Reward
        self.reward_history = deque(maxlen=self.stats_window_size)
        self.rewards_history = {}
        for key in self.reward_keys:
            self.rewards_history[key] = deque(maxlen=self.stats_window_size)
        
        # Distances
        self.target_dist_history = deque(maxlen=self.stats_window_size)
        self.corresponding_corner_distance_history  = deque(maxlen=self.stats_window_size)
        self.ruler_distance_history  = deque(maxlen=self.stats_window_size)
        
        # Contact bools
        self.plane_contact_history = deque(maxlen=self.stats_window_size)
        self.frag_contact_history = deque(maxlen=self.stats_window_size)
        # Contact counts
        self.fragment2robot_contacts_count_history = deque(maxlen=self.stats_window_size)
        self.fragment2fragment_contacts_count_history = deque(maxlen=self.stats_window_size)
        self.plane_contacts_count_history = deque(maxlen=self.stats_window_size)
        
        # Termination
        self.success_history = deque(maxlen=self.stats_window_size)
        self.timeout_history = deque(maxlen=self.stats_window_size)

        # Fragment drop height
        self.fragment_drop_height_history = deque(maxlen=self.stats_window_size)
        self.fragment_drop_angle_history = deque(maxlen=self.stats_window_size)
        self.corner_drop_closeness_history = deque(maxlen=self.stats_window_size)
        self.ruler_drop_closeness_history = deque(maxlen=self.stats_window_size)

    def _on_step(self) -> bool:
        # test1 = self.num_timesteps
        # test2 = self.n_calls
        # test3 = self.training_env.envs[0].total_steps
        # test4= self.training_env.envs[0].env.env.current_episode

        if self.training_env.envs[0].env.env.termination_info:
            self.n_episodes += 1

            # Reward
            self.reward_history.append(float(self.training_env.envs[0].env.env.reward_utils.tensorboard_reward))
            for key in self.rewards_history.keys():
                self.rewards_history[key].append(float(self.training_env.envs[0].env.env.reward_utils.tensorboard_rewards[key]))
            # Reset tensorboard rewards
            self.training_env.envs[0].env.env.reward_utils.tensorboard_reward = 0.0
            for key in self.training_env.envs[0].env.env.reward_utils.tensorboard_rewards.keys():
                self.training_env.envs[0].env.env.reward_utils.tensorboard_rewards[key] = 0.0

            # Distances
            if "retract_distance" in self.reward_keys:
                self.target_dist_history.append(float(self.training_env.envs[0].env.env.tensorboard_target_dist))
            if "corner_distance" in self.reward_keys and self.training_env.envs[0].env.env.placing_fragment["first_fragment"] == False:
                self.corresponding_corner_distance_history.append(float(self.training_env.envs[0].env.env.tensorboard_corresponding_corner_distance))
            if "ruler_distance" in self.reward_keys and self.training_env.envs[0].env.env.placing_fragment["ruler_fragment"] == True:
                self.ruler_distance_history.append(float(self.training_env.envs[0].env.env.tensorboard_ruler_distance))
            # Contacts
            self.plane_contact_history.append(int(self.training_env.envs[0].env.env.tensorboard_plane_contact))
            self.frag_contact_history.append(int(self.training_env.envs[0].env.env.tensorboard_frag_contact))
            # Contact counts
            self.fragment2robot_contacts_count_history.append(int(self.training_env.envs[0].env.env.tensorboard_fragment2robot_contacts_count))
            self.fragment2fragment_contacts_count_history.append(int(self.training_env.envs[0].env.env.tensorboard_fragment2fragment_contacts_count))
            self.plane_contacts_count_history.append(int(self.training_env.envs[0].env.env.tensorboard_plane_contacts_count))
            # Termination
            self.success_history.append(int(self.training_env.envs[0].env.env.tensorboard_success))
            self.timeout_history.append(int(self.training_env.envs[0].env.env.tensorboard_timeout))
            
            # Fragment drop height
            self.fragment_drop_height_history.append(float(self.training_env.envs[0].env.env.tensorboard_fragment_drop_height))
            if self.training_env.envs[0].env.env.placing_fragment["first_fragment"] == False:
                self.corner_drop_closeness_history.append(float(self.training_env.envs[0].env.env.tensorboard_drop_corner_distance))
            if self.training_env.envs[0].env.env.placing_fragment["ruler_fragment"] == True:
                self.ruler_drop_closeness_history.append(float(self.training_env.envs[0].env.env.tensorboard_drop_ruler_distance))
            self.fragment_drop_angle_history.append(float(self.training_env.envs[0].env.env.tensorboard_drop_target_angle))
            
            # Log history                
            if self.n_episodes % self.log_interval == 0:
                # Curriculum step
                if self.use_curriculum_learning:
                    self.logger.record('curriculum_learning/curriculum_step', self.training_env.envs[0].env.env.current_curriculum_step)
                # Reward
                self.logger.record('reward/reward', np.mean(self.reward_history))

                for key in self.rewards_history.keys():
                    self.logger.record('reward_components/reward_'+str(key), np.mean(self.rewards_history[key]))

                # Distances
                if "retract_distance" in self.reward_keys:
                    self.logger.record('distances/restract_target_distance_[m]', np.mean(self.target_dist_history))
                if "corner_distance" in self.reward_keys:
                    self.logger.record('distances/corresponding_corner_distance_[m]', np.mean(self.corresponding_corner_distance_history))
                if "ruler_distance" in self.reward_keys:
                    self.logger.record('distances/corresponding_ruler_distance_[m]', np.mean(self.ruler_distance_history))
                # Contacts
                self.logger.record('termination/plane_contact', np.mean(self.plane_contact_history))
                self.logger.record('termination/frag_contact', np.mean(self.frag_contact_history))
                # Contact counts
                self.logger.record('termination/count_fragment2robot_contacts', np.mean(self.fragment2robot_contacts_count_history))
                self.logger.record('termination/count_fragment2fragment_contacts', np.mean(self.fragment2fragment_contacts_count_history))
                self.logger.record('termination/count_plane_contacts', np.mean(self.plane_contacts_count_history))
                # Termination
                self.logger.record('termination/success', np.mean(self.success_history))
                self.logger.record('termination/timeout', np.mean(self.timeout_history))

                # Fragment drop height
                self.logger.record('distances/drop_height_[m]', np.mean(self.fragment_drop_height_history))
                self.logger.record('distances/drop_angle_[°]', np.mean(self.fragment_drop_angle_history))
                self.logger.record('distances/drop_closeness_corner_[m]', np.mean(self.corner_drop_closeness_history))
                self.logger.record('distances/drop_closeness_ruler_[m]', np.mean(self.ruler_drop_closeness_history))  
    
        return True


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class HParamCallback(BaseCallback):
    def __init__(self):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
    
    from stable_baselines3.common.callbacks import BaseCallback

class CurriculumLearningCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, config, verbose=0):
    #def __init__(self, config, yaml_path, stats_window_size = 100, verbose=0):
        super(CurriculumLearningCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.config = config
        #self.current_episode = 0
        self.previous_episode = 0
        #self.current_curriculum_step = 0
        self.curriculum_transition_trigger = str(self.config["curriculum_transition_trigger"])
        self.curriculum_steps = int(self.config["curriculum_steps"])
        self.curriculum_transition_episodes = int(self.config["curriculum_transition_episodes"])
        self.curriculum_transition_success_rate = float(self.config["curriculum_transition_success_rate"])

        self.eval_cb_frequency = int(self.config["eval_cb_frequency"])
        self.eval_success_buffer = []
        self.eval_success_rate = -1.0
        

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # Get evaluation success rate
        if self.curriculum_transition_trigger == "eval_success":
            self.eval_success_buffer = []
            eval_success_buffer = self.locals["callback"].callbacks[1]._is_success_buffer
            if len(eval_success_buffer) > 0:
                self.eval_success_rate = np.mean(eval_success_buffer)

        if ((self.curriculum_transition_trigger == "episodes"
            and self.training_env.envs[0].env.env.current_episode != self.previous_episode
            and self.training_env.envs[0].env.env.current_curriculum_step < self.curriculum_steps-1
            and (self.training_env.envs[0].env.env.current_episode%self.curriculum_transition_episodes) == 0)
            or
            (self.curriculum_transition_trigger == "eval_success"
            and self.training_env.envs[0].env.env.current_episode != self.previous_episode
            and self.training_env.envs[0].env.env.current_curriculum_step < self.curriculum_steps-1
            and self.eval_success_rate >= self.curriculum_transition_success_rate)
            ):
            self.training_env.envs[0].env.env.current_curriculum_step += 1
            self.previous_episode = self.training_env.envs[0].env.env.current_episode
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass