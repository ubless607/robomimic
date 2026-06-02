"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import shutil
import json
import multiprocessing as mp
import faulthandler
import signal
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import tempfile
import traceback

import torch

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config.config import Config

from robomimic.utils.dataset import SequenceDataset, MetaDataset
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper, FrameStackWrapper
from robomimic.algo import RolloutPolicy, algo_factory
from robomimic.config.base_config import config_factory


_WORKER_ROLLOUT_POLICY = None
_WORKER_ROLLOUT_ENV = None


def _to_plain_python(value):
    """
    Recursively convert nested config-like containers to plain Python types.
    """
    if isinstance(value, Config):
        return {k: _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, OrderedDict):
        return OrderedDict((k, _to_plain_python(v)) for k, v in value.items())
    if isinstance(value, list):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_python(v) for v in value)
    return value


def _write_worker_crash_log(prefix, exc):
    """
    Write a worker crash traceback to a temp file so pool failures are debuggable.
    """
    crash_path = os.path.join(tempfile.gettempdir(), "{}_{}.log".format(prefix, os.getpid()))
    with open(crash_path, "w") as crash_file:
        crash_file.write(traceback.format_exc())
        crash_file.write("\nEXC: {}\n".format(repr(exc)))
    print("worker crash log written to {}".format(crash_path), flush=True)


def _init_rollout_worker(policy_state, algo_name, config_dict, obs_key_shapes, ac_dim, obs_normalization_stats, action_normalization_stats, device_str):
    """
    Initialize a worker-local rollout policy once per process.
    """
    global _WORKER_ROLLOUT_POLICY, _WORKER_ROLLOUT_ENV
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        faulthandler.enable()
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        config = config_factory(algo_name, dic=deepcopy(_to_plain_python(config_dict)))
        ObsUtils.initialize_obs_utils_with_config(config, verbose=False)
        model = algo_factory(
            algo_name=algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device,
        )
        model.deserialize(policy_state, load_optimizers=False)
        _WORKER_ROLLOUT_POLICY = RolloutPolicy(
            model,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )
        _WORKER_ROLLOUT_ENV = None
    except Exception as exc:
        _write_worker_crash_log("rollout_worker_init", exc)
        raise


def get_exp_dir(config, auto_remove_exp_dir=False, resume=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
        resume (bool): if True, resume an existing training run instead of creating a 
            new experiment directory
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if resume:
        assert os.path.exists(base_output_dir), "Resuming training run, but output dir {} does not exist".format(base_output_dir)
        subdir_lst = os.listdir(base_output_dir)
        time_str = sorted(subdir_lst)[-1]  # get the most recent subdirectory
        assert os.path.isdir(os.path.join(base_output_dir, time_str)), "Found item {} that is not a subdirectory in {}".format(time_str, base_output_dir)
    elif os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir, exist_ok=resume)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir, exist_ok=resume)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir, exist_ok=resume)

    time_dir = os.path.join(base_output_dir, time_str)
    
    return log_dir, output_dir, video_dir, time_dir


def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.experiment.validate, "specified validation filter key {}, but config.experiment.validate is not set".format(valid_filter_by_attribute)

    # load the dataset into memory
    if config.experiment.validate:
        # assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        assert isinstance(config.train.data, list), "config.train.data should be a list of datasets, not a single dataset"
        for dataset_cfg in config.train.data:
            train_demo_keys = FileUtils.get_demos_for_filter_key(
                hdf5_path=os.path.expanduser(dataset_cfg["path"]),
                filter_key=train_filter_by_attribute,
            )
            valid_demo_keys = FileUtils.get_demos_for_filter_key(
                hdf5_path=os.path.expanduser(dataset_cfg["path"]),
                filter_key=valid_filter_by_attribute,
            )
            assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
                "validation demonstrations!"
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    # NOTE: currently supporting fixed language embedding per dataset
    ## that is fetched from dataset config and not from file
    if LangUtils.LANG_EMB_OBS_KEY in obs_keys:
        obs_keys.remove(LangUtils.LANG_EMB_OBS_KEY)
        ds_langs = [ds_cfg.get("lang", "dummy") for ds_cfg in config.train.data]
    else:
        ds_langs = [None for _ in config.train.data]

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
        load_next_obs=config.train.hdf5_load_next_obs, # whether to load next observations (s') from dataset
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute,
    )

    ds_kwargs["hdf5_path"] = [ds_cfg["path"] for ds_cfg in config.train.data]
    ds_kwargs["filter_by_attribute"] = [ds_cfg.get("filter_key", filter_by_attribute) for ds_cfg in config.train.data]
    ds_kwargs["demo_limit"] = [ds_cfg.get("demo_limit", None) for ds_cfg in config.train.data]
    ds_weights = [ds_cfg.get("weight", 1.0) for ds_cfg in config.train.data]

    meta_ds_kwargs = dict()

    dataset = get_dataset(
        ds_class=SequenceDataset,
        ds_kwargs=ds_kwargs,
        ds_weights=ds_weights,
        ds_langs=ds_langs,
        normalize_weights_by_ds_size=config.train.normalize_weights_by_ds_size,
        meta_ds_class=MetaDataset,
        meta_ds_kwargs=meta_ds_kwargs,
    )

    return dataset


def get_dataset(
    ds_class,
    ds_kwargs,
    ds_weights,
    ds_langs,
    normalize_weights_by_ds_size,
    meta_ds_class=MetaDataset,
    meta_ds_kwargs=None,
):
    """
    Create a dataset object from the provided class and parameters.

    Args:
        ds_class (class): class of the dataset to create (e.g. SequenceDataset)
        ds_kwargs (dict): keyword arguments to pass to the dataset class constructor
        ds_weights (list): list of weights for each dataset instance, used in MetaDataset
        ds_langs (list): list of language embeddings for each dataset instance
        normalize_weights_by_ds_size (bool): if True, normalize dataset weights by the size of each dataset
        meta_ds_class (class): class of the meta dataset to create (e.g. MetaDataset)
        meta_ds_kwargs (dict): keyword arguments to pass to the meta dataset class constructor
    
    Returns:
        ds (SequenceDataset or MetaDataset instance): dataset object created from the provided class and parameters
    """
    ds_list = []
    for i in range(len(ds_weights)):
        
        ds_kwargs_copy = deepcopy(ds_kwargs)

        keys = ["hdf5_path", "filter_by_attribute", "demo_limit"]

        for k in keys:
            ds_kwargs_copy[k] = ds_kwargs[k][i]

        ds_kwargs_copy["lang"] = ds_langs[i]
        
        ds_list.append(ds_class(**ds_kwargs_copy))
    
    if len(ds_weights) == 1:
        ds = ds_list[0]
    else:
        if meta_ds_kwargs is None:
            meta_ds_kwargs = dict()
        ds = meta_ds_class(
            datasets=ds_list,
            ds_weights=ds_weights,
            normalize_weights_by_ds_size=normalize_weights_by_ds_size,
            **meta_ds_kwargs
        )

    return ds


def batchify_obs(obs_list):
    """
    Converts a list of observation dictionaries into a single dictionary.
    """
    keys = list(obs_list[0].keys())
    obs = {
        k: np.stack([obs_list[i][k] for i in range(len(obs_list))]) for k in keys
    }
    
    return obs

def _set_random_seed(seed, env=None):
    """
    Seed all random number generators and the environment.
    """
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass

    # Try to seed the environment and its underlying layers if applicable
    if env is not None:
        for env_obj in [env, getattr(env, "env", None), getattr(env, "base_env", None)]:
            if env_obj is not None:
                if hasattr(env_obj, "seed"):
                    try:
                        env_obj.seed(seed)
                    except Exception:
                        pass
                if hasattr(env_obj, "unwrapped") and hasattr(env_obj.unwrapped, "seed"):
                    try:
                        env_obj.unwrapped.seed(seed)
                    except Exception:
                        pass


def run_rollout(
        policy, 
        env, 
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
        seed=None,
    ):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    if seed is not None:
        _set_random_seed(seed, env=env)

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    rews = []
    success = None # success metrics

    end_step = None

    video_frames = []
    
    try:
        for step_i in range(horizon):
            # get action from policy
            policy_ob = ob_dict
            ac = policy(ob=policy_ob, goal=goal_dict)

            # play action
            ob_dict, r, done, _ = env.step(ac)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            rews.append(r)

            cur_success_metrics = env.is_success()

            if success is None:
                success = deepcopy(cur_success_metrics)
            else:
                for k in success:
                    success[k] = success[k] | cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    frame = env.render(mode="rgb_array", height=128, width=128)
                    video_frames.append(frame)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                end_step = step_i
                break

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))


    if video_writer is not None:
        for frame in video_frames:
            video_writer.append_data(frame)

    end_step = end_step or step_i
    total_reward = np.sum(rews[:end_step + 1])
    
    results["Return"] = total_reward
    results["Horizon"] = end_step + 1
    results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results


def _get_rollout_env_template(env):
    """
    Capture enough information from an environment to re-create rollout copies.
    """
    frame_stack_num_frames = None
    while isinstance(env, FrameStackWrapper):
        if frame_stack_num_frames is not None:
            raise Exception("parallel rollout currently supports a single frame stack wrapper")
        frame_stack_num_frames = env.num_frames
        env = env.env

    env_meta = env.serialize()
    use_image_obs = bool(getattr(getattr(env, "env", None), "use_camera_obs", False))
    use_depth_obs = bool(getattr(getattr(env, "env", None), "camera_depths", False))

    return dict(
        env_meta=env_meta,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
        frame_stack_num_frames=frame_stack_num_frames,
    )


def _clone_rollout_env(env_template, render_offscreen):
    """
    Create a fresh rollout environment from a captured template.
    """
    env = EnvUtils.create_env_from_metadata(
        env_meta=deepcopy(env_template["env_meta"]),
        render=False,
        render_offscreen=render_offscreen,
        use_image_obs=env_template["use_image_obs"],
        use_depth_obs=env_template["use_depth_obs"],
        verbose=False,
    )
    if env_template["frame_stack_num_frames"] is not None:
        env = FrameStackWrapper(env, num_frames=env_template["frame_stack_num_frames"])
    return env


def _close_rollout_env(env):
    """
    Close a rollout environment if it exposes a close method.
    """
    env_close = getattr(env, "close", None)
    if callable(env_close):
        env_close()


def _get_worker_rollout_env(env_template, render_offscreen):
    """
    Lazily create one rollout environment per worker process and reuse it.
    """
    global _WORKER_ROLLOUT_ENV
    if _WORKER_ROLLOUT_ENV is None:
        _WORKER_ROLLOUT_ENV = _clone_rollout_env(env_template, render_offscreen=render_offscreen)
    return _WORKER_ROLLOUT_ENV


def _run_rollout_chunk_worker(
        env_template,
        episode_indices,
        env_key,
        horizon,
        use_goals,
        render,
        video_dir,
        epoch,
        video_skip,
        terminate_on_success,
    ):
    """
    Run a chunk of episodes using the worker-local rollout policy.
    """
    try:
        assert _WORKER_ROLLOUT_POLICY is not None
        return _run_rollout_chunk(
            policy=_WORKER_ROLLOUT_POLICY,
            env=_get_worker_rollout_env(env_template, render_offscreen=(video_dir is not None)),
            episode_indices=episode_indices,
            env_key=env_key,
            horizon=horizon,
            use_goals=use_goals,
            render=render,
            video_dir=video_dir,
            epoch=epoch,
            video_skip=video_skip,
            terminate_on_success=terminate_on_success,
            close_env=False,
        )
    except Exception as exc:
        _write_worker_crash_log("rollout_worker_chunk", exc)
        raise


def _run_rollout_chunk_worker_star(args):
    """
    Multiprocessing helper that expands a packed argument tuple.
    """
    return _run_rollout_chunk_worker(*args)


def _run_rollout_chunk(
        policy,
        env,
        episode_indices,
        env_key,
        horizon,
        use_goals,
        render,
        video_dir,
        epoch,
        video_skip,
        terminate_on_success,
        progress_bar=None,
        close_env=True,
    ):
    """
    Run a subset of rollouts for a single environment.
    """
    chunk_results = []
    try:
        for episode_index in episode_indices:
            env_video_writer = None
            episode_video_path = None
            if video_dir is not None:
                env_video_dir = os.path.join(video_dir, env_key)
                os.makedirs(env_video_dir, exist_ok=True)
                episode_video_path = os.path.join(
                    env_video_dir,
                    "epoch_{}_episode_{}.mp4".format(epoch, episode_index + 1),
                )
                env_video_writer = imageio.get_writer(episode_video_path, fps=20)

            base_seed = 0
            if hasattr(policy, "policy") and hasattr(policy.policy, "global_config"):
                config = policy.policy.global_config
                if hasattr(config, "train") and hasattr(config.train, "seed"):
                    base_seed = config.train.seed
            seed = base_seed + episode_index

            rollout_timestamp = time.time()
            rollout_info = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
                seed=seed,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            if env_video_writer is not None:
                env_video_writer.close()

            chunk_results.append((episode_index, rollout_info, episode_video_path))
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if close_env:
            _close_rollout_env(env)

    return chunk_results


def _combine_video_files(output_path, input_paths, fps=20):
    """
    Combine a sequence of video files into a single output video.
    """
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for input_path in input_paths:
            reader = imageio.get_reader(input_path)
            try:
                for frame in reader:
                    writer.append_data(frame)
            finally:
                reader.close()
    finally:
        writer.close()


def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
        num_parallel_envs=1,
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory. In serial mode this
            is one video per environment; in parallel mode this is one video per episode under a
            per-environment subdirectory.

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        num_parallel_envs (int): number of parallel rollout workers to use per environment. If > 1,
            rollout episodes are split across workers and each episode gets its own video file.

        verbose (bool): if True, print results of each rollout
    
    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...) 
            averaged across all rollouts 

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)
    assert num_episodes is not None and num_episodes > 0

    num_parallel_envs = max(1, int(num_parallel_envs))
    use_parallel = (num_parallel_envs > 1 and num_episodes > 1)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    if use_parallel:
        assert video_path is None, "parallel rollout currently supports video_dir only"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = { k : video_path for k in envs }
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = { k : video_writer for k in envs }
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
        video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
        if not use_parallel:
            video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }

    for env_key, env in envs.items():
        env_name = env.name

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env_name, horizon, use_goals, num_episodes,
        ))

        if not use_parallel:
            env_video_writer = None
            if write_video:
                print("video writes to " + video_paths[env_key])
                env_video_writer = video_writers[env_key]

            rollout_logs = []
            iterator = range(num_episodes)
            if not verbose:
                iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

            num_success = 0
            for ep_i in iterator:
                base_seed = 0
                if hasattr(policy, "policy") and hasattr(policy.policy, "global_config"):
                    config = policy.policy.global_config
                    if hasattr(config, "train") and hasattr(config.train, "seed"):
                        base_seed = config.train.seed
                seed = base_seed + ep_i

                rollout_timestamp = time.time()
                rollout_info = run_rollout(
                    policy=policy,
                    env=env,
                    horizon=horizon,
                    render=render,
                    use_goals=use_goals,
                    video_writer=env_video_writer,
                    video_skip=video_skip,
                    terminate_on_success=terminate_on_success,
                    seed=seed,
                )
                rollout_info["time"] = time.time() - rollout_timestamp

                rollout_logs.append(rollout_info)
                num_success += rollout_info["Success_Rate"]

                if verbose:
                    print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                    print(json.dumps(rollout_info, sort_keys=True, indent=4))

            if env_video_writer is not None and video_dir is not None:
                env_video_writer.close()

            rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
            rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
            rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
            all_rollout_logs[env_key] = rollout_logs_mean
            continue

        env_template = _get_rollout_env_template(env)
        num_workers = min(num_parallel_envs, num_episodes)
        episode_chunks = [[episode_index] for episode_index in range(num_episodes)]
        env_video_dir = None
        if video_dir is not None:
            env_video_dir = os.path.join(video_dir, env_key)
            os.makedirs(env_video_dir, exist_ok=True)

        rollout_logs = []
        combined_video_path = video_paths[env_key] if video_dir is not None else None
        temp_video_paths = []
        base_model = policy.policy
        policy_state = TensorUtils.to_device(TensorUtils.clone(base_model.nets.state_dict()), "cpu")
        algo_name = base_model.global_config.algo_name
        config_dict = _to_plain_python(base_model.global_config.to_dict())
        obs_key_shapes = _to_plain_python(deepcopy(base_model.obs_key_shapes))
        ac_dim = base_model.ac_dim
        worker_device_str = str(base_model.device)
        env_template = _to_plain_python(env_template)
        obs_normalization_stats = _to_plain_python(policy.obs_normalization_stats)
        action_normalization_stats = _to_plain_python(policy.action_normalization_stats)

        with LogUtils.custom_tqdm(total=num_episodes, desc=env_name, dynamic_ncols=True, mininterval=0.0, miniters=1, leave=True) as progress_bar:
            chunk_results = []
            pool_context = mp.get_context("spawn")
            with pool_context.Pool(
                processes=num_workers,
                initializer=_init_rollout_worker,
                initargs=(
                    policy_state,
                    algo_name,
                    config_dict,
                    obs_key_shapes,
                    ac_dim,
                    obs_normalization_stats,
                    action_normalization_stats,
                    worker_device_str,
                ),
            ) as pool:
                try:
                    chunk_args = [
                        (
                            env_template,
                            episode_chunk,
                            env_key,
                            horizon,
                            use_goals,
                            render,
                            video_dir,
                            epoch,
                            video_skip,
                            terminate_on_success,
                        )
                        for episode_chunk in episode_chunks
                    ]

                    for chunk_result in pool.imap_unordered(_run_rollout_chunk_worker_star, chunk_args, chunksize=1):
                        chunk_results.extend(chunk_result)
                        progress_bar.update(len(chunk_result))
                        progress_bar.refresh()
                except KeyboardInterrupt:
                    pool.terminate()
                    pool.join()
                    raise

        chunk_results.sort(key=lambda item: item[0])
        num_success = 0
        for episode_index, rollout_info, episode_video_path in chunk_results:
            rollout_info["time"] = rollout_info.get("time", rollout_info.get("Time_Episode", 0.0))
            rollout_logs.append(rollout_info)
            if episode_video_path is not None:
                temp_video_paths.append(episode_video_path)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(episode_index + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if combined_video_path is not None:
            _combine_video_files(output_path=combined_video_path, input_paths=temp_video_paths, fps=20)
            for temp_video_path in temp_video_paths:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            assert video_dir is not None
            temp_video_dir = os.path.join(video_dir, env_key)
            if os.path.isdir(temp_video_dir) and len(os.listdir(temp_video_dir)) == 0:
                os.rmdir(temp_video_dir)

        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        all_rollout_logs[env_key] = rollout_logs_mean
        if combined_video_path is not None:
            video_paths[env_key] = combined_video_path

    if video_path is not None:
        video_writer.close()

    return all_rollout_logs, video_paths


def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
    ):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a 
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a 
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                should_save_ckpt = True
                ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )


def save_model(model, config, env_meta, shape_meta, ckpt_path, variable_state=None, obs_normalization_stats=None, action_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        variable_state (dict): internal variable state in main train loop, used for restoring training process
            from ckpt

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

        action_normalization_stats (dict): optionally pass a dictionary for action
            normalization. This should map action keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the action.
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
        variable_state=variable_state,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    if action_normalization_stats is not None:
        action_normalization_stats = deepcopy(action_normalization_stats)
        params["action_normalization_stats"] = TensorUtils.to_list(action_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))


def run_epoch(model, data_loader, epoch, validate=False, num_steps=None, obs_normalization_stats=None):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

        obs_normalization_stats (dict or None): if provided, this should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)
    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)
        timing_stats["Data_Loading"].append(time.time() - t)

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)
        model.on_gradient_step()

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    # add in timing stats
    for k in timing_stats:
        # sum across all training steps, and convert from seconds to minutes
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.

    return step_log_all


def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval. 
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)
    
    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0
