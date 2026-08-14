import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'LIBERO'))
import time
import copy
import math
import tqdm
import yaml
import json
import torch
import imageio
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from inference import modeling, predict_action, ACTTemporalEnsembler
from collections import deque

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DATE = time.strftime("%Y_%m_%d")


class Config:
    yaml_path: str = "./config/deco_libero_80m.yaml"
    task_config: str = "./dataset/libero/tasks.json"
    # LIBERO task suites: libero_spatial, libero_object, libero_goal, libero_10
    task_suite_name: str = "libero_object"
    # Number of trials per task
    num_trials_per_task: int = 50
    local_log_dir: str = "./eval_libero/0.5B/libero_object"        # Local directory for eval logs
    open_loop: bool = True
    select_action: int = 16  # total chunksize get better performance
    # ACTTemporalEnsembler config
    use_temporal_ensembler: bool = False  # whether to use temporal ensembler when open_loop=False
    temporal_ensemble_coeff: float = 0.1  # higher values favor recent predictions more


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def quat2axisangle(quat):
    """Convert quaternion [x, y, z, w] to axis-angle [ax, ay, az]."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.acos(w)
    return (quat[:3] * angle) / den


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, rollout_dir="./"):
    """Saves an MP4 replay of an episode."""
    rollout_dir = os.path.join(rollout_dir, DATE)
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


config = Config()
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def eval_libero():
    # Load model
    yaml_config = yaml.safe_load(open(config.yaml_path, 'r'))
    model = modeling(yaml_config)
    model = model.to(DEVICE)

    # Get chunk_size for temporal ensembler init
    chunk_size = yaml_config['model']['chunk_size']

    # Initialize temporal ensembler (only used when open_loop=False)
    if config.use_temporal_ensembler:
        temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, chunk_size)
        print(f"Initialized ACTTemporalEnsembler with coeff={config.temporal_ensemble_coeff}, chunk_size={chunk_size}")

    with open(config.task_config, 'r') as f:
        tasks_config = json.load(f)

    # reverse key and values in task
    tasks_config = {v: k for k, v in tasks_config.items()}

    # Initialize logging
    run_id = f"EVAL-{config.task_suite_name}-{DATE_TIME}"
    os.makedirs(config.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(config.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO benchmark tasks
    benchmark_dict = benchmark.get_benchmark_dict()

    # Get the specified task suite
    task_suite = benchmark_dict[config.task_suite_name]()

    # Number of tasks in the suite
    num_tasks_in_suite = task_suite.n_tasks

    print(f"Task suite: {config.task_suite_name}, task_num: {num_tasks_in_suite}")
    log_file.write(f"Task suite: {config.task_suite_name}, task_num: {num_tasks_in_suite}\n")

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):

        task = task_suite.get_task(task_id)
        # Get initial states for task i from the suite
        initial_states = task_suite.get_task_init_states(task_id)
        # Object positions vary slightly each trial; initial_states is an array of slight perturbations.

        # Initialize LIBERO simulation environment and get task description
        env, task_description = get_libero_env(task, resolution=256)
        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(config.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            # Get the condition index as a class embedding for model
            condition_idx = int(tasks_config[task_description])
            env.reset()

            # Set initial state
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            action_queue = deque()  # initialize action queue

            # Reset temporal ensembler (at the start of each episode)
            if config.use_temporal_ensembler:
                temporal_ensembler.reset()

            if config.task_suite_name == "libero_spatial":
                max_steps = 220 
            elif config.task_suite_name == "libero_object":
                max_steps = 280  
            elif config.task_suite_name == "libero_goal":
                max_steps = 300 
            elif config.task_suite_name == "libero_10":
                max_steps = 520 
            else:
                raise NotImplementedError

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            while t < max_steps + 10:
                # First 10 steps: do nothing, objects may still be falling
                if t < 10:
                    # Dummy action: 7 joint angles, gripper closed
                    obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                    t += 1
                    continue

                # Get environment observations
                img1 = obs["agentview_image"][::-1, ::-1]
                img2 = obs["robot0_eye_in_hand_image"][::-1, ::-1]
                replay_images.append(copy.deepcopy(img1))
                state = np.concatenate([
                            obs["robot0_eef_pos"],                       # 3D position
                            quat2axisangle(obs["robot0_eef_quat"]),      # quaternion → axis-angle (3D)
                            obs["robot0_gripper_qpos"],                  # 2D gripper
                        ])

                # Queue-based action management
                if len(action_queue) == 0:
                    # Queue empty: predict new actions and enqueue
                    actions = predict_action(model, DEVICE, yaml_config, imgs=[img1, img2], obs=state, task_idx=condition_idx)
                    actions[..., -1] = np.sign(actions[..., -1])

                    if config.open_loop:
                        # open_loop=True: use predicted actions directly
                        action_queue.extend(actions[:config.select_action].tolist())
                    else:
                        # open_loop=False: smooth with ACTTemporalEnsembler
                        if config.use_temporal_ensembler:
                            # temporal_ensembler.update expects (batch, chunk_size, dim)
                            action = temporal_ensembler.update(actions.unsqueeze(0))
                            action = action.squeeze(0).numpy()
                            action_queue.append(action.tolist())
                        else:
                            action_queue.append(actions[0].tolist())

                # Pop one action from queue and execute
                action = action_queue.popleft()
                # Step the environment
                obs, reward, done, info = env.step(action)
                t += 1

                # Check if done
                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            task_episodes += 1
            total_episodes += 1

            # Save video
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, rollout_dir=config.local_log_dir
            )

            # Log results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Summary log
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    eval_libero()