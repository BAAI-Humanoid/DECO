"""Evaluate a trained DECO policy on the DexJoCo benchmark."""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from inference import modeling, predict_action
from utils.convert_dexjoco import DEXJOCO_TASKS, DUAL_ARM_TASKS


CLICK_MOUSE_ALIGN_STEPS = 30
CLICK_MOUSE_ALIGN_ACTION = np.array(
    [
        -4.4294e-01,
        1.3729e-06,
        1.5170,
        -3.14156462,
        -6.91584035e-05,
        -1.40317984e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.263,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    with path.open('r') as file:
        return yaml.safe_load(file)


def validate_model_config(config):
    model_config = config['model']
    expected = {
        'action_dim': 44,
        'n_view': 3,
        'obs_dim': 46,
        'num_tasks': len(DEXJOCO_TASKS),
    }
    mismatches = {
        key: (model_config.get(key), value)
        for key, value in expected.items()
        if model_config.get(key) != value
    }
    if mismatches:
        details = ', '.join(
            f'{key}={actual} (expected {expected_value})'
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f'Invalid unified DexJoCo model config: {details}')


def load_model(config, checkpoint_path, device):
    model_config = copy.deepcopy(config)
    model_config['model']['pretrain_model_path'] = False
    model_config['model']['adapter_model_path'] = False
    model = modeling(model_config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {
        key.removeprefix('module.'): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def import_dexjoco_config_mapping(dexjoco_root):
    package_root = dexjoco_root / 'dexjoco'
    if not package_root.is_dir():
        raise FileNotFoundError(
            f'DexJoCo Python package directory not found: {package_root}'
        )

    sys.path.insert(0, str(package_root))
    from dexjoco.tasks import CONFIG_MAPPING

    return CONFIG_MAPPING


def load_task_config(dexjoco_root, regime, task_name):
    config_path = dexjoco_root / 'configs' / regime / f'{task_name}.yaml'
    if not config_path.is_file():
        raise FileNotFoundError(f'DexJoCo task config not found: {config_path}')

    config = load_yaml(config_path)
    if config.get('env_name') != task_name:
        raise ValueError(
            f'{config_path} defines env_name={config.get("env_name")!r}, '
            f'expected {task_name!r}'
        )
    return config


class DexJoCoEnv:
    """Adapt DexJoCo observations and actions to the DECO training format."""

    def __init__(
        self,
        task_name,
        task_config,
        config_mapping,
        regime,
        seed,
        render_mode,
        randomize_dynamics,
    ):
        self.task_name = task_name
        self.task_idx = DEXJOCO_TASKS.index(task_name)
        self.dual_arm = task_name in DUAL_ARM_TASKS
        self.camera_names = tuple(task_config['camera_mapping'].values())
        self.env = self._create_environment(
            task_config,
            config_mapping,
            regime,
            seed,
            render_mode,
            randomize_dynamics,
        )
        self.observation = None
        self.done = False
        self.success = False

    def _create_environment(
        self,
        task_config,
        config_mapping,
        regime,
        seed,
        render_mode,
        randomize_dynamics,
    ):
        expected_robot_type = 'dual_arm' if self.dual_arm else 'single_arm'
        if task_config.get('robot_type') != expected_robot_type:
            raise ValueError(
                f'{self.task_name} uses robot_type={task_config.get("robot_type")!r}, '
                f'expected {expected_robot_type!r}'
            )

        environment_kwargs = {}
        if self.task_name == 'bimanual_unlock_ipad':
            password = task_config.get('password')
            if password is not None:
                environment_kwargs['password'] = password

        task = config_mapping[self.task_name]()
        return task.get_environment(
            policy_mode=True,
            render_mode=render_mode,
            randomize=regime == 'rand_full',
            seed=seed,
            randomize_dynamics=randomize_dynamics,
            **environment_kwargs,
        )

    def reset(self):
        observation, _ = self.env.reset()
        self.observation = observation
        self.done = False
        self.success = False

    def close(self):
        self.env.close()

    def get_policy_observation(self, data_config):
        images = [
            np.ascontiguousarray(self.observation[name])
            for name in self.camera_names
        ]
        state = np.asarray(self.observation['state'], dtype=np.float32)

        if self.dual_arm:
            return images, state[:46]

        wrist_image = np.flip(images[1], axis=1).copy()
        if data_config['norm_type'] == 'mean_std':
            unified_state = np.asarray(
                data_config['observation_mean'],
                dtype=np.float32,
            ).copy()
        else:
            observation_min = np.asarray(
                data_config['observation_min'],
                dtype=np.float32,
            )
            observation_max = np.asarray(
                data_config['observation_max'],
                dtype=np.float32,
            )
            unified_state = (observation_min + observation_max) / 2
        unified_state[:7] = state[:7]
        unified_state[14:30] = state[7:23]
        return [images[0], images[1], wrist_image], unified_state

    def get_raw_images(self):
        return {
            name: np.ascontiguousarray(self.observation[name])
            for name in self.camera_names
        }

    def step(self, policy_action):
        environment_action = self._to_environment_action(policy_action)
        observation, _, terminated, truncated, info = self.env.step(
            environment_action
        )
        self.observation = observation
        self.done = bool(terminated or truncated)
        self.success = bool(info.get('succeed', False))

    def _to_environment_action(self, action):
        if self.dual_arm:
            right_quaternion = Rotation.from_rotvec(action[3:6]).as_quat(
                scalar_first=True
            )
            left_quaternion = Rotation.from_rotvec(action[25:28]).as_quat(
                scalar_first=True
            )
            return np.concatenate(
                [
                    action[:3],
                    right_quaternion,
                    action[22:25],
                    left_quaternion,
                    action[6:22],
                    action[28:44],
                ]
            )

        quaternion = Rotation.from_rotvec(action[3:6]).as_quat(
            scalar_first=True
        )
        return np.concatenate([action[:3], quaternion, action[6:22]])


def align_click_mouse(env):
    for _ in range(CLICK_MOUSE_ALIGN_STEPS):
        env.step(CLICK_MOUSE_ALIGN_ACTION)


def open_video_writers(output_dir, camera_names, fps):
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise ImportError(
            'Video recording requires imageio. Install it or omit --save-video.'
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        name: imageio.get_writer(output_dir / f'{name}.mp4', fps=fps)
        for name in camera_names
    }


def append_video_frames(writers, images):
    for name, writer in writers.items():
        writer.append_data(images[name])


def close_video_writers(writers):
    for writer in writers.values():
        writer.close()


def infer_action_chunk(model, device, config, env):
    images, state = env.get_policy_observation(config['data'])
    actions = predict_action(
        model,
        device,
        config,
        images,
        state,
        task_idx=env.task_idx,
    )
    actions = actions.numpy()
    return actions if env.dual_arm else actions[:, :22]


def evaluate_episode(
    model,
    device,
    config,
    env,
    max_steps,
    action_horizon,
    video_dir,
    video_fps,
):
    env.reset()
    if env.task_name == 'click_mouse':
        align_click_mouse(env)

    writers = {}
    if video_dir is not None:
        writers = open_video_writers(video_dir, env.camera_names, video_fps)
        append_video_frames(writers, env.get_raw_images())

    step = 0
    try:
        while step < max_steps and not env.done:
            action_chunk = infer_action_chunk(model, device, config, env)
            execution_horizon = min(action_horizon, len(action_chunk))

            for action in action_chunk[:execution_horizon]:
                env.step(action)
                step += 1
                if writers:
                    append_video_frames(writers, env.get_raw_images())
                if env.done or step >= max_steps:
                    break
    finally:
        close_video_writers(writers)

    return {
        'success': env.success,
        'steps': step,
    }


def evaluate_task(
    model,
    device,
    config,
    task_name,
    task_config,
    config_mapping,
    args,
    run_dir,
):
    env = DexJoCoEnv(
        task_name=task_name,
        task_config=task_config,
        config_mapping=config_mapping,
        regime=args.regime,
        seed=args.seed,
        render_mode=args.render_mode,
        randomize_dynamics=args.randomize_dynamics,
    )
    task_dir = run_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    episode_results = []

    try:
        for episode_idx in range(args.episodes):
            video_dir = None
            if args.save_video:
                video_dir = task_dir / f'episode_{episode_idx:03d}_temp'

            result = evaluate_episode(
                model=model,
                device=device,
                config=config,
                env=env,
                max_steps=args.max_steps,
                action_horizon=args.action_horizon,
                video_dir=video_dir,
                video_fps=args.video_fps,
            )
            episode_results.append(result)

            outcome = 'success' if result['success'] else 'failure'
            if video_dir is not None:
                video_dir.rename(
                    task_dir / f'episode_{episode_idx:03d}_{outcome}'
                )
            print(
                f'{task_name} episode {episode_idx + 1}/{args.episodes}: '
                f'{outcome}, steps={result["steps"]}'
            )
    finally:
        env.close()

    successes = sum(result['success'] for result in episode_results)
    return {
        'task_index': env.task_idx,
        'successes': successes,
        'episodes': args.episodes,
        'success_rate': successes / args.episodes,
        'episode_results': episode_results,
    }


def resolve_tasks(tasks_arg):
    if tasks_arg == 'all':
        return list(DEXJOCO_TASKS)

    tasks = [task.strip() for task in tasks_arg.split(',') if task.strip()]
    unknown_tasks = [task for task in tasks if task not in DEXJOCO_TASKS]
    if unknown_tasks:
        raise ValueError(f'Unknown DexJoCo tasks: {unknown_tasks}')
    if not tasks:
        raise ValueError('No DexJoCo tasks were selected')
    return tasks


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def validate_args(args, chunk_size):
    if args.episodes <= 0:
        raise ValueError('--episodes must be greater than zero')
    if args.max_steps <= 0:
        raise ValueError('--max-steps must be greater than zero')
    if args.action_horizon is None:
        args.action_horizon = chunk_size
    if not 1 <= args.action_horizon <= chunk_size:
        raise ValueError(
            f'--action-horizon must be between 1 and {chunk_size}'
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a unified DECO policy on DexJoCo.'
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='DECO DexJoCo model configuration YAML.',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Trained DECO checkpoint (best.pth or last_weights.pth).',
    )
    parser.add_argument(
        '--dexjoco-root',
        type=Path,
        required=True,
        help='Root of the local DexJoCo repository.',
    )
    parser.add_argument(
        '--regime',
        choices=('rand_obj', 'rand_full'),
        default='rand_obj',
    )
    parser.add_argument(
        '--tasks',
        default='all',
        help='all or a comma-separated list of task names.',
    )
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=1500)
    parser.add_argument(
        '--action-horizon',
        type=int,
        default=None,
        help='Actions executed per prediction. Defaults to the model chunk size.',
    )
    # 实际执行的chunk步数
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/dexjoco_eval'),
    )
    parser.add_argument(
        '--render-mode',
        choices=('rgb_array', 'human'),
        default='rgb_array',
    )
    parser.add_argument('--randomize-dynamics', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--video-fps', type=int, default=30)
    return parser


def main():
    args = build_parser().parse_args()
    os.environ.setdefault(
        'MUJOCO_GL',
        'egl' if args.render_mode == 'rgb_array' else 'glfw',
    )

    config = load_yaml(args.config)
    validate_model_config(config)
    validate_args(args, config['model']['chunk_size'])
    tasks = resolve_tasks(args.tasks)
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f'Loading checkpoint on {device}: {args.checkpoint}')
    model = load_model(config, args.checkpoint, device)
    config_mapping = import_dexjoco_config_mapping(args.dexjoco_root)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = args.output / f'{args.regime}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=False)

    task_results = {}
    for task_name in tasks:
        task_config = load_task_config(
            args.dexjoco_root,
            args.regime,
            task_name,
        )
        task_results[task_name] = evaluate_task(
            model=model,
            device=device,
            config=config,
            task_name=task_name,
            task_config=task_config,
            config_mapping=config_mapping,
            args=args,
            run_dir=run_dir,
        )

    total_successes = sum(
        result['successes'] for result in task_results.values()
    )
    total_episodes = sum(
        result['episodes'] for result in task_results.values()
    )
    summary = {
        'config': str(args.config),
        'checkpoint': str(args.checkpoint),
        'dexjoco_root': str(args.dexjoco_root),
        'regime': args.regime,
        'seed': args.seed,
        'action_horizon': args.action_horizon,
        'tasks': task_results,
        'average': {
            'successes': total_successes,
            'episodes': total_episodes,
            'success_rate': total_successes / total_episodes,
        },
    }
    summary_path = run_dir / 'summary.json'
    with summary_path.open('w') as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'Results saved to {summary_path}')


if __name__ == '__main__':
    main()
