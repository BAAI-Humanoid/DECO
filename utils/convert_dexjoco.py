import os
import json
import argparse
import subprocess
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm


SINGLE_ARM_TASKS = [
    'click_mouse',
    'fold_glasses',
    'hammer_nail',
    'pick_bucket',
    'pinch_tongs',
    'water_plant',
]

DUAL_ARM_TASKS = [
    'bimanual_assembly',
    'bimanual_hanoi',
    'bimanual_microwave_cook',
    'bimanual_photograph',
    'bimanual_unlock_ipad',
]

DEXJOCO_TASKS = SINGLE_ARM_TASKS + DUAL_ARM_TASKS

REGIME_DIRS = {
    'rand_obj': 'dexjoco_lerobot_datasets',
    'rand_full': 'dexjoco_lerobot_datasets_rand_full',
}

SINGLE_SCENE_CAMERAS = {
    'click_mouse': 'observation.images.ego_right',
    'fold_glasses': 'observation.images.front',
    'hammer_nail': 'observation.images.front',
    'pick_bucket': 'observation.images.front',
    'pinch_tongs': 'observation.images.front',
    'water_plant': 'observation.images.front',
}


def save_tasks_json(save_path):
    prompt_map = {
        str(idx).zfill(4): f"Perform task: {name.replace('_', ' ')}."
        for idx, name in enumerate(DEXJOCO_TASKS)
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_map, f, ensure_ascii=False, indent=2)


def get_camera_keys(task_name, regime):
    if task_name in DUAL_ARM_TASKS:
        scene_camera = (
            'observation.images.ego'
            if regime == 'rand_obj'
            else 'observation.images.random_camera'
        )
        return [
            scene_camera,
            'observation.images.wrist_left',
            'observation.images.wrist_right',
        ]

    scene_camera = (
        SINGLE_SCENE_CAMERAS[task_name]
        if regime == 'rand_obj'
        else 'observation.images.random_camera'
    )
    return [scene_camera, 'observation.images.wrist']


def extract_frames(video_path, save_path, start_time, num_frames):
    os.makedirs(save_path, exist_ok=True)
    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-y',
        '-ss',
        str(start_time),
        '-i',
        video_path,
        '-frames:v',
        str(num_frames),
        '-start_number',
        '0',
        os.path.join(save_path, 'frame_%06d.png'),
    ]
    subprocess.run(command, check=True)


def process_episode(args):
    (
        episode_idx,
        task_idx,
        task_path,
        output_path,
        is_bimanual,
        camera_keys,
        episode,
        states,
        actions,
    ) = args

    episode_info = pd.DataFrame({
        'state': list(states),
        'action': list(actions),
        'task': task_idx,
        'is_bimanual': is_bimanual,
    })

    episode_path = os.path.join(output_path, f'episode_{episode_idx:06d}')
    os.makedirs(episode_path, exist_ok=True)
    episode_info.to_pickle(os.path.join(episode_path, 'episode_info.pkl'))

    for image_idx, camera_key in enumerate(camera_keys, start=1):
        prefix = f'videos/{camera_key}'
        chunk_idx = int(episode[f'{prefix}/chunk_index'])
        file_idx = int(episode[f'{prefix}/file_index'])
        video_path = os.path.join(
            task_path,
            'videos',
            camera_key,
            f'chunk-{chunk_idx:03d}',
            f'file-{file_idx:03d}.mp4',
        )
        extract_frames(
            video_path,
            os.path.join(episode_path, f'img{image_idx}'),
            float(episode[f'{prefix}/from_timestamp']),
            len(states),
        )


def build_episode_jobs(
    data,
    episodes,
    episode_idx,
    task_idx,
    task_path,
    output_path,
    is_bimanual,
    camera_keys,
):
    for local_idx, (_, episode) in enumerate(episodes.iterrows()):
        source_episode_idx = int(episode['episode_index'])
        episode_data = data[data['episode_index'] == source_episode_idx]
        episode_data = episode_data.sort_values('frame_index')
        yield (
            episode_idx + local_idx,
            task_idx,
            task_path,
            output_path,
            is_bimanual,
            camera_keys,
            episode.to_dict(),
            np.stack(episode_data['observation.state']),
            np.stack(episode_data['action']),
        )


def convert_dataset(data_path, output_path, regime, workers, episode_idx):
    source_path = os.path.join(data_path, REGIME_DIRS[regime])

    with Pool(workers) as pool:
        for task_idx, task_name in enumerate(DEXJOCO_TASKS):
            task_path = os.path.join(source_path, task_name)
            data = pd.read_parquet(
                os.path.join(
                    task_path, 'data', 'chunk-000', 'file-000.parquet'
                )
            )
            episodes = pd.read_parquet(
                os.path.join(
                    task_path,
                    'meta',
                    'episodes',
                    'chunk-000',
                    'file-000.parquet',
                )
            )
            is_bimanual = task_name in DUAL_ARM_TASKS
            camera_keys = get_camera_keys(task_name, regime)
            jobs = build_episode_jobs(
                data,
                episodes,
                episode_idx,
                task_idx,
                task_path,
                output_path,
                is_bimanual,
                camera_keys,
            )

            for _ in tqdm(
                pool.imap_unordered(process_episode, jobs),
                total=len(episodes),
                desc=f'{regime}: {task_name}',
            ):
                pass
            episode_idx += len(episodes)

    return episode_idx


def main():
    parser = argparse.ArgumentParser(
        description='Convert DexJoCo datasets to the DECO data format.'
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the DexJoCo dataset root.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Output directory for converted datasets.')
    parser.add_argument('--workers', type=int,
                        default=min(32, os.cpu_count() or 1),
                        help='Number of parallel episode workers.')
    args = parser.parse_args()

    output_path = os.path.join(args.save_path, 'data')
    os.makedirs(output_path, exist_ok=True)

    episode_idx = 0
    for regime in REGIME_DIRS:
        episode_idx = convert_dataset(
            args.data_path,
            output_path,
            regime,
            args.workers,
            episode_idx,
        )

    save_tasks_json(os.path.join(args.save_path, 'tasks.json'))


if __name__ == '__main__':
    main()
