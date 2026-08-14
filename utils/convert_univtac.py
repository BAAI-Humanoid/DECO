import os
import io
import json
import argparse
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm


def save_tasks_json(task_name_to_idx, save_path):
    """tasks.json: {"0000": "Perform task: <name>."}  -- eval uses it to map description -> idx."""
    prompt_map = {str(idx).zfill(4): f"Perform task: {name.replace('_', ' ')}."
                  for name, idx in task_name_to_idx.items()}
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_map, f, ensure_ascii=False, indent=2)


def bgr_jpeg_to_rgb_jpeg(raw):
    """UniVTAC renders via OpenCV, so every JPEG byte stream in the hdf5 is BGR-ordered even
    though PIL decodes it as RGB. Swap R<->B and re-encode so downstream loaders get true RGB.
    Verified empirically: GelSight Mini frames fail the green-LED G-brightest check, and the
    grasp_classify 'orange_pad' shows up blue until swapped.
    """
    arr = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))[:, :, [2, 1, 0]]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG')
    return buf.getvalue()


def process_episode(args):
    """Convert one episode and return per-episode state/action statistics for aggregation.

    UniVTAC hdf5 stores head/wrist/tactile frames as JPEG byte strings, but they are
    BGR-ordered (rendered via OpenCV). We swap R<->B and re-encode to true RGB on write.

    state[t] = joint[t]      (t = 0..T-2)
    action[t] = joint[t+1]   (t = 0..T-2)
    Both are 9-dim (7 arm + 2 gripper fingers). Stats use float64 accumulation.
    """
    episode_id, hdf5_path, task_idx, save_root = args
    with h5py.File(hdf5_path, 'r') as f:
        img1 = f['observation/head/rgb'][()]
        img2 = f['observation/wrist/rgb'][()]
        tac1 = f['tactile/left_gsmini/rgb_marker'][()]
        tac2 = f['tactile/right_gsmini/rgb_marker'][()]
        joint = np.asarray(f['embodiment/joint'][()], dtype=np.float32)

    t = min(len(img1), len(img2), len(tac1), len(tac2), len(joint)) - 1
    episode_dir = Path(save_root) / "data" / str(episode_id).zfill(6)
    for sub in ('img1', 'img2', 'tac1', 'tac2'):
        (episode_dir / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(t):
        frame = f'frame_{str(i).zfill(6)}.jpg'
        (episode_dir / 'img1' / frame).write_bytes(bgr_jpeg_to_rgb_jpeg(img1[i]))
        (episode_dir / 'img2' / frame).write_bytes(bgr_jpeg_to_rgb_jpeg(img2[i]))
        (episode_dir / 'tac1' / frame).write_bytes(bgr_jpeg_to_rgb_jpeg(tac1[i]))
        (episode_dir / 'tac2' / frame).write_bytes(bgr_jpeg_to_rgb_jpeg(tac2[i]))
        rows.append({'state': joint[i], 'action': joint[i + 1], 'task': task_idx})
    pd.DataFrame(rows).to_pickle(episode_dir / 'episode_info.pkl')

    # per-episode stats (state = joint[:t], action = joint[1:t+1]); accumulate in float64
    state = joint[:t].astype(np.float64)
    action = joint[1:t + 1].astype(np.float64)
    return {
        'n': t,
        's_sum': state.sum(axis=0), 's_sumsq': (state * state).sum(axis=0),
        's_min': state.min(axis=0), 's_max': state.max(axis=0),
        'a_sum': action.sum(axis=0), 'a_sumsq': (action * action).sum(axis=0),
        'a_min': action.min(axis=0), 'a_max': action.max(axis=0),
    }


def aggregate_stats(results, dim):
    """Aggregate per-episode stats into dataset-level mean/std/min/max (list of length dim)."""
    n = sum(r['n'] for r in results)
    s_sum = np.zeros(dim, dtype=np.float64); s_sumsq = np.zeros(dim, dtype=np.float64)
    s_min = np.full(dim, np.inf, dtype=np.float64); s_max = np.full(dim, -np.inf, dtype=np.float64)
    a_sum = np.zeros(dim, dtype=np.float64); a_sumsq = np.zeros(dim, dtype=np.float64)
    a_min = np.full(dim, np.inf, dtype=np.float64); a_max = np.full(dim, -np.inf, dtype=np.float64)
    for r in results:
        s_sum += r['s_sum']; s_sumsq += r['s_sumsq']
        s_min = np.minimum(s_min, r['s_min']); s_max = np.maximum(s_max, r['s_max'])
        a_sum += r['a_sum']; a_sumsq += r['a_sumsq']
        a_min = np.minimum(a_min, r['a_min']); a_max = np.maximum(a_max, r['a_max'])
    return {
        'observation_mean': (s_sum / n).tolist(),
        'observation_std': np.sqrt(np.maximum(s_sumsq / n - (s_sum / n) ** 2, 0.0)).tolist(),
        'observation_min': s_min.tolist(), 'observation_max': s_max.tolist(),
        'action_mean': (a_sum / n).tolist(),
        'action_std': np.sqrt(np.maximum(a_sumsq / n - (a_sum / n) ** 2, 0.0)).tolist(),
        'action_min': a_min.tolist(), 'action_max': a_max.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Convert UniVTAC hdf5 dataset to image files and episode pickles.')
    parser.add_argument('--data_path', type=str, default='/home/sunyu/xukun/datasets/univtac',
                        help='Path to the UniVTAC dataset root.')
    parser.add_argument('--save_path', type=str, default='/home/sunyu/xukun/datasets/univtac_converted',
                        help='Output directory for converted dataset.')
    parser.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1),
                        help='Number of parallel worker processes.')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    save_path = Path(args.save_path)

    hdf5_files = sorted(data_path.rglob('*.hdf5'))
    task_names = sorted({p.relative_to(data_path).parts[0] for p in hdf5_files})
    task_name_to_idx = {name: idx for idx, name in enumerate(task_names)}

    # episode-level parallelism: each worker converts one whole episode and returns its stats.
    tasks = [(eid, str(hp), task_name_to_idx[hp.relative_to(data_path).parts[0]], str(save_path))
             for eid, hp in enumerate(hdf5_files)]

    results = []
    with Pool(args.workers) as pool:
        for res in tqdm(pool.imap_unordered(process_episode, tasks), total=len(tasks), desc='Converting UniVTAC'):
            results.append(res)

    save_tasks_json(task_name_to_idx, save_path / 'tasks.json')

    # aggregate state/action statistics (9-dim each) and save to data_statistics.yaml.
    # write by hand so each list stays on one line -- yaml.dump wraps long flow sequences
    # at its hard-coded best_width=80, which makes 9-dim float lists span 3 lines.
    stats = aggregate_stats(results, dim=len(results[0]['s_sum']))
    with open(save_path / 'data_statistics.yaml', 'w') as f:
        f.write('data:\n')
        for key, val in stats.items():
            f.write(f'  {key}: [{", ".join(repr(x) for x in val)}]\n')
    print(f'Saved stats -> {save_path / "data_statistics.yaml"} (n={sum(r["n"] for r in results)} frames)')


if __name__ == '__main__':
    main()
