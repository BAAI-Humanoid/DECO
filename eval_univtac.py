import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from pathlib import Path
from queue import Empty

import torch
import yaml
from isaaclab.app import AppLauncher

from inference import modeling, predict_action


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate policy on UniVTAC tasks.")
    parser.add_argument("--task_name", type=str, default="grasp_classify", help="UniVTAC task module name, e.g. lift_can")
    parser.add_argument("--task_config", type=str, default="contact",
                        help="Task config name or YAML path (resolved relative to <univtac_path>/task_config)")
    parser.add_argument("--univtac_path", type=str, required=True,
                        help="Path to the UniVTAC codebase root (must be given explicitly)")
    parser.add_argument("--model-config", type=str, required=True, help="Policy yaml config path")
    parser.add_argument("--dataset-root", type=str, default="/home/xukun/xukun/IL_training_codebase/config/univtac",
                        help="Dataset root (defaults to env var $UNIVTAC_DATASET, which must be set)")
    parser.add_argument("--total-num", type=int, default=100, help="Number of valid eval episodes (reset failures are re-seeded and not counted)")
    parser.add_argument("--select-action", type=int, default=16, help="Chunk actions consumed per inference (open-loop)")
    parser.add_argument("--action-stride", type=int, default=2,
                        help="Within a chunk, execute every N-th action and discard the rest "
                             "(e.g. 2 = execute frames 0,2,4...). 1 = no skipping (default).")
    parser.add_argument("--open-loop", dest="open_loop", action="store_true", help="Consume multiple actions per inference")
    parser.add_argument("--closed-loop", dest="open_loop", action="store_false", help="One action per inference")
    parser.set_defaults(open_loop=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes. 1 = single-threaded. "
                             "Suggested: 80M model -> 3-4, 0.5B model -> 1")
    # --headless is injected by AppLauncher below; default is False (GUI shown).
    # Pass --headless to run without a GUI window (saves a few hundred MB~1GB/worker,
    # avoids display issues over ssh). Cameras/rendering still work (enable_cameras=True).
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.livestream = 0
    args.num_envs = 1
    if args.dataset_root is None:
        parser.error("--dataset-root not given and env var $UNIVTAC_DATASET is not set. "
                     "Either pass --dataset-root or `export UNIVTAC_DATASET=...`.")
    return args


def load_policy(model_config_path, device):
    """Build the model and load weights.

    Weight loading is delegated to `models.modeling`, which reads
    `pretrain_model_path` / `adapter_model_path` from the yaml's `model:` block and
    applies the correct strategy (vision model, vision-tactile, or adapter base+plugin).
    This handles tactile adapter cases that a generic strict=False load could not.
    Returns (model, yaml_config, weights_path) where weights_path is the resolved
    checkpoint path used for locating the output directory.
    """
    yaml_config = yaml.safe_load(Path(model_config_path).read_text(encoding="utf-8"))
    # modeling() loads weights internally via pretrain_model_path/adapter_model_path.
    model = modeling(yaml_config).to(device).eval()
    weights_path = yaml_config.get("model", {}).get("pretrain_model_path") \
        or yaml_config.get("model", {}).get("adapter_model_path")
    return model, yaml_config, weights_path


def tensor_to_uint8_hwc(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 4:
            x = x[0]
        if x.dtype != torch.uint8:
            x = (x.clamp(0, 1) * 255).to(torch.uint8)
        return x.numpy()
    return x


def parse_observation(obs):
    """Extract head/wrist RGB, joint state, and tactile images from a UniVTAC env observation.
    img and tac from univtac are torch.tensor in range [0,1], to align with training data (using same preprocess func in inference.py)
    need to transfer img and tac to range [0,255]
    """
    img1 = tensor_to_uint8_hwc(obs["observation"]["head"]["rgb"])
    img2 = tensor_to_uint8_hwc(obs["observation"]["wrist"]["rgb"])
    joint = obs["embodiment"]["joint"].detach().cpu().float().flatten().numpy()
    tac1 = tensor_to_uint8_hwc(obs["tactile"]["left_tactile"]["rgb_marker"])
    tac2 = tensor_to_uint8_hwc(obs["tactile"]["right_tactile"]["rgb_marker"])
    return img1, img2, joint, tac1, tac2


def resolve_task_idx(task_name, dataset_root):
    mapping_path = Path(dataset_root) / "task_name_to_idx.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping = {str(k): int(v) for k, v in mapping.items()}
    if task_name in mapping:
        return mapping[task_name]
    normalized = task_name.replace("_", " ").lower().strip()
    for name, idx in mapping.items():
        if name.replace("_", " ").lower().strip() == normalized:
            return idx
    raise KeyError(f"Task '{task_name}' not in {mapping_path}")


def _build_env(args, univtac_path, worker_id=None):
    """Load UniVTAC task config + task module + env. Shared by single-thread and worker paths.

    worker_id, when given, isolates each worker's save_dir to avoid scene/cache contention.
    Returns (task, task_idx, prompt).
    """
    task_cfg_path = Path(args.task_config)
    task_cfg_path = task_cfg_path if task_cfg_path.suffix in (".yml", ".yaml") else \
        Path(univtac_path) / "task_config" / f"{args.task_config}.yml"
    task_cfg = yaml.safe_load(task_cfg_path.read_text(encoding="utf-8"))

    task_module = importlib.import_module(f"envs.{args.task_name}")
    env_cfg = task_module.TaskCfg()
    save_dir = Path(args.weights_path).parent / "eval" / args.task_name / time.strftime("%Y-%m-%d_%H-%M-%S")
    if worker_id is not None:
        save_dir = save_dir / f"worker{worker_id}"
    env_cfg.save_dir = str(save_dir)
    env_cfg.decimation = task_cfg.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_cfg.get("observations", {})
    env_cfg.save_frequency = task_cfg.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_cfg.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_cfg.get("random_texture", False)
    env_cfg.scene.num_envs = 1
    if getattr(args, "device", None):
        env_cfg.sim.device = args.device
    task = task_module.Task(env_cfg, mode="eval")

    task_idx = resolve_task_idx(args.task_name, args.dataset_root)
    tasks_path = Path(args.dataset_root) / "tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    prompt = tasks.get(str(task_idx).zfill(4)) or tasks.get(str(task_idx)) or \
        f"Perform task: {args.task_name.replace('_', ' ')}."

    return task, task_idx, prompt


def _run_episode(task, model, device, yaml_config, args, task_idx, prompt, seed,
                 use_tactile):
    """Run ONE episode for the given seed. Raises on reset failure (caller decides re-seeding).

    Multi-view rollout video is recorded by UniVTAC's own VideoHandler (controlled
    by `video_frequency` in the task config); this function does not save any video.
    Returns dict: {seed, succ, status, reason, steps, actions, diag}.
        reason: "success" | "early_stop" (slipped/dropped) | "timeout" (step_lim reached)
        diag: task-specific values written by check_early_stop/check_success (inhand_bias, min_depth, ...).
    """
    task.mode = "eval"
    task.reset(seed=seed, instructions=[prompt])  # may raise -> treated as re-seed by caller

    action_queue = deque()
    succ = False
    early_stopped = False
    task.mean_steps = task.cfg.step_lim

    while task.take_action_cnt < task.cfg.step_lim:
        obs = task._get_observations()
        img1, img2, state, tac1, tac2 = parse_observation(obs)

        if len(action_queue) == 0:
            tacs = [tac1, tac2] if use_tactile else None
            actions = predict_action(model, device, yaml_config, imgs=[img1, img2], obs=state,
                                     task_idx=task_idx, tacs=tacs)
            consume_n = min(args.select_action, actions.shape[0]) if args.open_loop else 1
            stride = max(1, args.action_stride)
            action_queue.extend(actions[:consume_n:stride].tolist())

        action = torch.tensor(action_queue.popleft(), dtype=torch.float32, device=task._robot_manager.device)
        _, eval_succ = task.take_action(action[:-1], action_type="qpos")
        if eval_succ:
            succ = True
            break
        if task.check_early_stop():
            early_stopped = True
            break

    # Classify failure reason: success / early_stop (slipped/dropped) / timeout (step_lim reached).
    # check_early_stop() writes diagnostic values into task.metadata (e.g. inhand_bias, min_depth, z_dis).
    if succ:
        reason = "success"
    elif early_stopped or task.metadata.get("early_stop"):
        reason = "early_stop"
    else:
        reason = "timeout"
    status = "success" if succ else "failed"

    # Collect diagnostic fields written by check_early_stop / check_success (task-specific).
    diag_keys = ("inhand_bias", "min_depth", "z_dis", "rel_pose")
    diag = {k: task.metadata[k] for k in diag_keys if k in task.metadata}

    return {
        "seed": seed,
        "succ": succ,
        "status": status,
        "reason": reason,
        "steps": getattr(task, "step_count", None),
        "actions": getattr(task, "take_action_cnt", None),
        "diag": diag,
    }


def _write_header(result_file, args):
    result_file.write(
        f"task={args.task_name} weights={args.weights_path} "
        f"model_config={args.model_config} total={args.total_num} "
        f"open_loop={args.open_loop} select_action={args.select_action} workers={args.workers}\n"
    )
    result_file.flush()


def _episode_line(res, done, succ):
    rate = succ / done * 100 if done > 0 else 0.0
    # Keep core fields (seed/status/steps/actions/total) aligned; append reason last.
    line = (f"[seed {res['seed']}] {res['status']} | steps={res['steps']} "
            f"actions={res['actions']} | total {succ}/{done} ({rate:.1f}%)")
    if not res["succ"]:
        diag = res.get("diag") or {}
        diag_str = ",".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in diag.items() if k != "rel_pose")
        suffix = f"reason={res.get('reason', '?')}"
        if diag_str:
            suffix += f" ({diag_str})"
        line += f" [{suffix}]"
    return line


# --------------------------------------------------------------------------- #
# Worker process (top-level so it is picklable for spawn).
# --------------------------------------------------------------------------- #
def _worker_main(args_dict, univtac_path, dataset_root, worker_id, seed_q, result_q):
    """One worker process: launch Isaac Sim, build env, consume seeds until sentinel."""
    # Make UniVTAC importable inside this fresh process.
    sys.path.insert(0, str(Path(univtac_path).parent))
    sys.path.insert(0, str(Path(univtac_path)))

    # Rebuild a lightweight argparse namespace carrying what we need, and launch the app
    # exactly once per worker (must happen before importing sim modules).
    args = argparse.Namespace(**args_dict)
    args.dataset_root = dataset_root

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])
    app_args.enable_cameras = True
    app_args.livestream = 0
    app_args.num_envs = 1
    app_args.device = args.device
    app_args.headless = getattr(args, "headless", False)
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    try:
        model, yaml_config, _ = load_policy(args.model_config,
                                            torch.device(args.device) if args.device else
                                            (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")))
        use_tactile = yaml_config.get("model", {}).get("use_tactile", False)
        device = next(model.parameters()).device

        task, task_idx, prompt = _build_env(args, univtac_path, worker_id=worker_id)
        print(f"[Worker {worker_id}] ready: task={args.task_name} task_idx={task_idx} "
              f"use_tactile={use_tactile}")

        while True:
            try:
                seed = seed_q.get(timeout=1.0)
            except Empty:
                continue
            if seed is None:  # sentinel
                break

            try:
                res = _run_episode(task, model, device, yaml_config, args,
                                   task_idx, prompt, seed, use_tactile)
                task.clean_cache(result=res["status"])
            except Exception as e:
                # reset / runtime failure -> not counted; caller re-seeds.
                task.clean_cache(result="error")
                result_q.put({"worker": worker_id, "seed": seed, "status": "error",
                              "succ": False, "steps": None, "actions": None, "err": str(e)})
                continue

            result_q.put({"worker": worker_id, "seed": res["seed"], "status": res["status"],
                          "succ": res["succ"], "reason": res["reason"], "steps": res["steps"],
                          "actions": res["actions"], "diag": res["diag"], "err": None})
    finally:
        try:
            task.close()
        except Exception:
            pass
        simulation_app.close()


# --------------------------------------------------------------------------- #
# Single-threaded path (workers == 1).
# --------------------------------------------------------------------------- #
def _run_single(args, univtac_path, result_path):
    device = torch.device(args.device) if args.device else \
        (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    model, yaml_config, _ = load_policy(args.model_config, device)
    use_tactile = yaml_config.get("model", {}).get("use_tactile", False)

    task, task_idx, prompt = _build_env(args, univtac_path, worker_id=None)
    print(f"[eval] task={args.task_name} task_idx={task_idx} use_tactile={use_tactile}")
    print(f"[eval] prompt: {prompt}")
    print(f"[eval] result_path: {result_path}")

    result_file = open(result_path, "w", encoding="utf-8")
    _write_header(result_file, args)

    done = succ = 0
    offset = 0
    while done < args.total_num:
        seed = offset  # matches original eval_univtac.py: seed = episode (0, 1, 2, ...)
        offset += 1
        try:
            res = _run_episode(task, model, device, yaml_config, args,
                               task_idx, prompt, seed, use_tactile)
            task.clean_cache(result=res["status"])
        except Exception as e:
            print(f"[seed {seed}] reset/runtime failed: {e}")
            task.clean_cache(result="error")
            continue  # re-seed, not counted

        done += 1
        if res["succ"]:
            succ += 1
        line = _episode_line(res, done, succ)
        print(line)
        result_file.write(line + "\n")
        result_file.flush()

    final_line = f"\n[final] {args.task_name}: {succ}/{args.total_num} ({succ / args.total_num * 100:.1f}%) success"
    print(final_line)
    result_file.write(final_line + "\n")
    result_file.close()
    task.close()


# --------------------------------------------------------------------------- #
# Multi-worker path (workers > 1).
# --------------------------------------------------------------------------- #
def _run_multi(args, univtac_path, result_path):
    mp.set_start_method("spawn", force=True)

    # Seed producer-consumer: prime the queue, main feeds more as results arrive.
    seed_q = mp.Queue()
    result_q = mp.Queue()

    # Keep only picklable, needed fields for the worker namespace.
    worker_args = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "model_config": args.model_config,
        "weights_path": args.weights_path,
        "total_num": args.total_num,
        "select_action": args.select_action,
        "action_stride": args.action_stride,
        "open_loop": args.open_loop,
        "workers": args.workers,
        "device": getattr(args, "device", None),
        "headless": getattr(args, "headless", False),
    }

    workers = []
    for wid in range(args.workers):
        p = mp.Process(target=_worker_main,
                       args=(worker_args, str(univtac_path), args.dataset_root, wid, seed_q, result_q),
                       name=f"Worker-{wid}")
        p.start()
        workers.append(p)

    result_file = open(result_path, "w", encoding="utf-8")
    _write_header(result_file, args)

    # Prime with one seed per worker so they can all start immediately.
    # seed starts from 0 to match original eval_univtac.py (seed = episode).
    next_seed = 0
    for _ in range(args.workers):
        seed_q.put(next_seed)
        next_seed += 1

    done = succ = 0
    try:
        while done < args.total_num and any(p.is_alive() for p in workers):
            try:
                event = result_q.get(timeout=1.0)
            except Empty:
                continue

            if event["status"] == "error":
                print(f"[Worker {event['worker']} seed {event['seed']}] error: {event['err']} | re-seeding")
                # error does not count; feed a replacement seed
                if done < args.total_num:
                    seed_q.put(next_seed)
                    next_seed += 1
                continue

            done += 1
            if event["succ"]:
                succ += 1
            res = {"seed": event["seed"], "status": event["status"], "succ": event["succ"],
                   "reason": event.get("reason", "?"), "diag": event.get("diag", {}),
                   "steps": event["steps"], "actions": event["actions"]}
            line = f"[Worker {event['worker']}] " + _episode_line(res, done, succ)
            print(line)
            result_file.write(line + "\n")
            result_file.flush()

            # Keep the queue fed until target reached.
            if done < args.total_num:
                seed_q.put(next_seed)
                next_seed += 1
    finally:
        # Send stop sentinels.
        for _ in range(args.workers):
            seed_q.put(None)
        for p in workers:
            p.join()
        final_line = f"\n[final] {args.task_name}: {succ}/{args.total_num} ({succ / args.total_num * 100:.1f}%) success"
        print(final_line)
        result_file.write(final_line + "\n")
        result_file.close()


def main():
    args = parse_args()

    univtac_path = Path(args.univtac_path).resolve()
    if not univtac_path.exists():
        raise FileNotFoundError(f"univtac_path does not exist: {univtac_path}")
    sys.path.insert(0, str(univtac_path.parent))
    sys.path.insert(0, str(univtac_path))

    # Resolve the weights path from the yaml (pretrain_model_path / adapter_model_path).
    # Used for locating the output dir, mirroring the old --checkpoint anchor.
    yaml_cfg = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    weights_rel = yaml_cfg.get("model", {}).get("pretrain_model_path") \
        or yaml_cfg.get("model", {}).get("adapter_model_path")
    if not weights_rel:
        raise ValueError("model config must set pretrain_model_path or adapter_model_path "
                         f"(in {args.model_config})")
    args.weights_path = str(Path(weights_rel).resolve())

    # Single-thread path keeps the original in-process Isaac Sim launch (smallest diff).
    if args.workers <= 1:
        global SIMULATION_APP
        SIMULATION_APP = AppLauncher(args).app
        try:
            result_path = Path(args.weights_path).parent / "eval" / args.task_name / "result.txt"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            _run_single(args, univtac_path, result_path)
        finally:
            SIMULATION_APP.close()
        return

    # Multi-worker: app is launched inside each worker process.
    result_path = Path(args.weights_path).parent / "eval" / args.task_name / "result.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _run_multi(args, univtac_path, result_path)


if __name__ == "__main__":
    main()
