#!/usr/bin/env python3
"""
Multi-GPU scheduling script v2: parallel evaluation of 50 RoboTwin tasks x (clean + random).

Improvements (based on FastWAM's manager-worker pattern):
  - Dynamic task queue: as soon as a GPU is free, pick the next task for automatic load balancing
  - subprocess.Popen + real-time streaming output: avoids pipe deadlock caused by capture_output
  - python -u ensures child process output is not buffered
  - Each task runs two phases serially: clean first, then random, on the same GPU
  - Fail fast: if any worker exits abnormally, terminate all workers immediately

Usage:
    python script/eval_all_tasks_v2.py                          # default 8 GPUs
    python script/eval_all_tasks_v2.py --gpus 0,1,2,3           # specify GPUs
    python script/eval_all_tasks_v2.py --gpus 0,1,2,3,4,5,6,7 --policy DECO --max-tasks-per-gpu 1
"""

import os
import sys
import subprocess
import shutil
import glob
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

# ---------- 50 RoboTwin task list ----------
ALL_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
    "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_burger_fries", "place_can_basket", "place_cans_plasticbox", "place_container_plate",
    "place_dual_shoes", "place_empty_cup", "place_fan", "place_mouse_pad",
    "place_object_basket", "place_object_scale", "place_object_stand", "place_phone_stand",
    "place_shoe", "press_stapler", "put_bottles_dustbin", "put_object_cabinet",
    "rotate_qrcode", "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two",
    "stamp_seal", "turn_switch",
]

# ---------- Config ----------
PHASE_CONFIGS = {
    "clean": "demo_clean",
    "random": "demo_randomized",
}

EVAL_RESULT_PATTERN = "eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 3


# ============================================================
# Completed task detection & cleanup
# ============================================================

def check_task_done(task_name, task_config, base_dir, policy_name="DECO_s2", ckpt_setting="best"):
    result_dir = os.path.join(
        base_dir,
        EVAL_RESULT_PATTERN.format(
            task_name=task_name,
            policy_name=policy_name,
            task_config=task_config,
            ckpt_setting=ckpt_setting,
        )
    )
    if not os.path.isdir(result_dir):
        return False
    result_files = glob.glob(os.path.join(result_dir, "*", "_result.txt"))
    return len(result_files) > 0


def clean_incomplete_task(task_name, task_config, base_dir, policy_name="DECO_s2", ckpt_setting="best"):
    result_dir = os.path.join(
        base_dir,
        EVAL_RESULT_PATTERN.format(
            task_name=task_name,
            policy_name=policy_name,
            task_config=task_config,
            ckpt_setting=ckpt_setting,
        )
    )
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir, ignore_errors=True)


def build_task_queue(base_dir, policy_name="DECO_s2", ckpt_setting="best"):
    """
    Returns (pending_phases, skipped_tasks, skipped_phases, already_done_clean, already_done_random).

    - pending_phases: list[(task_name, first_phase)], each entry indicates the phase a task starts from
    - skipped_tasks: number of tasks with both phases already completed
    - skipped_phases: number of phases where clean is done and we jump directly to random
    - already_done_clean: success rates of tasks whose clean phase is completed
    - already_done_random: success rates of tasks whose random phase is completed
    """
    pending = []
    skipped_tasks = 0
    skipped_phases = 0
    already_done_clean = {}
    already_done_random = {}

    for task_name in ALL_TASKS:
        clean_done = check_task_done(task_name, "demo_clean", base_dir, policy_name, ckpt_setting)
        random_done = check_task_done(task_name, "demo_randomized", base_dir, policy_name, ckpt_setting)

        if clean_done and random_done:
            skipped_tasks += 1
            already_done_clean[task_name] = find_result_file(task_name, "demo_clean", base_dir, policy_name, ckpt_setting)
            already_done_random[task_name] = find_result_file(task_name, "demo_randomized", base_dir, policy_name, ckpt_setting)
            continue

        if clean_done and not random_done:
            skipped_phases += 1
            clean_incomplete_task(task_name, "demo_randomized", base_dir, policy_name, ckpt_setting)
            pending.append((task_name, "random"))
            already_done_clean[task_name] = find_result_file(task_name, "demo_clean", base_dir, policy_name, ckpt_setting)
            continue

        clean_incomplete_task(task_name, "demo_clean", base_dir, policy_name, ckpt_setting)
        clean_incomplete_task(task_name, "demo_randomized", base_dir, policy_name, ckpt_setting)
        pending.append((task_name, "clean"))

    return pending, skipped_tasks, skipped_phases, already_done_clean, already_done_random


# ============================================================
# Result parsing
# ============================================================

def find_result_file(task_name, task_config, base_dir, policy_name="DECO_s2", ckpt_setting="best"):
    """Find the _result.txt for the given (task, config) and read the last numeric line as the success rate."""
    result_dir = os.path.join(
        base_dir,
        EVAL_RESULT_PATTERN.format(
            task_name=task_name,
            policy_name=policy_name,
            task_config=task_config,
            ckpt_setting=ckpt_setting,
        )
    )
    result_files = glob.glob(os.path.join(result_dir, "*", "_result.txt"))
    if not result_files:
        return None

    # Take the most recent result file
    result_files.sort(key=os.path.getmtime, reverse=True)
    with open(result_files[0], "r") as f:
        last_value = None
        for line in f:
            stripped = line.strip()
            if stripped == "":
                continue
            try:
                last_value = float(stripped)
            except ValueError:
                continue
    return last_value


# ============================================================
# RunningState
# ============================================================

@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" or "random"
    process: subprocess.Popen


# ============================================================
# Manager
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel evaluation of all RoboTwin tasks on multiple GPUs (v2)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="GPU IDs to use, comma-separated (default: 0,1,2,3,4,5,6,7)")
    parser.add_argument("--policy", type=str, default="DECO",
                        help="Policy name (default: DECO)")
    parser.add_argument("--config", type=str, default="policy/DECO/deploy_policy.yml",
                        help="Path to the deploy policy yml")
    parser.add_argument("--max-tasks-per-gpu", type=int, default=1,
                        help="Max number of tasks running simultaneously per GPU (default: 2)")
    parser.add_argument("--instruction-type", type=str, default="unseen",
                        help="Instruction type: unseen / seen")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    policy_name = args.policy
    deploy_config = args.config or f"policy/{policy_name}/deploy_policy.yml"
    max_tasks_per_gpu = args.max_tasks_per_gpu
    instruction_type = args.instruction_type

    base_dir = Path(__file__).resolve().parent.parent  # RoboTwin root dir

    # Validate that the deploy config exists
    if not (base_dir / deploy_config).exists():
        print(f"Error: deploy config file does not exist: {base_dir / deploy_config}")
        sys.exit(1)

    # Build the pending task queue (phase-level checkpoint resume)
    pending_phases, skipped_tasks, skipped_phases, already_done_clean, already_done_random = \
        build_task_queue(str(base_dir), policy_name, "best")
    total_tasks = len(ALL_TASKS)
    total_phases = total_tasks * 2
    pending_count = len(pending_phases)

    print(f"Total: {total_tasks} tasks (x2 phases = {total_phases})")
    print(f"  Fully completed (skipped): {skipped_tasks} tasks")
    print(f"  Clean done, jumping to random: {skipped_phases} phases")
    print(f"  Pending phases: {pending_count}")
    if pending_count == 0:
        print("\nAll tasks are already completed, nothing to run!")
        return
    print(f"GPU: {gpu_ids}")
    print(f"Policy: {policy_name} | Config: {deploy_config} | instruction: {instruction_type} | max_tasks_per_gpu: {max_tasks_per_gpu}")
    print("=" * 60)

    # Output directory
    run_dir = base_dir / "eval_result" / "_batch_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    manager_log = run_dir / "manager.log"
    summary_csv = run_dir / "summary.csv"

    # ---------- Logging & helper functions ----------
    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = PHASE_CONFIGS[phase]
        return [
            sys.executable,
            "-u",  # unbuffered
            "script/eval_policy.py",
            "--config", deploy_config,
            "--overrides",
            "--task_name", task_name,
            "--task_config", task_config,
            "--instruction_type", instruction_type,
        ]

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name, gpu_id, phase)
        log(f"LAUNCH task={task_name} phase={phase} gpu={gpu_id}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Do not capture stdout/stderr so child output goes directly to the terminal, avoiding pipe buffer deadlock
        process = subprocess.Popen(
            cmd,
            cwd=str(base_dir),
            env=env,
            # stdout/stderr are inherited from the parent process, output goes directly to the terminal
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )

    def terminate_all(running_states: list[RunningState]) -> None:
        """First SIGTERM, then SIGKILL after TERMINATE_TIMEOUT_SEC seconds"""
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"TERMINATE task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()

        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"KILL task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int, running_states: list[RunningState]) -> int:
        return sum(1 for s in running_states if s.gpu_id == gpu_id and s.process.poll() is None)

    def try_launch_pending(gpu_id: int,
                           pending_queue: deque,
                           running_states: list[RunningState]) -> None:
        while (len(pending_queue) > 0
               and gpu_running_count(gpu_id, running_states) < max_tasks_per_gpu):
            task_name, first_phase = pending_queue.popleft()
            running_states.append(launch_phase(task_name, gpu_id, first_phase))

    # ---------- Write summary ----------
    def write_summary(clean_rates: dict, random_rates: dict):
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task_name in ALL_TASKS:
                writer.writerow([
                    task_name,
                    clean_rates.get(task_name),
                    random_rates.get(task_name),
                ])

    # ---------- Main loop ----------
    pending_queue = deque(pending_phases)  # each element is (task_name, first_phase)
    running_states: list[RunningState] = []
    # Preload completed clean results (do not re-run clean on resume)
    clean_rates: dict[str, Optional[float]] = dict(already_done_clean)
    random_rates: dict[str, Optional[float]] = dict(already_done_random)
    failed_tasks: list[dict] = []

    # Fill up each GPU at startup
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id, pending_queue, running_states)

    start_time = datetime.now()
    log(f"START {pending_count} phases ({len(pending_phases)} pending, {skipped_tasks} tasks done, {skipped_phases} phases skipped) on {len(gpu_ids)} GPUs")

    while len(running_states) > 0:
        progressed = False

        for state in list(running_states):
            return_code = state.process.poll()
            if return_code is None:
                continue

            progressed = True
            running_states.remove(state)
            task_name = state.task_name
            phase = state.phase
            gpu_id = state.gpu_id

            if return_code != 0:
                log(f"FAIL task={task_name} phase={phase} gpu={gpu_id} exit_code={return_code}")
                failed_tasks.append({
                    "task_name": task_name,
                    "phase": phase,
                    "gpu_id": gpu_id,
                    "return_code": return_code,
                })
                # Terminate everything on any failure
                terminate_all(running_states)
                running_states.clear()
                break

            # Parse success rate
            task_config = PHASE_CONFIGS[phase]
            rate = find_result_file(task_name, task_config, str(base_dir), policy_name, "best")
            if phase == "clean":
                clean_rates[task_name] = rate
            else:
                random_rates[task_name] = rate

            log(f"DONE  task={task_name} phase={phase} gpu={gpu_id} rate={rate}")

            if phase == "clean":
                # After clean completes, launch the random phase (same GPU)
                running_states.append(launch_phase(task_name, gpu_id, "random"))
            else:
                # After random completes, a GPU slot frees up; try to pick up new tasks
                try_launch_pending(gpu_id, pending_queue, running_states)

        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # ---------- Summary ----------
    end_time = datetime.now()
    elapsed = end_time - start_time

    write_summary(clean_rates, random_rates)

    print("\n" + "=" * 60)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed: {elapsed}")
    print(f"Logs & results: {run_dir}")
    print("=" * 60)

    total_done = len(clean_rates) + len(random_rates)
    total_expected = total_tasks * 2
    print(f"\nCompleted phases: {total_done}/{total_expected} (skipped {skipped_tasks} tasks + {skipped_phases} phases)")

    if clean_rates:
        clean_mean = sum(v for v in clean_rates.values() if v is not None) / len(clean_rates)
        print(f"  Clean average success rate: {clean_mean:.2%} ({len(clean_rates)} tasks)")
    if random_rates:
        random_mean = sum(v for v in random_rates.values() if v is not None) / len(random_rates)
        print(f"  Random average success rate: {random_mean:.2%} ({len(random_rates)} tasks)")

    if failed_tasks:
        print(f"\nFailed ({len(failed_tasks)}):")
        for r in failed_tasks:
            print(f"  GPU {r['gpu_id']}: {r['task_name']} ({r['phase']}) exit={r['return_code']}")

    if pending_queue:
        print(f"\nNot started ({len(pending_queue)}):")
        for task_name, phase in pending_queue:
            print(f"  {task_name} ({phase})")


if __name__ == "__main__":
    main()
