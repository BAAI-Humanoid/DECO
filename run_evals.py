#!/usr/bin/env python3
"""Serially evaluate multiple policies over multiple tasks.

For each policy, every task is run (in order); the next policy starts only
after the current policy has finished all its tasks. Each (policy, task) pair
spawns one `eval_univtac.py` subprocess.

Design: a single list of policy dicts (self-contained, no index misalignment),
each carrying its own checkpoint / model-config / per-policy knobs (workers,
select_action, ...). Tasks are a shared list.

Usage:
    python run_evals.py                          # run all POLICIES x TASKS
    python run_evals.py --workers 4              # override workers for all policies
    python run_evals.py --tasks lift_can insert_hole   # subset of tasks
    python run_evals.py --headless               # pass --headless to every eval
    python run_evals.py --dry-run                # print commands without running
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Edit these two blocks.
# ---------------------------------------------------------------------------
# Each policy is a self-contained dict so fields never go out of sync.
# The checkpoint path is read from the yaml's `pretrain_model_path` /
# `adapter_model_path` by modeling(), so no explicit checkpoint field is needed.
POLICIES = [
    {
        "name": "deco_80m",
        "model_config": str(HERE / "config/deco_univtac_80m.yaml"),
        "workers": 1,           
        "select_action": 16,
        "action_stride": 2,
    },
    {
        "name": "deco.p_80m",
        "model_config": str(HERE / "config/deco.p_univtac_80m.yaml"),
        "workers": 1,               
        "select_action": 16,
        "action_stride": 2,
    }
]

# Shared across all policies. Same set of seeds (0..total_num-1) for everyone,
# so results stay comparable across policies.
TASKS = [
    "grasp_classify",
    "lift_can",
    "lift_bottle",
    "pull_out_key",
    "put_bottle_in_shelf",
    "insert_hole",
    "insert_tube",
    "insert_HDMI",
]
# ---------------------------------------------------------------------------

# Fixed kwargs passed to every eval run.
TOTAL_NUM = 100


def build_cmd(policy, task, *, univtac_path, workers_override, headless):
    workers = workers_override if workers_override is not None else policy.get("workers", 1)
    cmd = [
        sys.executable, str(HERE / "eval_univtac.py"),
        "--univtac_path", univtac_path,
        "--model-config", policy["model_config"],
        "--task_name", task,
        "--total-num", str(TOTAL_NUM),
        "--workers", str(workers),
        "--select-action", str(policy.get("select_action", 16)),
        "--action-stride", str(policy.get("action_stride", 1)),
    ]
    if headless:
        cmd.append("--headless")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Serially eval multiple policies x tasks.")
    parser.add_argument("--univtac_path", type=str, required=True,
                        help="Path to the UniVTAC codebase root (passed to every eval_univtac.py run)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override workers for ALL policies (default: use each policy's own value)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Subset of tasks (default: all TASKS)")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Subset of policy names (default: all POLICIES)")
    parser.add_argument("--headless", action="store_true", help="Pass --headless to every eval run")
    parser.add_argument("--total-num", type=int, default=None, help="Override episode count per task")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    global TOTAL_NUM
    if args.total_num is not None:
        TOTAL_NUM = args.total_num

    tasks = args.tasks if args.tasks else TASKS
    policies = [p for p in POLICIES if args.policies is None or p["name"] in args.policies]
    if not policies:
        print(f"[run_evals] no policy matched {args.policies}", file=sys.stderr)
        sys.exit(1)

    total_runs = len(policies) * len(tasks)
    print(f"[run_evals] {len(policies)} policy(ies) x {len(tasks)} task(s) = {total_runs} runs")
    print(f"[run_evals] headless={args.headless} workers_override={args.workers} total_num={TOTAL_NUM}")

    run_idx = 0
    t0 = time.time()
    for policy in policies:
        for task in tasks:
            run_idx += 1
            cmd = build_cmd(policy, task, univtac_path=args.univtac_path,
                            workers_override=args.workers, headless=args.headless)
            header = (f"\n[{run_idx}/{total_runs}] policy={policy['name']} task={task} "
                      f"workers={cmd[cmd.index('--workers')+1]}")
            print(header)
            print("  " + " ".join(cmd))

            if args.dry_run:
                continue

            t1 = time.time()
            try:
                subprocess.run(cmd, check=True, cwd=str(HERE))
            except subprocess.CalledProcessError as e:
                print(f"  ✗ FAILED (exit {e.returncode}) after {time.time()-t1:.0f}s — continuing to next run",
                      file=sys.stderr)
            else:
                print(f"  ✓ done in {time.time()-t1:.0f}s")

    print(f"\n[run_evals] all {run_idx} run(s) finished in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
