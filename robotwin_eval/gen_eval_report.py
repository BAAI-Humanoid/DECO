#!/usr/bin/env python3
"""
Aggregate results from all tasks in an eval_result directory and generate a Markdown table.
Completed tasks show success rates; uncompleted ones are left blank.

Usage:
    python script/gen_eval_report.py /path/to/eval_result_dir
    python script/gen_eval_report.py /path/to/eval_result_dir --policy DECO
    python script/gen_eval_report.py /path/to/eval_result_dir --output report.md
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# ---------- All 50 RoboTwin tasks (same order as 1.md) ----------
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

TASK_CONFIGS = ["demo_clean", "demo_randomized"]
RESULT_PATTERN = "{task_name}/{policy_name}/{task_config}/{ckpt_setting}"


def read_result(task_name, task_config, eval_dir, policy_name="DECO", ckpt_setting="best"):
    """Read success rate for a given (task, config). Returns None if not yet completed."""
    result_dir = os.path.join(
        eval_dir,
        RESULT_PATTERN.format(
            task_name=task_name,
            policy_name=policy_name,
            task_config=task_config,
            ckpt_setting=ckpt_setting,
        )
    )
    if not os.path.isdir(result_dir):
        return None

    # Find all _result.txt files (may exist in multiple timestamp dirs)
    result_files = sorted(glob.glob(os.path.join(result_dir, "*", "_result.txt")))
    if not result_files:
        return None

    # Take the latest one
    latest = result_files[-1]
    with open(latest, "r") as f:
        for line in f:
            line = line.strip().rstrip("#")
            try:
                val = float(line)
                return val  # return as decimal, e.g. 0.99
            except ValueError:
                continue
    return None


def format_rate(val):
    """Convert decimal to percentage string (2 decimal places). None returns empty string."""
    if val is None:
        return ""
    return f"{val * 100:.2f}%"


def compute_average(clean_val, random_val):
    """Compute average (returns decimal ratio). If only one is available, use that one."""
    if clean_val is not None and random_val is not None:
        return (clean_val + random_val) / 2
    elif clean_val is not None:
        return clean_val
    elif random_val is not None:
        return random_val
    else:
        return None


def generate_report(eval_dir, policy_name="DECO", ckpt_setting="best"):
    """Generate Markdown report."""
    results = OrderedDict()
    completed = 0
    total = 0

    for task_name in ALL_TASKS:
        clean_val = read_result(task_name, "demo_clean", eval_dir, policy_name, ckpt_setting)
        random_val = read_result(task_name, "demo_randomized", eval_dir, policy_name, ckpt_setting)

        results[task_name] = {
            "clean": clean_val,
            "random": random_val,
            "avg": compute_average(clean_val, random_val),
        }

        if clean_val is not None:
            completed += 1
        if random_val is not None:
            completed += 1
        total += 2

    # Compute overall averages
    clean_values = [r["clean"] for r in results.values() if r["clean"] is not None]
    random_values = [r["random"] for r in results.values() if r["random"] is not None]
    avg_values = [r["avg"] for r in results.values() if r["avg"] is not None]
    clean_overall = sum(clean_values) / len(clean_values) if clean_values else None
    random_overall = sum(random_values) / len(random_values) if random_values else None
    overall_avg = sum(avg_values) / len(avg_values) if avg_values else None

    # Build Markdown
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# {policy_name} Evaluation Results")
    lines.append("")
    lines.append(f"> Generated at: {now}")
    lines.append("")
    lines.append("| Task | demo_clean | demo_randomized | Average |")
    lines.append("|------|----------:|----------------:|-------:|")

    for task_name, r in results.items():
        clean_str = format_rate(r["clean"])
        random_str = format_rate(r["random"])
        avg_str = format_rate(r["avg"]) if r["avg"] is not None else ""
        lines.append(f"| {task_name} | {clean_str} | {random_str} | {avg_str} |")

    # Overall row (clean avg / random avg / total avg)
    clean_overall_str = format_rate(clean_overall)
    random_overall_str = format_rate(random_overall)
    total_overall_str = format_rate(overall_avg)
    lines.append(f"| **Overall** | {clean_overall_str} | {random_overall_str} | {total_overall_str} |")
    lines.append("")
    lines.append(f"> Results available: **{completed}/{total}**")
    lines.append("")

    return "\n".join(lines), results, completed, total


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate eval_result and generate a Markdown report in the given directory")
    parser.add_argument("--eval_dir", type=str, default='/share/yusun/RoboTwin/eval_result',
                        help="Path to the eval_result directory, e.g. /share/yusun/codes/results/eval_result_deco_seen")
    parser.add_argument("--policy", type=str, default="DECO", help="Policy name (default: DECO)")
    parser.add_argument("--ckpt", type=str, default="best", help="Checkpoint name (default: best)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: <eval_dir>/<policy>_report.md)")
    args = parser.parse_args()

    eval_dir = str(Path(args.eval_dir).resolve())
    
    if not os.path.isdir(eval_dir):
        print(f"Error: eval_result directory does not exist: {eval_dir}")
        sys.exit(1)

    report, results, completed, total = generate_report(eval_dir, args.policy, args.ckpt)

    output = args.output if args.output else os.path.join(eval_dir, f"{args.policy}_report.md")
    with open(output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {output}")

    # Summary statistics
    done_tasks = sum(1 for r in results.values() if r["clean"] is not None and r["random"] is not None)
    partial_tasks = sum(
        1 for r in results.values()
        if (r["clean"] is not None) != (r["random"] is not None)
    )
    missing_tasks = sum(1 for r in results.values() if r["clean"] is None and r["random"] is None)

    print(f"---")
    print(f"Done: {done_tasks}/{len(ALL_TASKS)} tasks (both clean + random)")
    print(f"Partial: {partial_tasks} tasks (only one config has results)")
    print(f"Missing: {missing_tasks} tasks")
    print(f"Data points: {completed}/{total}")


if __name__ == "__main__":
    main()
