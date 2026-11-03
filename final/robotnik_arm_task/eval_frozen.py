"""Frozen test: the original 24-point grid plus 20 held-out targets.

Every controller is scored on that same list. A run that leaves the workspace
or winds a continuous joint is a divergence, not a contribution to the mean
error. Success at 10, 25, 50 and 100 mm is the final true error, and the
expert's own dwell stop is reported beside the 10 mm figure.

The 2×2 swap is expert or learned base, crossed with expert or learned arm.
Robustness is observation noise, a short push, and a heavier end effector,
on the eight grid points at z = 0.70 m.

From the RoboEnv checkout::

    pixi run python -u ../final/robotnik_arm_task/eval_frozen.py --controller markov
    pixi run python -u ../final/robotnik_arm_task/eval_frozen.py --controller policy --name split-iter-00 --policy ../final/robotnik_arm_task/markov_v2/split_iter_00.joblib
"""
import os

os.environ["ROBOENV_HEADLESS"] = "1"

import argparse
import json

import joblib
import numpy as np

from policy_v2 import time_predict
from rollout_v2 import TOLERANCES_M, _measure, run_episode, run_phased
from robotnik_markov_expert import relative_target
import robotnik_arm_task as task


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TASK_DIR, "markov_v2")
FROZEN_PATH = os.path.join(ROOT, "frozen_targets.npz")
TOL_MM = ((10, 0.010), (25, 0.025), (50, 0.050), (100, 0.100))


def parse_args():
    parser = argparse.ArgumentParser(description="Score controllers on the frozen targets.")
    parser.add_argument("--frozen", default=FROZEN_PATH)
    parser.add_argument("--out", default=ROOT)
    parser.add_argument("--controller", default="markov",
                        choices=("markov", "phased", "policy", "swap-expert-base", "swap-learned-base"))
    parser.add_argument("--name", default="", help="Name written into the results. Defaults to the controller.")
    parser.add_argument("--policy", default="")
    parser.add_argument("--sets", default="", help="Comma-separated subset of grid,heldout. Phased defaults to grid.")
    parser.add_argument("--robustness", action="store_true", help="Noise, push, and payload on the z=0.70 grid.")
    parser.add_argument("--time", action="store_true", help="Time one policy step against one expert call.")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N selected targets.")
    return parser.parse_args()


def load_frozen(path):
    if not os.path.isfile(path):
        raise SystemExit(f"No frozen targets at {path}. Run collect_v2.py first.")
    with np.load(path) as saved:
        grid = np.asarray(saved["grid"], dtype=float)
        heldout = np.asarray(saved["heldout"], dtype=float)
    if len(grid) != 24 or len(heldout) != 20:
        raise SystemExit(f"Frozen set is {len(grid)} + {len(heldout)}, expected 24 + 20.")
    return grid, heldout


def result_path(out_dir):
    return os.path.join(out_dir, "results.jsonl")


def load_records(path):
    keyed = {}
    if not os.path.isfile(path):
        return keyed
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keyed[(row["controller"], row["set"], int(row["index"]))] = row
    return keyed


def append_record(path, row):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _json_time(result):
    raw = result.get("time_to_m") or {}
    out = {}
    for mm, tol in TOL_MM:
        out[str(mm)] = raw.get(str(tol))
    return out


def make_record(name, set_name, index, target, result):
    unstable = bool(result["unstable"])
    final = float(result["final_error"])
    within = {str(mm): bool((not unstable) and final <= tol) for mm, tol in TOL_MM}
    return {
        "controller": name,
        "set": set_name,
        "index": int(index),
        "target": np.asarray(target, dtype=float).tolist(),
        "final_error": final,
        "min_error": result["min_error"],
        "time": float(result["time"]),
        "stop_message": result["stop_message"],
        "reached": bool(result["reached"]),
        "diverged": bool(result["diverged"]),
        "spun": bool(result["spun"]),
        "unstable": unstable,
        "within": within,
        "time_to_mm": _json_time(result),
        "base_speed_tail": result["base_speed_tail"],
        "wheel_abs_tail": result["wheel_abs_tail"],
        "qd_abs_tail": result["qd_abs_tail"],
        "final_ee_pos": np.asarray(result["final_ee_pos"], dtype=float).tolist(),
        "final_base_xy": np.asarray(result["final_base_xy"], dtype=float).tolist(),
        "expert_fraction": result.get("expert_fraction"),
        "payload_link": result.get("payload_link"),
    }


def _median(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return None
    return float(np.median(values))


def _mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return None
    return float(np.mean(values))


def summarise(records):
    """One row per controller. Means and medians omit unstable runs."""
    by_name = {}
    for record in records:
        by_name.setdefault(record["controller"], []).append(record)
    rows = []
    for name in sorted(by_name):
        items = by_name[name]
        stable = [item for item in items if not item["unstable"]]
        reached = [item for item in items if item["reached"]]
        row = {
            "controller": name,
            "n": len(items),
            "n_grid": sum(item["set"] == "grid" for item in items),
            "n_heldout": sum(item["set"] == "heldout" for item in items),
            "n_reached": len(reached),
            "n_unstable": sum(item["unstable"] for item in items),
            "n_diverged": sum(item["diverged"] for item in items),
            "n_spun": sum(item["spun"] for item in items),
            "within": {},
            "median_error": _median([item["final_error"] for item in stable]),
            "mean_error": _mean([item["final_error"] for item in stable]),
            "median_time_reached": _median([item["time"] for item in reached]),
            "median_base_speed_cm_s": None if _median([item["base_speed_tail"] for item in stable]) is None
            else _median([item["base_speed_tail"] for item in stable]) * 100.0,
            "median_wheel_abs_tail": _median([item["wheel_abs_tail"] for item in stable]),
            "median_qd_abs_tail": _median([item["qd_abs_tail"] for item in stable]),
        }
        for mm, _tol in TOL_MM:
            hits = [item for item in items if item["within"][str(mm)]]
            row["within"][str(mm)] = len(hits)
            times = [item["time_to_mm"].get(str(mm)) for item in hits]
            row[f"median_time_to_{mm}"] = _median(times)
        rows.append(row)
    return rows


def print_summary(rows):
    print(
        f"\n{'controller':<22} {'reach':>7} {'10mm':>7} {'25mm':>7} {'50mm':>7} {'100mm':>7} "
        f"{'unstable':>9} {'median m':>9}",
        flush=True,
    )
    for row in rows:
        def rate(key):
            return f"{row['within'][key]}/{row['n']}"
        median = "n/a" if row["median_error"] is None else f"{row['median_error']:.3f}"
        print(
            f"{row['controller']:<22} {row['n_reached']:>3}/{row['n']:<3} {rate('10'):>7} {rate('25'):>7} "
            f"{rate('50'):>7} {rate('100'):>7} {row['n_unstable']:>9} {median:>9}",
            flush=True,
        )


def _run_one(ctx, policy, kind, target, noise=None, push=None, payload=0.0):
    if kind == "phased":
        return run_phased(ctx, target)
    if kind == "markov":
        return run_episode(ctx, target, base="expert", arm="expert", spin_qd=8.0)
    if kind == "policy":
        return run_episode(ctx, target, policy=policy, base="learned", arm="learned", spin_qd=8.0,
                           obs_noise=noise, push=push, payload_kg=payload)
    if kind == "swap-expert-base":
        return run_episode(ctx, target, policy=policy, base="expert", arm="learned", spin_qd=8.0)
    if kind == "swap-learned-base":
        return run_episode(ctx, target, policy=policy, base="learned", arm="expert", spin_qd=8.0)
    raise ValueError(kind)


def _targets_for(grid, heldout, sets):
    chosen = []
    if "grid" in sets:
        chosen.extend(("grid", index, point) for index, point in enumerate(grid))
    if "heldout" in sets:
        chosen.extend(("heldout", index, point) for index, point in enumerate(heldout))
    return chosen


def _robust_targets(grid):
    chosen = [(index, point) for index, point in enumerate(grid) if abs(point[2] - 0.70) < 1e-9]
    if len(chosen) != 8:
        raise RuntimeError(f"Expected 8 grid points at z=0.70, found {len(chosen)}")
    return chosen


def benchmark(ctx, policy, out_dir):
    target = np.array([1.0, 0.2, 0.70])
    from features_v2 import apply_initial_state
    apply_initial_state(ctx, None)
    base_pos, _ori, bearing, q_mes, qd_mes, q_pin, _ee = _measure(ctx)
    expert = task._markov_expert(ctx, target)
    timing = time_predict(
        policy, expert, q_pin, q_mes[4:11], qd_mes[4:11],
        relative_target(target, base_pos, bearing), np.zeros(3), repeats=100,
    )
    timing["note"] = (
        "Median of single-sample calls after warmup. The policy time includes the "
        "hand-coded cartesian error, reach weight, and gravity torque. The expert time "
        "is one full expert call, including inverse kinematics."
    )
    path = os.path.join(out_dir, "timing.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(timing, handle, indent=2)
    print(
        f"Step time: policy {timing['policy_ms']:.3f} ms, expert {timing['expert_ms']:.3f} ms, "
        f"{timing['n_params']} parameters. Saved {path}",
        flush=True,
    )


def run_robustness(ctx, policy, name, grid, out_dir):
    path = result_path(out_dir)
    done = load_records(path)
    conditions = {
        f"{name}-noise": {"noise": {"q": 0.02, "qd": 0.05, "xy": 0.01, "yaw": 0.02}},
        f"{name}-push": {"push": (2.0, 2.5, (50.0, 0.0, 0.0))},
        f"{name}-payload": {"payload": 0.5},
    }
    targets = _robust_targets(grid)
    for condition, template in conditions.items():
        for index, target in targets:
            key = (condition, "grid-z070", index)
            if key in done and "final_error" in done[key]:
                print(f"[{condition} {index + 1}/8] already scored", flush=True)
                continue
            print(f"[{condition} {index + 1}/8] target {np.round(target, 3)}", flush=True)
            options = dict(template)
            if "noise" in options:
                noise = dict(options["noise"])
                noise["rng"] = np.random.default_rng(np.random.SeedSequence([0, index]))
                options["noise"] = noise
            result = _run_one(ctx, policy, "policy", target, **options)
            row = make_record(condition, "grid-z070", index, target, result)
            append_record(path, row)
            done[key] = row
            print(
                f"    {'reached' if row['reached'] else 'missed'}  "
                f"error {row['final_error']:.4f} m  unstable {row['unstable']}",
                flush=True,
            )


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    grid, heldout = load_frozen(args.frozen)
    name = args.name or args.controller
    needs_policy = args.controller not in ("markov", "phased") or args.robustness or args.time
    policy = None
    if needs_policy:
        if not args.policy:
            raise SystemExit("--policy is required for this controller.")
        policy = joblib.load(args.policy)
    sets = tuple(part for part in args.sets.split(",") if part)
    if not sets:
        sets = ("grid",) if args.controller == "phased" else ("grid", "heldout")
    ctx = task.setup_simulation()
    try:
        if args.time:
            benchmark(ctx, policy, args.out)
        if args.robustness:
            run_robustness(ctx, policy, name, grid, args.out)
        else:
            path = result_path(args.out)
            done = load_records(path)
            planned = _targets_for(grid, heldout, sets)
            if args.limit:
                planned = planned[:args.limit]
            for set_name, index, target in planned:
                key = (name, set_name, index)
                if key in done and "final_error" in done[key]:
                    continue
                print(f"[{name} {set_name} {index + 1}] target {np.round(target, 3)}", flush=True)
                result = _run_one(ctx, policy, args.controller, target)
                row = make_record(name, set_name, index, target, result)
                append_record(path, row)
                done[key] = row
                flag = "unstable" if row["unstable"] else ("reached" if row["reached"] else "missed")
                print(f"    {flag}  error {row['final_error']:.4f} m at t={row['time']:.2f}s", flush=True)
        records = list(load_records(result_path(args.out)).values())
        rows = summarise(records)
        summary_path = os.path.join(args.out, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print_summary(rows)
        print(f"Saved {summary_path}", flush=True)
    finally:
        ctx["client"].disconnect()


if __name__ == "__main__":
    main()
