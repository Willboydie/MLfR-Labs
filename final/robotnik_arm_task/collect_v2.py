"""Collect expert episodes for the split base/arm policy.

Targets are drawn from the reachable height band, with about one in eight
slightly outside it, and they are kept off the frozen test. A quarter of
episodes start at the default pose. The rest perturb the base and the arm.
DART noise is on: the plant receives a noisy command and the label stays clean.

A spin of joint 1 or 3 aborts the episode. The spinning samples, and the
quarter second before them, are not written. Re-running skips files already
on disk.

From the RoboEnv checkout::

    pixi run python -u ../final/robotnik_arm_task/collect_v2.py --check
    pixi run python -u ../final/robotnik_arm_task/collect_v2.py
"""
import os
import sys

if "--gui" not in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import argparse
import json
import time

import numpy as np

from collect_markov_data import measure_reset, wheel_speed_limit
from features_v2 import (
    BEYOND_FRAC,
    CONTINUOUS_JOINTS,
    TRAIN_SEED,
    beyond_band,
    continuous_mask_from_limits,
    filter_logged,
    initial_state,
    sample_targets,
)
from rollout_v2 import run_episode
import robotnik_arm_task as task


N_EPISODES = int(os.environ.get("V2_EPISODES", "80"))
WHEEL_NOISE_STD = 0.15
TORQUE_NOISE_STD = 1.0
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TASK_DIR, "markov_v2")
DATA_DIR = os.path.join(ROOT, "data")
FROZEN_PATH = os.path.join(ROOT, "frozen_targets.npz")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect cleaned Markov demonstrations.")
    parser.add_argument("--out", default=DATA_DIR)
    parser.add_argument("--frozen", default=FROZEN_PATH)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _write_npz(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(tmp, path)


def ensure_frozen(ctx, path):
    """24-point assessment grid plus 20 held-out targets. Created once."""
    if os.path.isfile(path):
        with np.load(path) as saved:
            return np.asarray(saved["grid"], dtype=float), np.asarray(saved["heldout"], dtype=float), np.asarray(saved["origin_xy"], dtype=float)
    base, _shoulder = measure_reset(ctx)
    grid = np.stack([np.asarray(point, dtype=float) for point in task.sparse_target_grid()])
    if len(grid) != 24:
        raise RuntimeError(f"The assessment grid has {len(grid)} points, expected 24")
    heldout = sample_targets(
        20, base[:2], np.random.default_rng(17), avoid=grid, beyond_frac=0.15,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_npz(path, {
        "grid": grid,
        "heldout": heldout,
        "origin_xy": np.asarray(base[:2], dtype=np.float64),
        "seed": np.int64(17),
    })
    print(
        f"Froze 24 grid points and 20 held-out targets at {path}. "
        f"Held-out z [{heldout[:, 2].min():.2f}, {heldout[:, 2].max():.2f}].",
        flush=True,
    )
    return grid, heldout, np.asarray(base[:2], dtype=float)


def _val_index(n, rng):
    if n < 10:
        return np.zeros(0, dtype=np.int64)
    count = max(1, int(round(0.15 * n)))
    return np.sort(rng.choice(n, size=count, replace=False)).astype(np.int64)


def ensure_spec(ctx, data_dir, frozen, seed, n_episodes, wheel_std, torque_std):
    path = os.path.join(data_dir, "dataset_meta.npz")
    os.makedirs(data_dir, exist_ok=True)
    base, shoulder = measure_reset(ctx)
    expert = task._markov_expert(ctx, np.array([1.0, 0.0, 0.7]))
    continuous = continuous_mask_from_limits(ctx["arm_lower"], ctx["arm_upper"])
    expected = np.zeros(7, dtype=bool)
    expected[list(CONTINUOUS_JOINTS)] = True
    if not np.array_equal(continuous, expected):
        print(
            f"WARNING: unlimited joints are {np.flatnonzero(continuous).tolist()}, "
            f"not {list(CONTINUOUS_JOINTS)}. Encoding follows the simulator.",
            flush=True,
        )
    limits = {
        "wheel_limit": float(wheel_speed_limit(expert.base_params)),
        "torque_limit": float(expert.arm_params.max_torque),
        "z_max": float(expert.arm_params.z_max),
    }
    if os.path.isfile(path):
        with np.load(path) as meta:
            if int(meta["n_episodes"]) != n_episodes or int(meta["seed"]) != seed:
                raise SystemExit(
                    f"{path} was built for {int(meta['n_episodes'])} episodes at seed {int(meta['seed'])}. "
                    f"This run asked for {n_episodes} at seed {seed}."
                )
            if abs(float(meta["wheel_noise_std"]) - wheel_std) > 1e-12 or abs(float(meta["torque_noise_std"]) - torque_std) > 1e-12:
                raise SystemExit(f"{path} was collected at a different DART noise. Use a new --out.")
            print(f"Resuming {n_episodes} episodes from {path}.", flush=True)
            return {key: np.array(meta[key]) for key in meta.files}, limits
    grid, heldout, _origin = frozen
    avoid = np.vstack((grid, heldout))
    rng = np.random.default_rng(seed)
    targets = sample_targets(n_episodes, base[:2], rng, avoid=avoid, beyond_frac=BEYOND_FRAC)
    spec = {
        "targets": targets,
        "val_index": _val_index(n_episodes, rng),
        "seed": np.int64(seed),
        "n_episodes": np.int64(n_episodes),
        "dt": np.float64(ctx["time_step"]),
        "base_z": np.float64(base[2]),
        "base_xy": np.asarray(base[:2], dtype=np.float64),
        "shoulder_z": np.float64(shoulder[2]),
        "continuous_joints": continuous.astype(np.bool_),
        "wheel_noise_std": np.float64(wheel_std),
        "torque_noise_std": np.float64(torque_std),
        "wheel_limit": np.float64(limits["wheel_limit"]),
        "torque_limit": np.float64(limits["torque_limit"]),
        "z_max": np.float64(limits["z_max"]),
    }
    _write_npz(path, spec)
    beyond = int(np.sum(beyond_band(targets[:, 2])))
    print(
        f"Sampled {n_episodes} targets. z [{targets[:, 2].min():.2f}, {targets[:, 2].max():.2f}], "
        f"{beyond} outside [0.25, 1.25]. Val episodes: {len(spec['val_index'])}. "
        f"DART wheels {wheel_std} rad/s, torques {torque_std} N·m.",
        flush=True,
    )
    return spec, limits


def _index_path(data_dir):
    return os.path.join(data_dir, "index.jsonl")


def _load_index(data_dir):
    found = {}
    path = _index_path(data_dir)
    if not os.path.isfile(path):
        return found
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                found[int(row["episode"])] = row
    return found


def _append_index(data_dir, row):
    with open(_index_path(data_dir), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _save_episode(path, result, target, initial, limits_note):
    rows = filter_logged(result["rows"], result["spun"], result["time"])
    n_logged = 0 if result["rows"] is None else int(len(result["rows"]["time"]))
    if rows is None:
        rows = {
            "time": np.zeros(0),
            "observation": np.zeros((0, 22)),
            "wheel_cmd": np.zeros((0, 2)),
            "torque_arm": np.zeros((0, 7)),
            "wheel_applied": np.zeros((0, 4)),
            "cartesian_error": np.zeros((0, 3)),
            "reach_weight": np.zeros(0),
            "tau_gravity": np.zeros((0, 7)),
            "ee_dist": np.zeros(0),
        }
    init_q = np.asarray(task_q if initial is None else initial["q_arm"], dtype=np.float64)
    payload = {
        "time": rows["time"],
        "observation": rows["observation"],
        "wheel_cmd": rows["wheel_cmd"],
        "torque_arm": rows["torque_arm"],
        "wheel_applied": rows["wheel_applied"],
        "cartesian_error": rows["cartesian_error"],
        "reach_weight": rows["reach_weight"],
        "tau_gravity": rows["tau_gravity"],
        "ee_dist": rows["ee_dist"],
        "target": np.asarray(target, dtype=np.float64),
        "reached": np.bool_(result["reached"]),
        "spun": np.bool_(result["spun"]),
        "final_error": np.float64(result["final_error"]),
        "final_time": np.float64(result["time"]),
        "stop_message": np.asarray(result["stop_message"]),
        "randomized": np.bool_(initial is not None),
        "init_dx": np.float64(0.0 if initial is None else initial["dx"]),
        "init_dy": np.float64(0.0 if initial is None else initial["dy"]),
        "init_yaw": np.float64(0.0 if initial is None else initial["yaw"]),
        "init_q_arm": init_q,
        "n_logged": np.int64(n_logged),
        "n_kept": np.int64(len(rows["time"])),
    }
    payload.update(limits_note)
    _write_npz(path, payload)
    return int(len(rows["time"])), n_logged


# Filled in main once the context exists. A canonical episode stores the hold pose.
task_q = None


def _touch_report(data_dir):
    """Rewrite the lecturer report from whatever is already on disk. Never fails the collection."""
    try:
        from progress_report import REPORT_PATH, ROOT, write_report
        root = os.path.dirname(os.path.abspath(data_dir))
        dest = REPORT_PATH if os.path.abspath(root) == os.path.abspath(ROOT) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "progress_report_smoke.md",
        )
        path = write_report(root, dest)
        print(f"Wrote {path}", flush=True)
    except Exception as exc:
        print(f"Report update failed: {exc}", flush=True)


def collect(ctx, args, spec):
    global task_q
    task_q = np.asarray(ctx["q_init_arm"], dtype=np.float64)
    indexed = _load_index(args.out)
    targets = np.asarray(spec["targets"], dtype=float)
    n = len(targets)
    for index in range(n):
        path = os.path.join(args.out, f"episode_{index:03d}.npz")
        if os.path.isfile(path):
            print(f"[{index + 1}/{n}] already logged", flush=True)
            continue
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
        initial = initial_state(rng, ctx["q_init_arm"], ctx["arm_lower"], ctx["arm_upper"])
        target = targets[index]
        where = "canonical" if initial is None else "perturbed"
        print(f"[{index + 1}/{n}] {where}  target {np.round(target, 3)}", flush=True)
        started = time.perf_counter()
        dart = {
            "rng": np.random.default_rng(np.random.SeedSequence([args.seed, index, 1])),
            "wheel_std": float(spec["wheel_noise_std"]),
            "torque_std": float(spec["torque_noise_std"]),
            "wheel_limit": float(spec["wheel_limit"]),
            "torque_limit": float(spec["torque_limit"]),
        }
        result = run_episode(
            ctx, target, base="expert", arm="expert", initial=initial, dart=dart,
            spin_qd=3.0, log_every=1, verbose=False,
        )
        kept, logged = _save_episode(path, result, target, initial, {})
        row = {
            "episode": index,
            "reached": bool(result["reached"]),
            "spun": bool(result["spun"]),
            "randomized": initial is not None,
            "final_error": float(result["final_error"]),
            "final_time": float(result["time"]),
            "stop_message": result["stop_message"],
            "target": target.tolist(),
            "n_logged": logged,
            "n_kept": kept,
            "wall_s": time.perf_counter() - started,
            "base_speed_tail": result["base_speed_tail"],
        }
        _append_index(args.out, row)
        indexed[index] = row
        print(
            f"    {'reached' if result['reached'] else 'missed'}"
            f"{'  spin-cut' if result['spun'] else ''}  "
            f"error {result['final_error']:.4f} m at t={result['time']:.2f}s  "
            f"kept {kept}/{logged}  wall {row['wall_s']:.0f}s",
            flush=True,
        )
        if (index + 1) % 5 == 0 or index + 1 == n:
            _touch_report(args.out)
    return indexed


def main():
    args = parse_args()
    if args.check:
        from features_v2 import check as feature_check
        feature_check()
        return
    ctx = task.setup_simulation()
    try:
        frozen = ensure_frozen(ctx, args.frozen)
        spec, _limits = ensure_spec(
            ctx, args.out, frozen, args.seed, args.episodes, WHEEL_NOISE_STD, TORQUE_NOISE_STD,
        )
        collect(ctx, args, spec)
    finally:
        ctx["client"].disconnect()


if __name__ == "__main__":
    main()
