"""DAgger on the split policy, scored later on the frozen set.

Each iteration rolls the current policy out on 8 new targets that are not in
the frozen test. For the first iterations, half-second blocks still execute
the expert (probability 0.75, then 0.35, then 0). The label on every logged
row is the expert command. The plant follows the block's choice.

The network is warm-started from the previous weights and from the previous
scalers. It is not trained from scratch, and early stopping stays off.

The median error printed here is the error on these 8 rollouts. It is not the
test score. The test score is the frozen set, which this script does not touch.

From the RoboEnv checkout::

    pixi run python -u ../final/robotnik_arm_task/dagger_v2.py
"""
import os

os.environ["ROBOENV_HEADLESS"] = "1"

import argparse
import json
import time

import joblib
import numpy as np

from features_v2 import filter_logged, initial_state, sample_targets
from rollout_v2 import run_episode
from train_v2 import DATA_DIR, ROOT, fit_deployed, gather
import robotnik_arm_task as task


N_ITERS = int(os.environ.get("V2_DAGGER_ITERS", "3"))
N_ROLLOUTS = int(os.environ.get("V2_DAGGER_ROLLOUTS", "8"))
BETA = {1: 0.75, 2: 0.35, 3: 0.0}
DAGGER_EPOCHS = 30


def parse_args():
    parser = argparse.ArgumentParser(description="DAgger for the split policy.")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--out", default=ROOT)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    parser.add_argument("--rollouts", type=int, default=N_ROLLOUTS)
    parser.add_argument("--epochs", type=int, default=DAGGER_EPOCHS)
    return parser.parse_args()


def _beta(iteration):
    if iteration in BETA:
        return BETA[iteration]
    return 0.0


def _log_path(out_dir):
    return os.path.join(out_dir, "dagger", "iterations.json")


def _load_log(out_dir):
    path = _log_path(out_dir)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_log(out_dir, rows):
    path = _log_path(out_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    os.replace(tmp, path)


def _targets(out_dir, iteration, rollouts, origin_xy, avoid):
    path = os.path.join(out_dir, "dagger", f"iter_{iteration:02d}_targets.npy")
    if os.path.isfile(path):
        return np.load(path)
    rng = np.random.default_rng(np.random.SeedSequence([4, iteration]))
    targets = sample_targets(rollouts, origin_xy, rng, avoid=avoid, beyond_frac=0.125)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, targets)
    return targets


def _save_rollout(path, result, target, iteration, beta):
    rows = filter_logged(result["rows"], result["spun"], result["time"])
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            time=rows["time"],
            observation=rows["observation"],
            wheel_cmd=rows["wheel_cmd"],
            torque_arm=rows["torque_arm"],
            wheel_applied=rows["wheel_applied"],
            cartesian_error=rows["cartesian_error"],
            reach_weight=rows["reach_weight"],
            tau_gravity=rows["tau_gravity"],
            ee_dist=rows["ee_dist"],
            target=np.asarray(target, dtype=np.float64),
            reached=np.bool_(result["reached"]),
            spun=np.bool_(result["spun"]),
            unstable=np.bool_(result["unstable"]),
            final_error=np.float64(result["final_error"]),
            final_time=np.float64(result["time"]),
            stop_message=np.asarray(result["stop_message"]),
            iteration=np.int64(iteration),
            beta=np.float64(beta),
            expert_fraction=np.float64(result["expert_fraction"]),
        )
    os.replace(tmp, path)


def _rollout_record(path):
    with np.load(path) as saved:
        return {
            "reached": bool(saved["reached"]),
            "unstable": bool(saved["unstable"]) if "unstable" in saved.files else False,
            "final_error": float(saved["final_error"]),
            "n_rows": int(len(saved["time"])),
            "expert_fraction": float(saved["expert_fraction"]) if "expert_fraction" in saved.files else None,
        }


def _fit(args, iteration, warm_path):
    tag = f"split_iter_{iteration:02d}"
    dest = os.path.join(args.out, f"{tag}.joblib")
    if os.path.isfile(dest):
        print(f"Already trained: {dest}", flush=True)
        return dest
    meta_path = os.path.join(args.data, "dataset_meta.npz")
    with np.load(meta_path) as meta:
        continuous = np.asarray(meta["continuous_joints"], dtype=bool)
        limits = {"wheel_limit": float(meta["wheel_limit"]), "torque_limit": float(meta["torque_limit"])}
        base_z = float(meta["base_z"])
    rows = gather(args.data, iteration, continuous)
    warm = joblib.load(warm_path)
    print(f"Warm-starting {tag} from {warm_path} for {args.epochs} epochs.", flush=True)
    policy = fit_deployed(rows, continuous, limits, base_z, args.epochs, warm)
    policy["tag"] = tag
    policy["iteration"] = int(iteration)
    policy["dagger_through"] = int(iteration)
    policy["beta_collect"] = _beta(iteration)
    joblib.dump(policy, dest)
    print(f"Saved {dest} ({policy['n_params']} parameters, {policy['train_seconds']:.0f}s).", flush=True)
    return dest


def main():
    args = parse_args()
    frozen_path = os.path.join(args.out, "frozen_targets.npz")
    initial = os.path.join(args.out, "split_iter_00.joblib")
    if not os.path.isfile(initial):
        raise SystemExit(f"No behaviour-cloning split policy at {initial}.")
    if not os.path.isfile(frozen_path):
        raise SystemExit(f"No frozen targets at {frozen_path}.")
    with np.load(frozen_path) as frozen:
        avoid = np.vstack((np.asarray(frozen["grid"]), np.asarray(frozen["heldout"])))
        origin_xy = np.asarray(frozen["origin_xy"], dtype=float)
    ctx = task.setup_simulation()
    log = _load_log(args.out)
    try:
        policy_path = initial
        for iteration in range(1, args.iters + 1):
            beta = _beta(iteration)
            targets = _targets(args.out, iteration, args.rollouts, origin_xy, avoid)
            policy = joblib.load(policy_path)
            print(
                f"Iteration {iteration}: beta {beta:.2f}, {len(targets)} rollouts, policy {policy_path}.",
                flush=True,
            )
            folder = os.path.join(args.out, "dagger", f"iter_{iteration:02d}")
            for index, target in enumerate(targets):
                path = os.path.join(folder, f"episode_{index:02d}.npz")
                if os.path.isfile(path):
                    print(f"  [{index + 1}/{len(targets)}] already logged", flush=True)
                    continue
                rng = np.random.default_rng(np.random.SeedSequence([8, iteration, index]))
                initial_pose = initial_state(rng, ctx["q_init_arm"], ctx["arm_lower"], ctx["arm_upper"])
                print(f"  [{index + 1}/{len(targets)}] target {np.round(target, 3)}", flush=True)
                started = time.perf_counter()
                result = run_episode(
                    ctx, target, policy=policy, base="learned", arm="learned",
                    beta=beta, mix_rng=np.random.default_rng(np.random.SeedSequence([9, iteration, index])),
                    initial=initial_pose, spin_qd=8.0, log_every=25,
                )
                _save_rollout(path, result, target, iteration, beta)
                print(
                    f"    {'unstable' if result['unstable'] else ('reached' if result['reached'] else 'missed')}  "
                    f"error {result['final_error']:.3f} m  expert blocks {result['expert_fraction']:.2f}  "
                    f"wall {time.perf_counter() - started:.0f}s",
                    flush=True,
                )
            records = [_rollout_record(os.path.join(folder, f"episode_{index:02d}.npz")) for index in range(len(targets))]
            errors = np.array([row["final_error"] for row in records], dtype=float)
            stable = np.array([not row["unstable"] for row in records], dtype=bool)
            entry = {
                "iteration": iteration,
                "beta": beta,
                "n_rollouts": len(records),
                "n_reached": int(sum(row["reached"] for row in records)),
                "n_unstable": int(sum(row["unstable"] for row in records)),
                "rollout_median_error": None if not np.any(stable) else float(np.median(errors[stable])),
                "note": "rollout_median_error is these 8 targets only. The frozen set is the test.",
                "n_rows": int(sum(row["n_rows"] for row in records)),
            }
            next_path = _fit(args, iteration, policy_path)
            entry["policy"] = os.path.basename(next_path)
            log = [row for row in log if row.get("iteration") != iteration]
            log.append(entry)
            log.sort(key=lambda row: row["iteration"])
            _save_log(args.out, log)
            print(
                f"Iteration {iteration} rollouts: reached {entry['n_reached']}/{entry['n_rollouts']}, "
                f"rollout median {entry['rollout_median_error']}.",
                flush=True,
            )
            policy_path = next_path
    finally:
        ctx["client"].disconnect()


if __name__ == "__main__":
    main()
