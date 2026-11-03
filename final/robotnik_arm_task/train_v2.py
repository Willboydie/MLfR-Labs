"""Fit the split policy and the single-network comparison on the cleaned rows.

Early stopping is off. A random row holdout inside an episode would share
neighbours 1 ms apart and report an optimistic R². The fit diagnostic here
holds out whole episodes. The deployed weights are a separate fit on every
expert episode plus the closed-form base samples. Closed-loop numbers come
from the frozen test, not from this diagnostic.

From the RoboEnv checkout::

    pixi run python -u ../final/robotnik_arm_task/train_v2.py
"""
import os

os.environ.setdefault("ROBOENV_HEADLESS", "1")

import argparse
import glob
import json
import time

import joblib
import numpy as np

from features_v2 import (
    arm_feature_matrix,
    arm_feature_names,
    base_feature_matrix,
    keep_mask,
    sample_weights,
    standoff_for_height,
    synthetic_base_samples,
)
from policy_v2 import fit_mono, fit_split, regression_rmse


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TASK_DIR, "markov_v2")
DATA_DIR = os.path.join(ROOT, "data")
STRIDE_S = 0.025
SYNTHETIC = (4000, 1500, 3000)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the split and monolithic policies.")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--out", default=ROOT)
    parser.add_argument("--dagger-through", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--warm", default="", help="Split policy to continue from.")
    parser.add_argument("--tag", default="split_iter_00")
    parser.add_argument("--set-winner", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain even if the output file exists.")
    return parser.parse_args()


def _stride_indices(time, mask):
    time = np.asarray(time, dtype=np.float64)
    if len(time) == 0:
        return np.zeros(0, dtype=int)
    period = float(np.median(np.diff(time))) if len(time) > 1 else STRIDE_S
    stride = max(1, int(round(STRIDE_S / period)))
    chosen = np.arange(0, len(time), stride)
    return chosen[mask[chosen]]


def _load_episode(path, episode_index):
    with np.load(path) as episode:
        if "observation" not in episode.files or len(episode["observation"]) == 0:
            return None
        time = np.asarray(episode["time"], dtype=np.float64)
        obs = np.asarray(episode["observation"], dtype=np.float64)
        wheels = np.asarray(episode["wheel_cmd"], dtype=np.float64)
        torque = np.asarray(episode["torque_arm"], dtype=np.float64)
        error = np.asarray(episode["cartesian_error"], dtype=np.float64)
        reach = np.asarray(episode["reach_weight"], dtype=np.float64)
        tau = np.asarray(episode["tau_gravity"], dtype=np.float64)
        target_z = float(np.asarray(episode["target"])[2])
    if obs.ndim != 2 or obs.shape[1] != 22:
        raise SystemExit(f"{path} observation width is {getattr(obs, 'shape', None)}")
    finite = np.isfinite(obs).all(axis=1) & np.isfinite(wheels).all(axis=1) & np.isfinite(torque).all(axis=1)
    period = float(np.median(np.diff(time))) if len(time) > 1 else STRIDE_S
    mask = keep_mask(obs[:, 5:12], obs[:, 12:19], period) & finite
    chosen = _stride_indices(time, mask)
    if len(chosen) == 0:
        return None
    return {
        "episode": int(episode_index),
        "obs": obs[chosen],
        "wheels": wheels[chosen],
        "torque": torque[chosen],
        "error": error[chosen],
        "reach": reach[chosen],
        "tau": tau[chosen],
        "target_z": target_z,
    }


def iter_paths(data_dir, dagger_through):
    expert = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    dagger = []
    dagger_root = os.path.join(os.path.dirname(data_dir), "dagger")
    if os.path.basename(data_dir) != "data":
        dagger_root = os.path.join(data_dir, "dagger")
    for iteration in range(1, dagger_through + 1):
        folder = os.path.join(dagger_root, f"iter_{iteration:02d}")
        dagger.extend(sorted(glob.glob(os.path.join(folder, "episode_*.npz"))))
    return expert, dagger


def gather(data_dir, dagger_through, continuous):
    expert_paths, dagger_paths = iter_paths(data_dir, dagger_through)
    if not expert_paths:
        raise SystemExit(f"No episodes in {data_dir}.")
    pieces = []
    for path in expert_paths:
        index = int(os.path.basename(path)[len("episode_"):-len(".npz")])
        loaded = _load_episode(path, index)
        if loaded is not None:
            pieces.append(loaded)
    for path in dagger_paths:
        loaded = _load_episode(path, -1)
        if loaded is not None:
            pieces.append(loaded)
    if not pieces:
        raise SystemExit("Every episode was empty after the spin filter.")
    obs = np.vstack([piece["obs"] for piece in pieces])
    wheels = np.vstack([piece["wheels"] for piece in pieces])
    torque = np.vstack([piece["torque"] for piece in pieces])
    error = np.vstack([piece["error"] for piece in pieces])
    reach = np.concatenate([piece["reach"] for piece in pieces])
    tau = np.vstack([piece["tau"] for piece in pieces])
    episode = np.concatenate([np.full(len(piece["obs"]), piece["episode"], dtype=np.int64) for piece in pieces])
    target_z = np.concatenate([np.full(len(piece["obs"]), piece["target_z"], dtype=np.float64) for piece in pieces])
    standoff = np.array([standoff_for_height(z) for z in target_z], dtype=np.float64)
    base_x = base_feature_matrix(obs[:, 0:3], standoff)
    arm_x = arm_feature_matrix(
        obs[:, 0:3], obs[:, 3:5], obs[:, 5:12], obs[:, 12:19], obs[:, 19:22],
        error, reach, tau, continuous,
    )
    weights = sample_weights(wheels, reach, error)
    print(
        f"Training rows: {len(obs)} at {STRIDE_S * 1000:.0f} ms "
        f"from {len(expert_paths)} expert files and {len(dagger_paths)} DAgger files.",
        flush=True,
    )
    return {
        "episode": episode,
        "base_x": base_x,
        "base_y": wheels,
        "arm_x": arm_x,
        "residual": torque - tau,
        "full": np.hstack((wheels, torque)),
        "weights": weights,
        "n_expert_files": len(expert_paths),
        "n_dagger_files": len(dagger_paths),
    }


def _score_split(policy, rows, mask, label):
    if not np.any(mask):
        return None
    base = regression_rmse(policy, rows["base_x"][mask], rows["base_y"][mask], "base")
    arm = regression_rmse(policy, rows["arm_x"][mask], rows["residual"][mask], "arm")
    probability = policy["park_model"].predict_proba(
        policy["park_scaler"].transform(rows["base_x"][mask])
    )[:, 1]
    prediction = policy["base_y_scaler"].inverse_transform(
        policy["base_model"].predict(policy["base_x_scaler"].transform(rows["base_x"][mask]))
    )
    prediction = np.asarray(prediction, dtype=np.float64)
    small = np.max(np.abs(prediction), axis=1) < 0.02
    fire = (probability >= policy["park_threshold"]) | ((probability >= 0.35) & small)
    prediction[fire] = 0.0
    truth = rows["base_y"][mask]
    dead = np.max(np.abs(truth), axis=1) < 1e-8
    moving = ~dead
    def _rmse(pred, true):
        if len(true) == 0:
            return None
        return float(np.sqrt(np.mean((pred - true) ** 2)))
    return {
        "label": label,
        "n": int(np.sum(mask)),
        "wheel_rmse": float(np.sqrt(np.mean(base ** 2))),
        "wheel_rmse_per_output": base.tolist(),
        "arm_residual_rmse": float(np.sqrt(np.mean(arm ** 2))),
        "arm_residual_rmse_per_joint": arm.tolist(),
        "gated_wheel_rmse": _rmse(prediction, truth),
        "gated_deadzone_rmse": _rmse(prediction[dead], truth[dead]) if np.any(dead) else None,
        "gated_moving_rmse": _rmse(prediction[moving], truth[moving]) if np.any(moving) else None,
        "deadzone_fraction": float(np.mean(dead)),
        "park_recall_on_rows": float(np.mean(probability[dead] >= policy["park_threshold"])) if np.any(dead) else None,
        "park_false_on_moving": float(np.mean(probability[moving] >= policy["park_threshold"])) if np.any(moving) else None,
    }


def _with_synthetic(rows, base_z, rng):
    features, labels = synthetic_base_samples(*SYNTHETIC, base_z=base_z, rng=rng)
    weights = np.ones(len(features), dtype=np.float64)
    weights[:SYNTHETIC[0]] = 4.0
    weights[SYNTHETIC[0]:SYNTHETIC[0] + SYNTHETIC[1]] = 2.0
    return (
        np.vstack((rows["base_x"], features)),
        np.vstack((rows["base_y"], labels)),
        np.concatenate((rows["weights"], weights)),
        int(len(features)),
    )


def fit_deployed(rows, continuous, limits, base_z, max_iter, warm_policy):
    rng = np.random.default_rng(0)
    base_x, base_y, base_w, n_synthetic = _with_synthetic(rows, base_z, rng)
    started = time.perf_counter()
    policy = fit_split(
        base_x, base_y, base_w,
        rows["arm_x"], rows["residual"], rows["weights"],
        continuous, limits, max_iter, warm=warm_policy,
    )
    policy["train_seconds"] = time.perf_counter() - started
    policy["n_synthetic_base"] = n_synthetic
    policy["feature_names_arm"] = arm_feature_names(continuous)
    return policy


def main():
    args = parse_args()
    meta_path = os.path.join(args.data, "dataset_meta.npz")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"No dataset metadata at {meta_path}.")
    with np.load(meta_path) as meta:
        continuous = np.asarray(meta["continuous_joints"], dtype=bool)
        limits = {"wheel_limit": float(meta["wheel_limit"]), "torque_limit": float(meta["torque_limit"])}
        base_z = float(meta["base_z"])
        val_index = np.asarray(meta["val_index"], dtype=np.int64)
    split_path = os.path.join(args.out, f"{args.tag}.joblib")
    mono_path = os.path.join(args.out, "mono.joblib")
    mono_needed = (not args.warm) and args.dagger_through == 0
    if os.path.isfile(split_path) and not args.force and (not mono_needed or os.path.isfile(mono_path)):
        print(f"Already trained: {split_path}", flush=True)
        if args.set_winner:
            winner = os.path.join(args.out, "winner.joblib")
            if not os.path.isfile(winner):
                joblib.dump(joblib.load(split_path), winner)
                print(f"Saved {winner}", flush=True)
        return
    if os.path.isfile(split_path) and not args.force:
        print(f"Split weights exist; fitting the missing monolithic network.", flush=True)
        rows = gather(args.data, args.dagger_through, continuous)
        started = time.perf_counter()
        mono = fit_mono(rows["arm_x"], rows["full"], rows["weights"], continuous, limits, args.max_iter)
        mono["train_seconds"] = time.perf_counter() - started
        mono["tag"] = "mono"
        joblib.dump(mono, mono_path)
        print(f"Saved {mono_path} ({mono['n_params']} parameters).", flush=True)
        return
    rows = gather(args.data, args.dagger_through, continuous)
    os.makedirs(args.out, exist_ok=True)
    warm = joblib.load(args.warm) if args.warm else None
    if warm is not None and warm.get("kind") != "split":
        raise SystemExit(f"--warm file is {warm.get('kind')!r}, not a split policy.")
    print(f"Fitting {args.tag} (max_iter {args.max_iter}, warm_start {warm is not None}).", flush=True)
    policy = fit_deployed(rows, continuous, limits, base_z, args.max_iter, warm)
    policy["train_seconds"] = policy.get("train_seconds", 0.0)
    policy["tag"] = args.tag
    policy["dagger_through"] = int(args.dagger_through)
    split_path = os.path.join(args.out, f"{args.tag}.joblib")
    joblib.dump(policy, split_path)
    print(f"Saved {split_path} ({policy['n_params']} parameters, {policy['train_seconds']:.0f}s).", flush=True)

    summary = {
        "tag": args.tag,
        "dagger_through": int(args.dagger_through),
        "n_rows": int(len(rows["episode"])),
        "n_expert_files": rows["n_expert_files"],
        "n_dagger_files": rows["n_dagger_files"],
        "n_synthetic_base": int(policy["n_synthetic_base"]),
        "n_params_split": int(policy["n_params"]),
        "base_n_iter": int(policy["base_n_iter"]),
        "arm_n_iter": int(policy["arm_n_iter"]),
        "train_seconds": policy["train_seconds"],
        "park_stats": policy["park_stats"],
        "early_stopping": False,
        "max_iter": int(args.max_iter),
        "warm_start": warm is not None,
        "arm_inputs": int(rows["arm_x"].shape[1]),
        "base_inputs": int(rows["base_x"].shape[1]),
        "continuous_joints": np.flatnonzero(continuous).tolist(),
    }
    if warm is None and len(val_index):
        val = np.isin(rows["episode"], val_index)
        train = ~val
        print(f"Episode holdout: {int(val.sum())} val rows, {int(train.sum())} train rows.", flush=True)
        held_rows = {
            "base_x": rows["base_x"][train],
            "base_y": rows["base_y"][train],
            "arm_x": rows["arm_x"][train],
            "residual": rows["residual"][train],
            "weights": rows["weights"][train],
        }
        holdout = fit_deployed(held_rows, continuous, limits, base_z, args.max_iter, None)
        summary["holdout"] = _score_split(holdout, rows, val, "episode holdout, not used for the deployed weights")
        summary["holdout_episodes"] = val_index.tolist()
        if summary["holdout"] is None:
            print("Episode holdout had no rows after filtering.", flush=True)
        else:
            print(
                f"Holdout wheel RMSE {summary['holdout']['gated_wheel_rmse']:.3f} rad/s, "
                f"arm residual RMSE {summary['holdout']['arm_residual_rmse']:.3f} N·m.",
                flush=True,
            )
        print("Fitting the monolithic network on the same rows.", flush=True)
        started = time.perf_counter()
        mono = fit_mono(rows["arm_x"], rows["full"], rows["weights"], continuous, limits, args.max_iter)
        mono["train_seconds"] = time.perf_counter() - started
        mono["tag"] = "mono"
        mono_path = os.path.join(args.out, "mono.joblib")
        joblib.dump(mono, mono_path)
        summary["n_params_mono"] = int(mono["n_params"])
        summary["mono_seconds"] = mono["train_seconds"]
        summary["mono_n_iter"] = int(mono["n_iter"])
        print(f"Saved {mono_path} ({mono['n_params']} parameters, {mono['train_seconds']:.0f}s).", flush=True)
    elif warm is None and args.dagger_through == 0 and not os.path.isfile(os.path.join(args.out, "mono.joblib")):
        print("Fitting the monolithic network on the same rows.", flush=True)
        started = time.perf_counter()
        mono = fit_mono(rows["arm_x"], rows["full"], rows["weights"], continuous, limits, args.max_iter)
        mono["train_seconds"] = time.perf_counter() - started
        mono["tag"] = "mono"
        mono_path = os.path.join(args.out, "mono.joblib")
        joblib.dump(mono, mono_path)
        summary["n_params_mono"] = int(mono["n_params"])
        summary["mono_seconds"] = mono["train_seconds"]
        summary["mono_n_iter"] = int(mono["n_iter"])
        print(f"Saved {mono_path} ({mono['n_params']} parameters, {mono['train_seconds']:.0f}s).", flush=True)
    summary_path = os.path.join(args.out, f"{args.tag}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if args.set_winner:
        winner = os.path.join(args.out, "winner.joblib")
        if not os.path.isfile(winner):
            joblib.dump(policy, winner)
            with open(os.path.join(args.out, "winner_selection.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "tag": args.tag,
                    "controller": args.tag,
                    "source": f"{args.tag}.joblib",
                    "reason": "First split policy. A later iteration replaces it only if it wins on the full frozen set.",
                    "rule": "Provisional, before any frozen-set comparison.",
                }, handle, indent=2)
            print(f"Saved {winner}", flush=True)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
