"""5-fold MLP search on the full Markov log and on the successful episodes only.

The split, the shuffle, the 25 ms stride, and the standardised-MSE score match
``cv_markov_mlp.py``. Training size is still a factor: 25 episodes, 55, and
every episode in the training fold. On 150 episodes that last setting is 120.
On the success-only set it is the fold's own pool (71 or 72).

The first search already moved width, depth, activation, and one harsh L2
value. It left the Adam step size at 1e-3 and the batch size at 256, and it
never moved L2 on the tanh network that won. Those three are the next factors.
Each is changed on its own, around tanh-128x128, and scored in the same table
as the original six networks:

* learning rate 1e-4 and 1e-2 (1e-3 is the centre)
* L2 penalty 1e-5 and 1e-3 (1e-4 is the centre)
* batch size 64 and 1024 (256 is the centre)

From the RoboEnv checkout::

    pixi run python -u ../final/robotnik_arm_task/cv_dataset_mlp.py --dry-run
    pixi run python -u ../final/robotnik_arm_task/cv_dataset_mlp.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from cv_markov_mlp import (
    CONFIGS,
    MAX_ITER,
    N_FOLDS,
    N_ITER_NO_CHANGE,
    SAMPLE_STRIDE_S,
    SEED,
    score_fold,
    shuffle_rows,
    stack,
)

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(TASK_DIR, "markov_data")
DATASET_DIR = os.path.join(TASK_DIR, "markov_datasets")
N_SOURCE = 150
SIZE_LEVELS = ("25", "55", "all")
SIZE_SEED = {"25": 25, "55": 55, "all": 1_000_000}

_EPISODES = None


def parse_args():
    parser = argparse.ArgumentParser(description="5-fold MLP search on both Markov datasets.")
    parser.add_argument("--workers", type=int, default=0, help="Fit processes. 0 keeps one core free.")
    parser.add_argument("--dry-run", action="store_true", help="Write the manifests and print the job count.")
    parser.add_argument("--smoke", action="store_true", help="Fit one fold of the full dataset, then exit.")
    parser.add_argument("--rebuild", action="store_true", help="Rescan the episode files into the manifests.")
    parser.add_argument("--skip-train", action="store_true", help="Do not fit the winning networks at the end.")
    return parser.parse_args()


def source_path(index):
    return os.path.join(SOURCE_DIR, f"episode_{index:03d}.npz")


def manifest_path(name):
    return os.path.join(DATASET_DIR, name, "dataset_meta.npz")


def out_dir(name):
    return os.path.join(TASK_DIR, f"markov_mlp_cv_{name}")


def make_configs():
    """Original six networks, plus one-factor moves of lr, L2, and batch size."""
    configs = []
    for config in CONFIGS:
        item = {
            "name": config["name"],
            "hidden": tuple(config["hidden"]),
            "activation": config["activation"],
            "lr": float(config["lr"]),
            "alpha": float(config["alpha"]),
            "batch_size": 256,
            "factor": "architecture",
        }
        configs.append(item)
    center = {
        "hidden": (128, 128),
        "activation": "tanh",
        "lr": 1e-3,
        "alpha": 1e-4,
        "batch_size": 256,
    }
    variants = (
        ("tanh-128x128-lr1e-4", {"lr": 1e-4}, "learning_rate"),
        ("tanh-128x128-lr1e-2", {"lr": 1e-2}, "learning_rate"),
        ("tanh-128x128-a1e-5", {"alpha": 1e-5}, "alpha"),
        ("tanh-128x128-a1e-3", {"alpha": 1e-3}, "alpha"),
        ("tanh-128x128-b64", {"batch_size": 64}, "batch_size"),
        ("tanh-128x128-b1024", {"batch_size": 1024}, "batch_size"),
    )
    for name, updates, factor in variants:
        item = dict(center)
        item.update(updates)
        item["name"] = name
        item["factor"] = factor
        configs.append(item)
    names = [item["name"] for item in configs]
    if len(names) != len(set(names)):
        raise RuntimeError("Two configs share a name.")
    return configs


def build_manifests(force):
    """Point the two datasets at the saved episodes. The trajectories stay where they are."""
    all_path = manifest_path("all")
    success_path = manifest_path("success")
    if not force and os.path.isfile(all_path) and os.path.isfile(success_path):
        with np.load(all_path) as meta:
            n_all = int(len(meta["source_index"]))
        with np.load(success_path) as meta:
            n_success = int(len(meta["source_index"]))
        print(f"Manifests on disk: all {n_all}, success {n_success}.", flush=True)
        return
    missing = [index for index in range(N_SOURCE) if not os.path.isfile(source_path(index))]
    if missing:
        raise SystemExit(f"Missing {len(missing)} episodes, starting at {missing[0]}.")
    with np.load(os.path.join(SOURCE_DIR, "dataset_meta.npz")) as meta:
        limits = {
            "dt": np.float64(meta["dt"]),
            "wheel_limit": np.float64(meta["wheel_limit"]),
            "torque_limit": np.float64(meta["torque_limit"]),
            "z_max": np.float64(meta["z_max"]),
        }
    targets = []
    reached = []
    final_error = []
    for index in range(N_SOURCE):
        with np.load(source_path(index)) as episode:
            targets.append(np.asarray(episode["target"], dtype=np.float64))
            reached.append(bool(episode["reached"]))
            final_error.append(float(episode["final_error"]))
    targets = np.vstack(targets)
    reached = np.asarray(reached, dtype=bool)
    final_error = np.asarray(final_error, dtype=np.float64)
    source_index = np.arange(N_SOURCE, dtype=np.int64)
    _write_manifest(all_path, source_index, targets, reached, final_error, limits, "all")
    keep = np.flatnonzero(reached)
    _write_manifest(
        success_path, source_index[keep], targets[keep], reached[keep], final_error[keep], limits, "success",
    )
    print(
        f"Wrote manifests: all {N_SOURCE} episodes, success {int(reached.sum())}, "
        f"misses {int((~reached).sum())}.",
        flush=True,
    )


def _write_manifest(path, source_index, targets, reached, final_error, limits, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "name": np.asarray(name),
        "source_index": np.asarray(source_index, dtype=np.int64),
        "target": np.asarray(targets, dtype=np.float64),
        "reached": np.asarray(reached, dtype=bool),
        "final_error": np.asarray(final_error, dtype=np.float64),
        **limits,
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez(handle, **payload)
    os.replace(tmp, path)


def load_manifest(name):
    path = manifest_path(name)
    with np.load(path) as meta:
        return {key: np.array(meta[key]) for key in meta.files}


def load_episodes(name):
    meta = load_manifest(name)
    dt = float(meta["dt"])
    stride = int(round(SAMPLE_STRIDE_S / dt))
    if stride < 1 or abs(stride * dt - SAMPLE_STRIDE_S) > 1e-9:
        raise SystemExit(f"25 ms is not an integer number of steps at dt={dt}.")
    episodes = []
    for index, src in enumerate(np.asarray(meta["source_index"], dtype=int)):
        path = source_path(int(src))
        with np.load(path) as episode:
            obs = np.asarray(episode["observation"], dtype=np.float64)[::stride]
            wheels = np.asarray(episode["wheel_cmd"], dtype=np.float64)[::stride]
            torque = np.asarray(episode["torque_arm"], dtype=np.float64)[::stride]
        if obs.ndim != 2 or obs.shape[1] != 22 or wheels.shape[1] != 2 or torque.shape[1] != 7:
            raise SystemExit(f"Episode {int(src)} has an unexpected array shape.")
        if not (len(obs) == len(wheels) == len(torque)):
            raise SystemExit(f"Episode {int(src)} logs do not share a length.")
        labels = np.hstack((wheels, torque))
        if not np.isfinite(obs).all() or not np.isfinite(labels).all():
            raise SystemExit(f"Episode {int(src)} has non-finite values.")
        episodes.append({
            "X": obs,
            "Y": labels,
            "index": index,
            "source_index": int(src),
        })
    n_samples = sum(len(item["X"]) for item in episodes)
    print(
        f"{name}: {len(episodes)} episodes, {n_samples} samples at a {SAMPLE_STRIDE_S * 1000:.0f} ms stride.",
        flush=True,
    )
    return episodes


def episode_folds(n):
    groups = np.arange(n)
    folds = []
    for train_pool, test_eps in GroupKFold(n_splits=N_FOLDS).split(groups, groups=groups):
        train_pool = np.asarray(train_pool, dtype=int)
        test_eps = np.asarray(test_eps, dtype=int)
        if len(test_eps) + len(train_pool) != n or np.intersect1d(train_pool, test_eps).size:
            raise RuntimeError("Episode fold is not a partition.")
        folds.append((train_pool, test_eps))
    if len(folds) != N_FOLDS:
        raise RuntimeError(f"Expected {N_FOLDS} folds, got {len(folds)}.")
    return folds


def result_key(record):
    return (int(record["fold"]), record["name"], record["size_label"])


def load_done(path):
    done = {}
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            done[result_key(record)] = record
    return done


def append_result(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _init(episodes):
    global _EPISODES
    _EPISODES = episodes


def _run(job):
    try:
        return _fit_job(job)
    except Exception as exc:
        config = job["config"]
        raise RuntimeError(
            f"fold {job['fold']} {config['name']} size {job['size_label']} failed"
        ) from exc


def _fit_job(job):
    episodes = _EPISODES
    if episodes is None:
        raise RuntimeError("Worker was started without the episode table.")
    train_pool = np.asarray(job["train_pool"], dtype=int)
    test_eps = np.asarray(job["test_eps"], dtype=int)
    size_label = job["size_label"]
    n_episodes = len(train_pool) if size_label == "all" else int(size_label)
    if n_episodes > len(train_pool) or n_episodes < 1:
        raise RuntimeError(f"Cannot draw {n_episodes} episodes from a pool of {len(train_pool)}.")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(job["fold"]), int(job["size_seed"])]))
    train_eps = rng.permutation(train_pool)[:n_episodes]
    if np.intersect1d(train_eps, test_eps).size:
        raise RuntimeError("Training episodes overlap the held-out fold.")
    x_test, y_test = stack(episodes, test_eps)
    x_train, y_train = stack(episodes, train_eps)
    x_train, y_train = shuffle_rows(x_train, y_train, rng)
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    config = job["config"]
    model = MLPRegressor(
        hidden_layer_sizes=tuple(config["hidden"]),
        activation=config["activation"],
        alpha=float(config["alpha"]),
        learning_rate_init=float(config["lr"]),
        batch_size=int(config["batch_size"]),
        max_iter=MAX_ITER,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=N_ITER_NO_CHANGE,
        shuffle=True,
        random_state=SEED,
        solver="adam",
    )
    started = time.perf_counter()
    model.fit(x_scaler.transform(x_train), y_scaler.transform(y_train))
    pred = y_scaler.inverse_transform(model.predict(x_scaler.transform(x_test)))
    std_mse, wheel_rmse, torque_rmse = score_fold(y_test, pred, y_scaler)
    return {
        "fold": int(job["fold"]),
        "name": config["name"],
        "factor": config["factor"],
        "hidden": [int(value) for value in config["hidden"]],
        "activation": config["activation"],
        "lr": float(config["lr"]),
        "alpha": float(config["alpha"]),
        "batch_size": int(config["batch_size"]),
        "size_label": size_label,
        "n_episodes": int(n_episodes),
        "n_train_samples": int(len(x_train)),
        "n_test_samples": int(len(x_test)),
        "n_test_episodes": int(len(test_eps)),
        "std_mse": float(std_mse),
        "wheel_rmse": float(wheel_rmse),
        "torque_rmse": float(torque_rmse),
        "n_iter": int(model.n_iter_),
        "seconds": float(time.perf_counter() - started),
    }


def iter_jobs(folds, configs):
    for fold, (train_pool, test_eps) in enumerate(folds):
        for size_label in SIZE_LEVELS:
            n_episodes = len(train_pool) if size_label == "all" else int(size_label)
            if n_episodes > len(train_pool):
                raise RuntimeError(
                    f"Fold {fold} has {len(train_pool)} training episodes, which is below {size_label}."
                )
            for config in configs:
                yield {
                    "fold": fold,
                    "train_pool": train_pool.tolist(),
                    "test_eps": test_eps.tolist(),
                    "size_label": size_label,
                    "size_seed": SIZE_SEED[size_label],
                    "config": config,
                }


def summarise(records):
    grouped = {}
    for record in records:
        grouped.setdefault((record["name"], record["size_label"]), []).append(record)
    rows = []
    for (name, size_label), items in grouped.items():
        if len(items) != N_FOLDS:
            continue
        std_mse = np.array([item["std_mse"] for item in items], dtype=float)
        std_rmse = np.sqrt(std_mse)
        wheel = np.array([item["wheel_rmse"] for item in items], dtype=float)
        torque = np.array([item["torque_rmse"] for item in items], dtype=float)
        n_episodes = np.array([item["n_episodes"] for item in items], dtype=float)
        first = items[0]
        rows.append({
            "name": name,
            "factor": first["factor"],
            "hidden": first["hidden"],
            "activation": first["activation"],
            "lr": first["lr"],
            "alpha": first["alpha"],
            "batch_size": first["batch_size"],
            "size_label": size_label,
            "n_episodes": float(n_episodes.mean()),
            "n_episodes_min": int(n_episodes.min()),
            "n_episodes_max": int(n_episodes.max()),
            "std_mse_mean": float(std_mse.mean()),
            "std_mse_std": float(std_mse.std(ddof=1)),
            "std_rmse_mean": float(std_rmse.mean()),
            "std_rmse_std": float(std_rmse.std(ddof=1)),
            "wheel_rmse_mean": float(wheel.mean()),
            "wheel_rmse_std": float(wheel.std(ddof=1)),
            "torque_rmse_mean": float(torque.mean()),
            "torque_rmse_std": float(torque.std(ddof=1)),
            "n_iter_mean": float(np.mean([item["n_iter"] for item in items])),
        })
    rows.sort(key=lambda row: (row["std_mse_mean"], row["std_mse_std"], row["name"], row["size_label"]))
    return rows


def plot_scores(rows, configs, path):
    names = [config["name"] for config in configs]
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    x = np.arange(len(names))
    width = 0.24
    for offset, size_label in enumerate(SIZE_LEVELS):
        means, errors = [], []
        for name in names:
            match = [row for row in rows if row["name"] == name and row["size_label"] == size_label]
            if not match:
                means.append(np.nan)
                errors.append(0.0)
                continue
            means.append(match[0]["std_rmse_mean"])
            errors.append(match[0]["std_rmse_std"])
        ax.bar(
            x + (offset - 1) * width,
            means,
            width=width,
            yerr=errors,
            capsize=2,
            label=f"{size_label} episodes",
        )
    ax.set_xticks(x, names, rotation=28, ha="right")
    ax.set_ylabel("Standardised RMSE")
    ax.set_title("5-fold episode holdout")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def print_table(rows, limit):
    print(
        f"\n{'rank':>4}  {'std MSE':>10}  {'spread':>8}  {'wheel':>8}  {'torque':>8}  "
        f"{'size':>5}  {'n':>6}  model",
        flush=True,
    )
    shown = rows if limit is None else rows[:limit]
    for rank, row in enumerate(shown, start=1):
        print(
            f"{rank:4d}  {row['std_mse_mean']:10.4f}  {row['std_mse_std']:8.4f}  "
            f"{row['wheel_rmse_mean']:8.4f}  {row['torque_rmse_mean']:8.4f}  "
            f"{row['size_label']:>5}  {row['n_episodes']:6.1f}  {row['name']}",
            flush=True,
        )


def run_search(name, episodes, configs, workers, smoke):
    folds = episode_folds(len(episodes))
    pool_sizes = sorted(len(train_pool) for train_pool, _test in folds)
    print(
        f"{name}: test folds {[len(test) for _train, test in folds]}, "
        f"training pools {pool_sizes[0]}..{pool_sizes[-1]}.",
        flush=True,
    )
    jobs = list(iter_jobs(folds, configs))
    directory = out_dir(name)
    os.makedirs(directory, exist_ok=True)
    jsonl_path = os.path.join(directory, "fold_results.jsonl")
    done = load_done(jsonl_path)
    pending = [job for job in jobs if result_key({
        "fold": job["fold"], "name": job["config"]["name"], "size_label": job["size_label"],
    }) not in done]
    print(f"{name}: {len(jobs)} fits, {len(done)} already saved, {len(pending)} to run.", flush=True)
    if smoke:
        pending = pending[:1]
        if not pending:
            print(f"{name}: smoke fit is already saved.", flush=True)
            return None
        started = time.perf_counter()
        _init(episodes)
        record = _run(pending[0])
        append_result(jsonl_path, record)
        print(
            f"smoke {record['name']} size {record['size_label']}  std MSE {record['std_mse']:.4f}  "
            f"iters {record['n_iter']}  {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        return None

    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(episodes,)) as pool:
            futures = [pool.submit(_run, job) for job in pending]
            finished = 0
            for future in as_completed(futures):
                record = future.result()
                append_result(jsonl_path, record)
                done[result_key(record)] = record
                finished += 1
                print(
                    f"[{finished}/{len(pending)}] {name}  fold {record['fold']}  "
                    f"{record['name']:<22}  size {record['size_label']:<3}  "
                    f"std MSE {record['std_mse']:.4f}  wheel {record['wheel_rmse']:.4f}  "
                    f"torque {record['torque_rmse']:.4f}  iters {record['n_iter']:3d}  "
                    f"{record['seconds']:.1f}s",
                    flush=True,
                )
    records = list(load_done(jsonl_path).values())
    rows = summarise(records)
    expected = len(configs) * len(SIZE_LEVELS)
    if len(rows) != expected:
        raise SystemExit(f"{name}: expected {expected} completed settings, found {len(rows)}.")
    summary_path = os.path.join(directory, "summary.json")
    payload = {
        "dataset": name,
        "n_episodes": len(episodes),
        "protocol": (
            "GroupKFold on episodes, 5 folds. Training sizes 25, 55, and every episode "
            "in the training fold. Training episodes are shuffled after the split and "
            "samples are shuffled before the fit. Stride 25 ms. Score is mean "
            "standardised MSE across folds. Learning rate, L2, and batch size are varied "
            "one at a time around tanh-128x128; the original six networks stay in the table."
        ),
        "sample_stride_s": SAMPLE_STRIDE_S,
        "seed": SEED,
        "ranking": rows,
        "best": rows[0],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    plot_path = os.path.join(directory, "cv_standardized_rmse.png")
    plot_scores(rows, configs, plot_path)
    print_table(rows, limit=12)
    best = rows[0]
    print(
        f"\n{name} best: {best['name']} size {best['size_label']} "
        f"(standardised MSE {best['std_mse_mean']:.4f} ± {best['std_mse_std']:.4f}).",
        flush=True,
    )
    print(f"Saved {summary_path}", flush=True)
    return best


def train_winner(name, episodes, best):
    import joblib

    meta = load_manifest(name)
    directory = out_dir(name)
    path = os.path.join(directory, "winner.joblib")
    if os.path.isfile(path):
        previous = joblib.load(path)
        if previous.get("name") == best["name"] and previous.get("size_label") == best["size_label"]:
            print(f"{name}: winner already saved at {path}", flush=True)
            return path
    n_draw = len(episodes) if best["size_label"] == "all" else int(best["size_label"])
    rng = np.random.default_rng(SEED)
    chosen = rng.permutation(len(episodes))[:n_draw]
    features, labels = stack(episodes, chosen)
    features, labels = shuffle_rows(features, labels, rng)
    x_scaler = StandardScaler().fit(features)
    y_scaler = StandardScaler().fit(labels)
    model = MLPRegressor(
        hidden_layer_sizes=tuple(best["hidden"]),
        activation=best["activation"],
        alpha=float(best["alpha"]),
        learning_rate_init=float(best["lr"]),
        batch_size=int(best["batch_size"]),
        max_iter=MAX_ITER,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=N_ITER_NO_CHANGE,
        shuffle=True,
        random_state=SEED,
        solver="adam",
    )
    print(
        f"Training {name} winner {best['name']} on {len(chosen)} episodes, {len(features)} samples.",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(x_scaler.transform(features), y_scaler.transform(labels))
    source = np.array([episodes[int(index)]["source_index"] for index in chosen], dtype=int)
    joblib.dump({
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "name": best["name"],
        "dataset": name,
        "factor": best["factor"],
        "hidden": list(best["hidden"]),
        "activation": best["activation"],
        "lr": float(best["lr"]),
        "alpha": float(best["alpha"]),
        "batch_size": int(best["batch_size"]),
        "size_label": best["size_label"],
        "n_episodes": int(len(chosen)),
        "source_episodes": np.sort(source),
        "wheel_limit": float(meta["wheel_limit"]),
        "torque_limit": float(meta["torque_limit"]),
        "sample_stride_s": SAMPLE_STRIDE_S,
        "n_iter": int(model.n_iter_),
        "cv_std_mse_mean": float(best["std_mse_mean"]),
    }, path)
    print(
        f"Saved {path} after {model.n_iter_} iterations ({time.perf_counter() - started:.1f}s).",
        flush=True,
    )
    return path


def main():
    args = parse_args()
    configs = make_configs()
    build_manifests(args.rebuild)
    if args.dry_run:
        from eval_mlp_lattice import lattice_targets

        for name in ("all", "success"):
            n = len(load_manifest(name)["source_index"])
            folds = episode_folds(n)
            pools = [len(train_pool) for train_pool, _test in folds]
            print(
                f"{name}: {n} episodes, test {[len(test) for _train, test in folds]}, "
                f"training pool {min(pools)}..{max(pools)}.",
                flush=True,
            )
        targets = lattice_targets()
        print(
            f"{len(configs)} networks x {len(SIZE_LEVELS)} sizes x {N_FOLDS} folds "
            f"= {len(configs) * len(SIZE_LEVELS) * N_FOLDS} fits per dataset.",
            flush=True,
        )
        print(
            f"Lattice {len(targets)} targets, centre {targets.mean(axis=0)}, "
            f"span x {targets[:, 0].min():.1f}..{targets[:, 0].max():.1f} "
            f"y {targets[:, 1].min():.1f}..{targets[:, 1].max():.1f} "
            f"z {targets[:, 2].min():.1f}..{targets[:, 2].max():.1f}.",
            flush=True,
        )
        return
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    names = ("all",) if args.smoke else ("all", "success")
    for name in names:
        episodes = load_episodes(name)
        best = run_search(name, episodes, configs, workers, smoke=args.smoke)
        if args.smoke or args.skip_train or best is None:
            continue
        train_winner(name, episodes, best)


if __name__ == "__main__":
    main()
