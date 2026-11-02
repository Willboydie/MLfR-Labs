"""5-fold cross-validation of MLP behaviour-cloning models on the Markov dataset.

Episodes are the split unit: a trajectory is entirely in training or entirely
in the held-out fold. GroupKFold makes five such folds of the 100 episodes
(20 scored, 80 available to train). The training-size settings are 25, 55,
and 80. Eighty is the whole training fold. Training on 85 would take episodes
out of the fold being scored.

After that split, the training episodes are shuffled and the requested count
is kept. Their samples are shuffled again before the fit, so the network does
not see a trajectory in time order.

Samples are taken every 25 ms. The log is 1 ms, and once rows are shuffled
those neighbours are repeated copies of the same command.

The selection score is the mean, over the five folds, of the mean squared
error on labels that were standardised with the training fold only. Wheels
and torques then count equally. Raw RMSE in rad/s and N*m is reported beside it.

From the RoboEnv checkout::

    pixi run python -u ../final/robotnik_arm_task/cv_markov_mlp.py
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "markov_data")
OUT_DIR = os.path.join(TASK_DIR, "markov_mlp_cv")

N_EPISODES = 100
N_FOLDS = 5
# Training episodes drawn from the 80 available in a fold.
TRAIN_SIZES = (25, 55, 80)
SAMPLE_STRIDE_S = 0.025
SEED = 0

# Width, depth, activation, step size, and L2. Batch size and the Adam
# settings stay fixed so this grid is the thing being compared.
CONFIGS = (
    {"name": "relu-64x64", "hidden": (64, 64), "activation": "relu", "lr": 1e-3, "alpha": 1e-4},
    {"name": "relu-128x128", "hidden": (128, 128), "activation": "relu", "lr": 1e-3, "alpha": 1e-4},
    {"name": "relu-256x128", "hidden": (256, 128), "activation": "relu", "lr": 1e-3, "alpha": 1e-4},
    {"name": "relu-128x128x64", "hidden": (128, 128, 64), "activation": "relu", "lr": 1e-3, "alpha": 1e-4},
    {"name": "relu-128x128-a1e-2", "hidden": (128, 128), "activation": "relu", "lr": 1e-3, "alpha": 1e-2},
    {"name": "tanh-128x128", "hidden": (128, 128), "activation": "tanh", "lr": 1e-3, "alpha": 1e-4},
)

MAX_ITER = 200
BATCH_SIZE = 256
N_ITER_NO_CHANGE = 6


def parse_args():
    parser = argparse.ArgumentParser(description="5-fold MLP cross-validation on the Markov episodes.")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument(
        "--train-winner",
        action="store_true",
        help="Fit tanh-128x128 on 80 training episodes and save it for the GUI.",
    )
    return parser.parse_args()


def load_episodes(data_dir):
    meta_path = os.path.join(data_dir, "dataset_meta.npz")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"No dataset metadata at {meta_path}.")
    with np.load(meta_path) as meta:
        dt = float(meta["dt"])
        n = len(meta["targets"])
    if n != N_EPISODES:
        raise SystemExit(f"Expected {N_EPISODES} episodes in {meta_path}, found {n}.")
    stride = int(round(SAMPLE_STRIDE_S / dt))
    if stride < 1 or abs(stride * dt - SAMPLE_STRIDE_S) > 1e-9:
        raise SystemExit(f"25 ms is not an integer number of steps at dt={dt}.")

    episodes = []
    for index in range(n):
        path = os.path.join(data_dir, f"episode_{index:03d}.npz")
        if not os.path.isfile(path):
            raise SystemExit(f"Missing {path}.")
        with np.load(path) as episode:
            obs = np.asarray(episode["observation"], dtype=np.float64)
            wheels = np.asarray(episode["wheel_cmd"], dtype=np.float64)
            torque = np.asarray(episode["torque_arm"], dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != 22 or wheels.shape[1] != 2 or torque.shape[1] != 7:
            raise SystemExit(f"Episode {index} has an unexpected array shape.")
        if not (len(obs) == len(wheels) == len(torque)):
            raise SystemExit(f"Episode {index} logs do not share a length.")
        features = obs[::stride]
        labels = np.hstack((wheels, torque))[::stride]
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise SystemExit(f"Episode {index} has non-finite values.")
        episodes.append({"X": features, "Y": labels, "index": index})
    n_samples = sum(len(item["X"]) for item in episodes)
    print(
        f"Loaded {n} episodes, {n_samples} samples at a {SAMPLE_STRIDE_S * 1000:.0f} ms stride.",
        flush=True,
    )
    return episodes


def episode_folds():
    """Five folds over episodes. Each episode is scored in exactly one fold."""
    groups = np.arange(N_EPISODES)
    splitter = GroupKFold(n_splits=N_FOLDS)
    folds = []
    for train_pool, test_eps in splitter.split(groups, groups=groups):
        train_pool = np.asarray(train_pool, dtype=int)
        test_eps = np.asarray(test_eps, dtype=int)
        if len(test_eps) + len(train_pool) != N_EPISODES or np.intersect1d(train_pool, test_eps).size:
            raise RuntimeError("Episode fold is not a partition.")
        folds.append((train_pool, test_eps))
    if len(folds) != N_FOLDS:
        raise RuntimeError(f"Expected {N_FOLDS} folds, got {len(folds)}.")
    return folds


def stack(episodes, indices):
    chosen = [episodes[int(i)] for i in indices]
    return (
        np.vstack([item["X"] for item in chosen]),
        np.vstack([item["Y"] for item in chosen]),
    )


def shuffle_rows(features, labels, rng):
    order = rng.permutation(len(features))
    return features[order], labels[order]


def score_fold(y_true, y_pred, y_scaler):
    """Standardised MSE, plus RMSE in the physical units of each command group."""
    true_s = y_scaler.transform(y_true)
    pred_s = y_scaler.transform(y_pred)
    std_mse = float(np.mean((pred_s - true_s) ** 2))
    wheel_rmse = float(np.sqrt(np.mean((y_pred[:, :2] - y_true[:, :2]) ** 2)))
    torque_rmse = float(np.sqrt(np.mean((y_pred[:, 2:] - y_true[:, 2:]) ** 2)))
    return std_mse, wheel_rmse, torque_rmse


def fit_config(config, x_train, y_train, x_test):
    model = MLPRegressor(
        hidden_layer_sizes=config["hidden"],
        activation=config["activation"],
        alpha=config["alpha"],
        learning_rate_init=config["lr"],
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=N_ITER_NO_CHANGE,
        shuffle=True,
        random_state=SEED,
        solver="adam",
    )
    model.fit(x_train, y_train)
    return model.predict(x_test), int(model.n_iter_)


def result_key(record):
    return (record["fold"], record["name"], int(record["n_episodes"]))


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


def summarise(records):
    """One row per (architecture, training size): mean and spread across folds."""
    grouped = {}
    for record in records:
        key = (record["name"], int(record["n_episodes"]))
        grouped.setdefault(key, []).append(record)
    rows = []
    for (name, n_episodes), items in grouped.items():
        if len(items) != N_FOLDS:
            continue
        std_mse = np.array([item["std_mse"] for item in items], dtype=float)
        std_rmse = np.sqrt(std_mse)
        wheel = np.array([item["wheel_rmse"] for item in items], dtype=float)
        torque = np.array([item["torque_rmse"] for item in items], dtype=float)
        config = next(item for item in CONFIGS if item["name"] == name)
        rows.append({
            "name": name,
            "hidden": list(config["hidden"]),
            "activation": config["activation"],
            "lr": config["lr"],
            "alpha": config["alpha"],
            "n_episodes": n_episodes,
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
    rows.sort(key=lambda row: (row["std_mse_mean"], row["std_mse_std"], row["name"]))
    return rows


def plot_scores(rows, path):
    names = [config["name"] for config in CONFIGS]
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    x = np.arange(len(names))
    width = 0.24
    for offset, n_episodes in enumerate(TRAIN_SIZES):
        means, errors = [], []
        for name in names:
            match = [row for row in rows if row["name"] == name and row["n_episodes"] == n_episodes]
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
            capsize=3,
            label=f"{n_episodes} episodes",
        )
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("Standardised RMSE")
    ax.set_title("5-fold episode holdout")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def print_table(rows):
    print(
        f"\n{'rank':>4}  {'std MSE':>10}  {'±':>8}  {'wheel RMSE':>11}  {'torque RMSE':>12}  "
        f"{'n':>3}  model",
        flush=True,
    )
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:4d}  {row['std_mse_mean']:10.4f}  {row['std_mse_std']:8.4f}  "
            f"{row['wheel_rmse_mean']:11.4f}  {row['torque_rmse_mean']:12.4f}  "
            f"{row['n_episodes']:3d}  {row['name']}",
            flush=True,
        )


def train_winner(data_dir, out_dir):
    """Fit the selected network on 80 episodes from the stored training split."""
    import joblib

    from markov_mlp import N_TRAIN_EPISODES, WINNER

    episodes = load_episodes(data_dir)
    meta_file = os.path.join(data_dir, "dataset_meta.npz")
    with np.load(meta_file) as meta:
        pool = np.asarray(meta["train_index"], dtype=int)
        wheel_limit = float(meta["wheel_limit"])
        torque_limit = float(meta["torque_limit"])
    if len(pool) < N_TRAIN_EPISODES:
        raise SystemExit(f"Training split has {len(pool)} episodes; need {N_TRAIN_EPISODES}.")
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.permutation(pool)[:N_TRAIN_EPISODES])
    features, labels = stack(episodes, chosen)
    features, labels = shuffle_rows(features, labels, rng)
    x_scaler = StandardScaler().fit(features)
    y_scaler = StandardScaler().fit(labels)
    model = MLPRegressor(
        hidden_layer_sizes=WINNER["hidden"],
        activation=WINNER["activation"],
        alpha=WINNER["alpha"],
        learning_rate_init=WINNER["lr"],
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=N_ITER_NO_CHANGE,
        shuffle=True,
        random_state=SEED,
        solver="adam",
    )
    print(
        f"Training {WINNER['name']} on {len(chosen)} episodes, {len(features)} samples.",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(x_scaler.transform(features), y_scaler.transform(labels))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{WINNER['name']}.joblib")
    joblib.dump({
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "name": WINNER["name"],
        "hidden": list(WINNER["hidden"]),
        "activation": WINNER["activation"],
        "lr": WINNER["lr"],
        "alpha": WINNER["alpha"],
        "n_episodes": int(len(chosen)),
        "train_episodes": chosen.astype(int),
        "wheel_limit": wheel_limit,
        "torque_limit": torque_limit,
        "sample_stride_s": SAMPLE_STRIDE_S,
        "n_iter": int(model.n_iter_),
    }, path)
    print(
        f"Saved {path} after {model.n_iter_} iterations "
        f"({time.perf_counter() - started:.1f}s).",
        flush=True,
    )


def main():
    args = parse_args()
    if args.train_winner:
        train_winner(args.data, args.out)
        return
    os.makedirs(args.out, exist_ok=True)
    episodes = load_episodes(args.data)
    folds = episode_folds()
    jsonl_path = os.path.join(args.out, "fold_results.jsonl")
    done = load_done(jsonl_path)
    if done:
        print(f"Resuming: {len(done)} fold results already on disk.", flush=True)

    for fold, (train_pool, test_eps) in enumerate(folds):
        x_test, y_test = stack(episodes, test_eps)
        for n_episodes in TRAIN_SIZES:
            rng = np.random.default_rng(np.random.SeedSequence([SEED, fold, n_episodes]))
            train_eps = rng.permutation(train_pool)[:n_episodes]
            if np.intersect1d(train_eps, test_eps).size:
                raise RuntimeError("Training episodes overlap the held-out fold.")
            x_train, y_train = stack(episodes, train_eps)
            x_train, y_train = shuffle_rows(x_train, y_train, rng)
            x_scaler = StandardScaler().fit(x_train)
            y_scaler = StandardScaler().fit(y_train)
            x_train_s = x_scaler.transform(x_train)
            y_train_s = y_scaler.transform(y_train)
            x_test_s = x_scaler.transform(x_test)
            for config in CONFIGS:
                key = (fold, config["name"], n_episodes)
                if key in done:
                    continue
                started = time.perf_counter()
                pred_s, n_iter = fit_config(config, x_train_s, y_train_s, x_test_s)
                pred = y_scaler.inverse_transform(pred_s)
                std_mse, wheel_rmse, torque_rmse = score_fold(y_test, pred, y_scaler)
                record = {
                    "fold": fold,
                    "name": config["name"],
                    "hidden": list(config["hidden"]),
                    "activation": config["activation"],
                    "lr": config["lr"],
                    "alpha": config["alpha"],
                    "n_episodes": n_episodes,
                    "n_train_samples": int(len(x_train)),
                    "n_test_samples": int(len(x_test)),
                    "n_test_episodes": int(len(test_eps)),
                    "std_mse": std_mse,
                    "wheel_rmse": wheel_rmse,
                    "torque_rmse": torque_rmse,
                    "n_iter": n_iter,
                    "seconds": float(time.perf_counter() - started),
                }
                append_result(jsonl_path, record)
                done[key] = record
                print(
                    f"fold {fold}  {config['name']:<20}  n={n_episodes:<2}  "
                    f"std MSE {std_mse:.4f}  wheel {wheel_rmse:.4f}  torque {torque_rmse:.4f}  "
                    f"iters {n_iter:2d}  {record['seconds']:.1f}s",
                    flush=True,
                )

    records = list(done.values())
    rows = summarise(records)
    if len(rows) != len(CONFIGS) * len(TRAIN_SIZES):
        raise SystemExit(
            f"Expected {len(CONFIGS) * len(TRAIN_SIZES)} completed settings, found {len(rows)}."
        )
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "protocol": (
                "GroupKFold on episodes, 5 folds, 20 episodes scored per fold. "
                "Training sizes 25, 55, and 80 are drawn from the other 80 after a shuffle. "
                "Training samples are shuffled after that. "
                "Score is mean standardised MSE across folds."
            ),
            "sample_stride_s": SAMPLE_STRIDE_S,
            "seed": SEED,
            "ranking": rows,
            "best": rows[0],
        }, handle, indent=2)
    plot_path = os.path.join(args.out, "cv_standardized_rmse.png")
    plot_scores(rows, plot_path)
    print_table(rows)
    best = rows[0]
    print(
        f"\nBest: {best['name']} with {best['n_episodes']} training episodes "
        f"(standardised MSE {best['std_mse_mean']:.4f} ± {best['std_mse_std']:.4f}).",
        flush=True,
    )
    print(f"Saved {summary_path}", flush=True)
    print(f"Saved {plot_path}", flush=True)


if __name__ == "__main__":
    main()
