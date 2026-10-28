"""Train the behaviour-cloning policy and the phase diagnostic.

Trajectories are split with GroupShuffleSplit: 80 episodes train, 20 validate.
Samples are shuffled inside the training set only. One XGBoost regressor with
``multi_strategy="multi_output_tree"`` predicts the two wheel speeds and seven
arm torques, with early stopping on the held-out trajectories.

Phase and ``q_des_arm`` are logged for this diagnostic and are not features.
A classifier on the same features predicts the logged phase. Accuracy near the
majority-class baseline means the features are missing information.

From the RoboEnv checkout, after ``collect_bc_data.py``::

    pixi run python ../final/robotnik_arm_task/train_bc.py
"""
import os

os.environ.setdefault("ROBOENV_HEADLESS", "1")

import argparse
import json

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier, XGBRegressor

# Imports the task module, which selects the matplotlib backend before pyplot loads.
import robotnik_arm_task as task
from bc_features import (
    DATA_DIR,
    FEATURE_NAMES,
    MODEL_DIR,
    N_TARGETS,
    N_TRAIN_TRAJECTORIES,
    N_VAL_TRAJECTORIES,
    OUTPUT_NAMES,
    PHASE_LABELS,
    PHASE_ORDER,
    SEED,
    action_labels,
    episode_feature_matrix,
    episode_path,
    meta_path,
)
import matplotlib.pyplot as plt

# 1 ms samples repeat almost the same state, so a leaf must cover more than a
# single step. Depth 8 can still represent the align / drive / orient / reach split.
N_ESTIMATORS = 600
MAX_DEPTH = 8
LEARNING_RATE = 0.1
MIN_CHILD_WEIGHT = 50
EARLY_STOPPING_ROUNDS = 40
CLASSIFIER_ESTIMATORS = 200
CLASSIFIER_DEPTH = 6
CLASSIFIER_EARLY_STOPPING = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Train the behaviour-cloning regressor.")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--model", default=MODEL_DIR)
    return parser.parse_args()


def load_episodes(data_dir):
    meta_file = meta_path(data_dir)
    if not os.path.isfile(meta_file):
        raise SystemExit(f"No dataset metadata at {meta_file}. Run collect_bc_data.py first.")
    meta = np.load(meta_file)
    n = len(meta["targets"])
    if n != N_TARGETS:
        raise SystemExit(f"Expected {N_TARGETS} targets in {meta_file}, found {n}.")
    dt = float(meta["dt"])
    missing = [i for i in range(n) if not os.path.isfile(episode_path(data_dir, i))]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} of {n} episodes (first index {missing[0]}). "
            "Finish collection before training."
        )

    episodes = []
    for index in range(n):
        with np.load(episode_path(data_dir, index), allow_pickle=True) as episode:
            time = np.asarray(episode["time"], dtype=np.float64)
            if len(time) > 1 and np.max(np.abs(np.diff(time) - dt)) > 1e-9:
                raise SystemExit(f"Episode {index} time step does not match dataset dt {dt}")
            phase = np.asarray(episode["phase"]).astype("U16")
            unknown = set(phase.tolist()) - set(PHASE_ORDER)
            if unknown:
                raise SystemExit(f"Episode {index} has unexpected phases {sorted(unknown)}")
            features = episode_feature_matrix(
                time,
                episode["base_pos"],
                episode["base_bearing"],
                episode["q_arm"],
                episode["qd_arm"],
                episode["target"],
                episode["q_init_arm"],
                episode["start_base_pos"],
                dt,
            ).astype(np.float32)
            labels = action_labels(episode["wheel_cmds"], episode["torque_arm"]).astype(np.float32)
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise SystemExit(f"Episode {index} has non-finite features or labels")
        episodes.append({"X": features, "Y": labels, "phase": phase, "index": index})
        print(f"  episode {index:03d}  {len(time):6d} samples", flush=True)
    return episodes, dt


def stack(episodes, indices):
    chosen = [episodes[int(i)] for i in indices]
    return (
        np.vstack([item["X"] for item in chosen]),
        np.vstack([item["Y"] for item in chosen]),
        np.concatenate([item["phase"] for item in chosen]),
    )


def rmse_table(y_true, y_pred, phase):
    """Rows are phases plus an overall row. Columns follow OUTPUT_NAMES."""
    rows = []
    counts = []
    for name in PHASE_ORDER:
        mask = phase == name
        counts.append(int(mask.sum()))
        if mask.any():
            rows.append(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2, axis=0)))
        else:
            rows.append(np.full(y_true.shape[1], np.nan))
    counts.append(int(len(y_true)))
    rows.append(np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0)))
    return np.vstack(rows), np.asarray(counts, dtype=int)


def print_rmse(table, counts):
    phases = PHASE_ORDER + ["all"]
    print(
        "One multi-output tree fits all 9 labels in raw units, so Newton-metres and "
        "rad/s share one squared loss. Read the per-output RMSE, not the training score."
    )
    sections = (("Wheels (rad/s)", slice(0, 2)), ("Arm torques (N*m)", slice(2, 9)))
    for title, cols in sections:
        print(f"\n{title}")
        header = f"{'phase':<16} {'n':>8}" + "".join(f"{name:>14}" for name in OUTPUT_NAMES[cols])
        print(header)
        for i, phase in enumerate(phases):
            label = PHASE_LABELS.get(phase, phase)
            values = "".join(f"{table[i, j]:14.4f}" for j in range(cols.start, cols.stop))
            print(f"{label:<16} {counts[i]:8d}{values}")


def plot_rmse(table, counts, path):
    phases = [PHASE_LABELS.get(name, name) for name in PHASE_ORDER] + ["all"]
    ylabels = [f"{name}\nn={counts[i]}" for i, name in enumerate(phases)]
    sections = (
        (OUTPUT_NAMES[:2], table[:, :2], "Wheels (rad/s)"),
        (OUTPUT_NAMES[2:], table[:, 2:], "Arm torques (N·m)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.15, 2.6]})
    for ax, (names, values, title) in zip(axes, sections):
        image = ax.imshow(values, cmap="viridis", aspect="auto", vmin=0.0)
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
        ax.set_yticks(range(len(ylabels)), ylabels)
        ax.set_title(title)
        finite = values[np.isfinite(values)]
        cutoff = float(np.median(finite)) if len(finite) else 0.0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                text = "—" if not np.isfinite(values[i, j]) else f"{values[i, j]:.3f}"
                color = "white" if np.isfinite(values[i, j]) and values[i, j] >= cutoff else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("RMSE")
    fig.suptitle(
        "Held-out trajectories: RMSE by phase\n"
        "done (reach) does not observe q_des, so it is expected to be the hardest",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_confusion(cm, accuracy, balanced, majority, path):
    totals = cm.sum(axis=1, keepdims=True)
    recall = cm / np.maximum(totals, 1)
    labels = [PHASE_LABELS[name] for name in PHASE_ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(recall, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("Logged phase")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if recall[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{recall[i, j]:.0%}", ha="center", va="center", color=color, fontsize=9)
    ax.set_title(
        f"Phase from the same features\n"
        f"accuracy {accuracy:.1%}   balanced {balanced:.1%}   majority baseline {majority:.1%}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def phase_ids(phase):
    mapping = {name: i for i, name in enumerate(PHASE_ORDER)}
    return np.array([mapping[value] for value in phase], dtype=np.int32)


def fit_regressor(x_train, y_train, x_val, y_val):
    model = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_child_weight=MIN_CHILD_WEIGHT,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        device="cpu",
        multi_strategy="multi_output_tree",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=25)
    if model.best_iteration is None:
        raise RuntimeError("Early stopping did not record a best iteration")
    return model


def fit_phase_classifier(x_train, phase_train, x_val, phase_val):
    model = XGBClassifier(
        n_estimators=CLASSIFIER_ESTIMATORS,
        max_depth=CLASSIFIER_DEPTH,
        learning_rate=LEARNING_RATE,
        min_child_weight=MIN_CHILD_WEIGHT,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cpu",
        early_stopping_rounds=CLASSIFIER_EARLY_STOPPING,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(
        x_train, phase_ids(phase_train),
        eval_set=[(x_val, phase_ids(phase_val))],
        verbose=25,
    )
    return model


def best_score_value(model):
    score = model.best_score
    if isinstance(score, dict):
        score = next(iter(score.values()))
    return float(score)


def main():
    args = parse_args()
    print(f"Loading {N_TARGETS} episodes from {args.data}", flush=True)
    episodes, dt = load_episodes(args.data)
    n_samples = sum(len(item["X"]) for item in episodes)
    print(f"{n_samples} samples, dt={dt}", flush=True)

    groups = np.arange(len(episodes))
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=N_TRAIN_TRAJECTORIES,
        test_size=N_VAL_TRAJECTORIES,
        random_state=SEED,
    )
    train_eps, val_eps = next(splitter.split(np.zeros((len(episodes), 1)), groups=groups))
    train_eps = np.sort(train_eps)
    val_eps = np.sort(val_eps)
    if len(train_eps) != N_TRAIN_TRAJECTORIES or len(val_eps) != N_VAL_TRAJECTORIES:
        raise RuntimeError(f"Split sizes were {len(train_eps)} / {len(val_eps)}, expected 80 / 20")
    if np.intersect1d(train_eps, val_eps).size:
        raise RuntimeError("A trajectory landed in both sides of the split")
    print(
        f"GroupShuffleSplit trajectories: train {train_eps.tolist()}\n"
        f"  validation {val_eps.tolist()}",
        flush=True,
    )

    x_train, y_train, phase_train = stack(episodes, train_eps)
    x_val, y_val, phase_val = stack(episodes, val_eps)
    del episodes
    order = np.random.default_rng(SEED).permutation(len(x_train))
    x_train = x_train[order]
    y_train = y_train[order]
    phase_train = phase_train[order]
    print(
        f"Shuffled {len(x_train)} training samples. "
        f"Left {len(x_val)} validation samples in time order.",
        flush=True,
    )

    print("Fitting the 9-output regressor", flush=True)
    model = fit_regressor(x_train, y_train, x_val, y_val)
    best_iteration = int(model.best_iteration)
    os.makedirs(args.model, exist_ok=True)
    policy_path = os.path.join(args.model, "policy.json")
    model.save_model(policy_path)
    meta = {
        "best_iteration": best_iteration,
        "best_score": best_score_value(model),
        "iteration_range": [0, best_iteration + 1],
        "feature_names": FEATURE_NAMES,
        "output_names": OUTPUT_NAMES,
        "train_episodes": train_eps.tolist(),
        "val_episodes": val_eps.tolist(),
        "seed": SEED,
        "dt": dt,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "multi_strategy": "multi_output_tree",
    }
    meta_json = os.path.join(args.model, "meta.json")
    with open(meta_json, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(
        f"Best iteration {best_iteration} / {N_ESTIMATORS - 1}, "
        f"validation score {meta['best_score']:.6f}. Saved {policy_path}",
        flush=True,
    )

    prediction = model.predict(x_val)
    table, counts = rmse_table(y_val, prediction, phase_val)
    print_rmse(table, counts)
    rmse_path = os.path.join(task.plots_dir("bc"), "bc_val_rmse_by_phase.png")
    plot_rmse(table, counts, rmse_path)

    print("\nFitting the phase classifier", flush=True)
    classifier = fit_phase_classifier(x_train, phase_train, x_val, phase_val)
    y_val_id = phase_ids(phase_val)
    y_hat = classifier.predict(x_val)
    accuracy = float(accuracy_score(y_val_id, y_hat))
    balanced = float(balanced_accuracy_score(y_val_id, y_hat))
    class_counts = np.bincount(y_val_id, minlength=len(PHASE_ORDER))
    majority = float(class_counts.max() / class_counts.sum())
    cm = confusion_matrix(y_val_id, y_hat, labels=list(range(len(PHASE_ORDER))))
    print(
        f"Phase classifier accuracy {accuracy:.3f} on held-out trajectories "
        f"(balanced {balanced:.3f}, majority-class baseline {majority:.3f})."
    )
    print(
        "Accuracy near the majority baseline would mean these features are missing "
        "the phase, rather than the regressor being too weak."
    )
    recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    for name, recall, count in zip(PHASE_ORDER, recalls, cm.sum(axis=1)):
        print(f"  {PHASE_LABELS[name]:<16} recall {recall:.3f}  n={int(count)}")
    confusion_path = os.path.join(task.plots_dir("bc"), "bc_phase_classifier.png")
    plot_confusion(cm, accuracy, balanced, majority, confusion_path)

    with open(os.path.join(args.model, "diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "rmse_rows": PHASE_ORDER + ["all"],
                "rmse_columns": OUTPUT_NAMES,
                "rmse": table.tolist(),
                "counts": counts.tolist(),
                "phase_accuracy": accuracy,
                "phase_balanced_accuracy": balanced,
                "phase_majority_baseline": majority,
                "confusion_matrix": cm.tolist(),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
