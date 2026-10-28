"""Closed-loop evaluation of the behaviour-cloning policy on the assessment grid.

The model replaces the expert. Wheel outputs are copied back to
``[right, left, left, right]``. ``Booster.inplace_predict`` is used at every
step, limited to the early-stopped trees. Forward kinematics is only for the
end-effector error and the stop rule; it is not a model input.

From the RoboEnv checkout, after ``train_bc.py``::

    pixi run python ../final/robotnik_arm_task/eval_bc.py
"""
import os
import sys

if "--gui" not in sys.argv:
    os.environ.setdefault("ROBOENV_HEADLESS", "1")
else:
    os.environ["ROBOENV_HEADLESS"] = "0"

import argparse
import json
import time

import numpy as np
import pinocchio as pin
from simulation_and_control import MotorCommands, wrap_angle
from xgboost import XGBRegressor

from robotnik_arm_control import StopCriteria, check_stop, quaternion2bearing
import robotnik_arm_task as task
from bc_features import (
    FEATURE_NAMES,
    MODEL_DIR,
    OUTPUT_NAMES,
    TASK_DIR,
    FeatureBuffer,
    expand_wheel_commands,
)
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the behaviour clone on the sparse grid.")
    parser.add_argument("--model", default=MODEL_DIR)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help="Run the expert and the clone on one random assessment-grid target and plot commands.",
    )
    return parser.parse_args()


def load_policy(model_dir):
    meta_file = os.path.join(model_dir, "meta.json")
    policy_file = os.path.join(model_dir, "policy.json")
    if not os.path.isfile(meta_file) or not os.path.isfile(policy_file):
        raise SystemExit(f"Need {policy_file} and {meta_file}. Run train_bc.py first.")
    with open(meta_file, encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta["feature_names"] != FEATURE_NAMES:
        raise SystemExit("Saved feature names do not match the current feature code. Retrain.")
    if meta["output_names"] != OUTPUT_NAMES:
        raise SystemExit("Saved output names do not match the current labels. Retrain.")
    model = XGBRegressor()
    model.load_model(policy_file)
    iteration_range = tuple(int(v) for v in meta["iteration_range"])
    return model, model.get_booster(), iteration_range


def load_expert_grid():
    """Expert final errors, from a combined file or from ``expert_grid_part_*.npz``."""
    combined = os.path.join(TASK_DIR, "expert_grid.npz")
    if os.path.isfile(combined):
        data = np.load(combined, allow_pickle=True)
        return np.asarray(data["targets"], dtype=float), np.asarray(data["errors"], dtype=float)
    parts = sorted(
        os.path.join(TASK_DIR, name)
        for name in os.listdir(TASK_DIR)
        if name.startswith("expert_grid_part_") and name.endswith(".npz")
    )
    if not parts:
        return None
    chunks = [np.load(path, allow_pickle=True) for path in parts]
    targets = np.vstack([chunk["targets"] for chunk in chunks])
    errors = np.concatenate([np.asarray(chunk["errors"], dtype=float) for chunk in chunks])
    return targets, errors


def align_expert(model_targets, expert_targets, expert_errors):
    aligned = np.full(len(model_targets), np.nan)
    for i, target in enumerate(model_targets):
        distance = np.linalg.norm(expert_targets - target, axis=1)
        nearest = int(np.argmin(distance))
        if distance[nearest] < 1e-4:
            aligned[i] = expert_errors[nearest]
    return aligned


def reaching_window(base_pos, base_bearing, goal_xy, goal_bearing, current_time, params):
    """Whether reach and stall checks are allowed. Not a model input.

    Matches the expert's handover: the base is inside its position and bearing
    tolerances, or the navigation timeout has fired (the expert then forces the
    base phase to done).
    """
    distance = float(np.linalg.norm(np.asarray(base_pos)[:2] - goal_xy))
    bearing_error = abs(float(wrap_angle(goal_bearing - base_bearing)))
    arrived = distance < params.pos_tol and bearing_error < params.ang_tol
    return arrived or current_time > params.nav_timeout


def predict_command(booster, buf, iteration_range):
    pred = np.asarray(booster.inplace_predict(buf, iteration_range=iteration_range))
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if pred.shape != (1, len(OUTPUT_NAMES)) or not np.isfinite(pred).all():
        raise RuntimeError(f"Unexpected prediction {pred}")
    torque = np.asarray(pred[0, 2:], dtype=np.float64)
    return expand_wheel_commands(float(pred[0, 0]), float(pred[0, 1])), torque, pred


def run_model_episode(ctx, model, booster, iteration_range, target_ee_pos, record=False):
    """Closed loop from the default reset. The expert controllers are not called."""
    sim = ctx["sim"]
    client = ctx["client"]
    dyn_model = ctx["dyn_model"]
    target_ee_pos = np.asarray(target_ee_pos, dtype=float)
    time_step = ctx["time_step"]
    q_init_arm = ctx["q_init_arm"]
    goal_xy, goal_bearing = task._base_goal(target_ee_pos)
    goal_xy = np.asarray(goal_xy, dtype=float)

    sim.ResetPose()
    client.resetBaseVelocity(ctx["rid"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sim.ReceiveObservation()
    task._place_target_marker(ctx, target_ee_pos)

    stop_criteria = StopCriteria(reach_tolerance=task.REACH_TOLERANCE, max_steps=task.max_steps)
    cmd = MotorCommands()
    features = None
    buf = np.empty((1, len(FEATURE_NAMES)), dtype=np.float32)
    checked = False
    current_time = 0.0
    ee_pos = None
    stop_message = None
    time_all, wheel_all, torque_all = [], [], []

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ctx["ee_frame_id"]].translation.copy()
        ee_dist_error = float(np.linalg.norm(target_ee_pos - ee_pos))

        if features is None:
            features = FeatureBuffer(time_step, q_init_arm, base_pos, target_ee_pos)
        row = features.push(current_time, base_pos, base_bearing, q_mes[4:11], qd_mes[4:11])
        buf[0] = row
        if not checked:
            reference = np.asarray(model.predict(buf.copy(), iteration_range=iteration_range))
            if reference.ndim == 1:
                reference = reference.reshape(1, -1)
            wheel_cmds, torque_arm, pred = predict_command(booster, buf, iteration_range)
            if not np.allclose(reference, pred, rtol=0, atol=1e-4):
                diff = float(np.max(np.abs(reference - pred)))
                raise RuntimeError(
                    f"inplace_predict disagrees with the early-stopped model (max abs diff {diff:.3e})"
                )
            checked = True
        else:
            wheel_cmds, torque_arm, _pred = predict_command(booster, buf, iteration_range)

        cmd_all = np.concatenate((wheel_cmds, torque_arm, np.zeros(6)))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        navigation_done = reaching_window(
            base_pos, base_bearing, goal_xy, goal_bearing, current_time, ctx["base_params"]
        )
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, navigation_done)
        if stop_message:
            break

        if record:
            time_all.append(current_time)
            wheel_all.append(np.asarray(wheel_cmds, dtype=float).copy())
            torque_all.append(np.asarray(torque_arm, dtype=float).copy())

        if int(current_time / 5.0) != int((current_time - time_step) / 5.0) and current_time >= 5.0:
            print(f"    t={current_time:.0f}s  ee {ee_dist_error:.3f} m", flush=True)

        current_time += time_step

    result = {
        "target": target_ee_pos,
        "final_error": float(np.linalg.norm(target_ee_pos - ee_pos)),
        "final_ee_pos": np.asarray(ee_pos, dtype=float),
        "stop_message": stop_message or "",
        "time": current_time,
    }
    if record:
        result["time_all"] = np.asarray(time_all, dtype=float)
        result["wheel_cmds_all"] = np.asarray(wheel_all, dtype=float)
        result["torque_arm_all"] = np.asarray(torque_all, dtype=float)
    return result


def plot_comparison(targets, model_errors, expert_errors, path):
    known = np.isfinite(expert_errors)
    heights = np.unique(np.round(targets[:, 2], 5))
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.0, 1.2]}
    )
    scatter = axes[0]
    hi = max(float(np.nanmax(model_errors)), float(np.nanmax(expert_errors[known])))
    limit = max(hi * 1.08, task.REACH_TOLERANCE * 2)
    grid = np.linspace(0.0, limit, 40)
    scatter.plot(grid, grid, color="0.4", lw=1.0, label="Equal error")
    scatter.axhline(task.REACH_TOLERANCE, color="0.5", ls=":", lw=0.8)
    scatter.axvline(task.REACH_TOLERANCE, color="0.5", ls=":", lw=0.8)
    for z in heights:
        mask = known & np.isclose(targets[:, 2], z)
        scatter.scatter(
            expert_errors[mask], model_errors[mask], s=46, zorder=3, label=f"z = {z:.2f} m",
        )
    scatter.set_xlim(0.0, limit)
    scatter.set_ylim(0.0, limit)
    scatter.set_aspect("equal", adjustable="box")
    scatter.set_xlabel("Expert final EE error (m)")
    scatter.set_ylabel("Clone final EE error (m)")
    scatter.set_title("Each point is one assessment-grid target")
    scatter.grid(True, alpha=0.3)
    scatter.legend(loc="upper left", fontsize=8)

    bars = axes[1]
    x = np.arange(len(targets))
    width = 0.38
    bars.bar(x[known] - width / 2, expert_errors[known], width=width, label="Expert", color="C0")
    bars.bar(x + width / 2, model_errors, width=width, label="Clone", color="C1")
    bars.axhline(task.REACH_TOLERANCE, color="k", ls="--", lw=0.8, label="10 mm")
    bars.set_xticks(
        x,
        [f"({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})" for t in targets],
        rotation=75,
        ha="right",
        fontsize=7,
    )
    bars.set_ylabel("Final end-effector error (m)")
    bars.set_title("Same targets, grid order")
    bars.grid(True, axis="y", alpha=0.3)
    bars.legend(loc="upper right", fontsize=8)
    fig.suptitle("Closed-loop behaviour clone vs expert on the 24-point grid", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def report(targets, model_errors, expert_errors, messages):
    within = int(np.sum(model_errors <= task.REACH_TOLERANCE))
    print("==================================================================")
    print(f"Clone within {task.REACH_TOLERANCE * 1000:.0f} mm : {within}/{len(model_errors)}")
    print(
        "Clone error min / median / max (m) : "
        f"{model_errors.min():.4f} / {np.median(model_errors):.4f} / {model_errors.max():.4f}"
    )
    known = np.isfinite(expert_errors)
    if np.any(known):
        expert_known = expert_errors[known]
        within_expert = int(np.sum(expert_known <= task.REACH_TOLERANCE))
        print(f"Expert within {task.REACH_TOLERANCE * 1000:.0f} mm : {within_expert}/{len(expert_known)}")
        print(
            "Expert error min / median / max (m): "
            f"{expert_known.min():.4f} / {np.median(expert_known):.4f} / {expert_known.max():.4f}"
        )
        delta = model_errors[known] - expert_known
        print(
            "Clone minus expert min / median / max (m): "
            f"{delta.min():+.4f} / {np.median(delta):+.4f} / {delta.max():+.4f}"
        )
    else:
        print("No expert grid file found, so there is no per-target comparison.")
        print("Expected expert_grid.npz or expert_grid_part_*.npz next to the task script.")
    order = np.argsort(model_errors)[::-1]
    print(f"{'target':>26}  {'clone':>8}  {'expert':>8}  stop")
    for index in order:
        expert_text = f"{expert_errors[index]:8.4f}" if np.isfinite(expert_errors[index]) else f"{'—':>8}"
        xyz = targets[index]
        print(
            f"({xyz[0]:6.2f}, {xyz[1]:6.2f}, {xyz[2]:5.2f})  "
            f"{model_errors[index]:8.4f}  {expert_text}  {messages[index]}"
        )
    print("==================================================================")


def plot_signal_comparison(expert, clone):
    """Overlay expert and clone wheel commands and arm torques for one target.

    Both policies send ``[right, left, left, right]``, so the wheel figure shows
    the two distinct sides. Traces end when that rollout stops.
    """
    tag = task._target_tag(expert["target"])
    subtitle = (
        f"Expert error {expert['final_error']:.3f} m at t={expert['time']:.1f}s    "
        f"Clone error {clone['final_error']:.3f} m at t={clone['time']:.1f}s"
    )
    expert_t = expert["time_all"]
    clone_t = clone["time_all"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for ax, index, name in zip(axes, (0, 1), ("Right wheels", "Left wheels")):
        ax.plot(expert_t, expert["wheel_cmds_all"][:, index], color="C0", lw=1.3, label="Expert")
        ax.plot(clone_t, clone["wheel_cmds_all"][:, index], color="C1", lw=1.1, label="Clone")
        ax.set_ylabel("Wheel command (rad/s)")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Wheel commands — {tag}\n{subtitle}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    wheel_path = os.path.join(task.plots_dir("bc"), "bc_vs_expert_wheels.png")
    fig.savefig(wheel_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {wheel_path}")

    fig, axes = plt.subplots(4, 2, figsize=(10, 8.2), sharex=True)
    flat = axes.ravel()
    for i, ax in enumerate(flat[:7]):
        ax.plot(expert_t, expert["torque_arm_all"][:, i], color="C0", lw=1.15, label="Expert")
        ax.plot(clone_t, clone["torque_arm_all"][:, i], color="C1", lw=1.0, label="Clone")
        ax.set_ylabel(f"J{i + 1} (N·m)")
        ax.grid(True, alpha=0.3)
    flat[0].legend(loc="upper right", fontsize=8)
    flat[5].set_xlabel("Time (s)")
    flat[6].set_xlabel("Time (s)")
    flat[7].axis("off")
    fig.suptitle(f"Arm torques — {tag}\n{subtitle}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    torque_path = os.path.join(task.plots_dir("bc"), "bc_vs_expert_torques.png")
    fig.savefig(torque_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {torque_path}")
    return wheel_path, torque_path


def compare_random(model_dir):
    """One random assessment-grid target, expert then clone, from the default reset."""
    model, booster, iteration_range = load_policy(model_dir)
    targets = task.sparse_target_grid()
    index = int(np.random.default_rng().integers(len(targets)))
    target = targets[index]
    print(f"Random trajectory {index + 1}/{len(targets)}: {np.round(target, 3)}", flush=True)
    ctx = task.setup_simulation()
    try:
        print("Expert", flush=True)
        expert = task.run_episode(ctx, target, record=True, verbose=False, realtime=False)
        print(
            f"    error {expert['final_error']:.4f} m at t={expert['time']:.2f}s | {expert['stop_message']}",
            flush=True,
        )
        print("Clone", flush=True)
        wall_start = time.perf_counter()
        clone = run_model_episode(ctx, model, booster, iteration_range, target, record=True)
        print(
            f"    error {clone['final_error']:.4f} m at t={clone['time']:.2f}s "
            f"(wall {time.perf_counter() - wall_start:.1f}s) | {clone['stop_message']}",
            flush=True,
        )
    finally:
        ctx["client"].disconnect()
    plot_signal_comparison(expert, clone)


def main():
    args = parse_args()
    if args.compare_random:
        compare_random(args.model)
        return
    model, booster, iteration_range = load_policy(args.model)
    print(f"Policy trees {iteration_range[0]}:{iteration_range[1]}", flush=True)
    targets = np.stack([np.asarray(point, dtype=float) for point in task.sparse_target_grid()])
    expert = load_expert_grid()
    if expert is None:
        expert_errors = np.full(len(targets), np.nan)
        print("No saved expert grid results; comparison figure will be skipped if none match.", flush=True)
    else:
        expert_errors = align_expert(targets, expert[0], expert[1])
        print(f"Matched expert errors for {int(np.isfinite(expert_errors).sum())}/{len(targets)} targets", flush=True)

    ctx = task.setup_simulation()
    results = []
    try:
        for i, target in enumerate(targets):
            print(f"[{i + 1}/{len(targets)}] target {np.round(target, 3)}", flush=True)
            wall_start = time.perf_counter()
            result = run_model_episode(ctx, model, booster, iteration_range, target)
            results.append(result)
            print(
                f"    error {result['final_error']:.4f} m at t={result['time']:.2f}s "
                f"(wall {time.perf_counter() - wall_start:.1f}s) | {result['stop_message']}",
                flush=True,
            )
    finally:
        ctx["client"].disconnect()

    model_errors = np.array([result["final_error"] for result in results])
    messages = np.array([result["stop_message"] for result in results])
    scored = np.vstack([result["target"] for result in results])
    out_npz = os.path.join(TASK_DIR, "bc_eval_grid.npz")
    np.savez(
        out_npz,
        targets=scored,
        errors=model_errors,
        expert_errors=expert_errors,
        messages=messages,
        times=np.array([result["time"] for result in results]),
    )
    print(f"Saved {out_npz}")

    fig = task.plot_targets_by_error(
        scored,
        model_errors,
        title=(
            "Behaviour clone on the assessment grid — colour is final EE error (linear scale)\n"
            "Black + is the robot start. Slice labels are the error in metres."
        ),
    )
    grid_path = os.path.join(task.plots_dir("bc"), "bc_grid_error.png")
    fig.savefig(grid_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {grid_path}")

    if np.any(np.isfinite(expert_errors)):
        plot_comparison(scored, model_errors, expert_errors, os.path.join(task.plots_dir("bc"), "bc_vs_expert.png"))
    report(scored, model_errors, expert_errors, messages)


if __name__ == "__main__":
    main()
