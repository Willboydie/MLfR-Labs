"""DAgger for the tanh-128x128 network with batch size 64.

Iteration 0 is the behaviour-cloning policy trained on all 150 expert
episodes. Each later iteration rolls that policy out on new targets drawn
from the same two boxes as the expert data, asks the Markov expert what it
would have commanded in those states, and retrains on the aggregated set.
The 27-point test lattice is not used here.

The plant always receives the learner. The stored label is the expert
command. The integrator z is the one the learner is carrying, updated from
the expert's cartesian error exactly as at playback. Rows are kept every
25 ms, matching the cross-validation.

From the RoboEnv checkout::

    pixi run python -u ../final/robotnik_arm_task/dagger_markov_mlp.py
"""
import os

os.environ["ROBOENV_HEADLESS"] = "1"

import json
import time

import numpy as np
import pinocchio as pin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from simulation_and_control import MotorCommands

from collect_markov_data import (
    box_bounds,
    clean_wheel_pair,
    measure_reset,
    near_bounds,
    sample_targets,
)
from cv_markov_mlp import MAX_ITER, N_ITER_NO_CHANGE, SAMPLE_STRIDE_S, SEED, shuffle_rows
from markov_mlp import predict_command
from robotnik_arm_control import quaternion2bearing
from robotnik_markov_expert import (
    StopCriteria,
    check_stop,
    integrator_update,
    observation_vector,
    relative_target,
)
import robotnik_arm_task as task

# The All-dataset winner. Learning rate and L2 stay at the values that search fixed.
HIDDEN = (128, 128)
ACTIVATION = "tanh"
LR = 1e-3
ALPHA = 1e-4
BATCH_SIZE = 64
N_ITERS = 10
N_WIDE = 7
N_NEAR = 3
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(TASK_DIR, "markov_data")
N_SOURCE_EPISODES = 150
OUT_DIR = os.path.join(TASK_DIR, "markov_dagger")
BC_POLICY = os.path.join(TASK_DIR, "markov_mlp_cv_all", "winner.joblib")
WINNER_PATH = os.path.join(OUT_DIR, "winner.joblib")


def source_episode_path(index):
    return os.path.join(SOURCE_DIR, f"episode_{index:03d}.npz")


def policy_path(iteration):
    return os.path.join(OUT_DIR, f"policy_iter_{iteration:02d}.joblib")


def targets_path(iteration):
    return os.path.join(OUT_DIR, f"iter_{iteration:02d}_targets.npy")


def episode_path(iteration, index):
    return os.path.join(OUT_DIR, f"iter_{iteration:02d}", f"episode_{index:02d}.npz")


def expert_cache_path():
    return os.path.join(OUT_DIR, "expert_rows.npz")


def log_path():
    return os.path.join(OUT_DIR, "iterations.json")


def load_log():
    path = log_path()
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_log(rows):
    tmp = log_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    os.replace(tmp, log_path())


def _check_bc_policy(policy):
    if policy.get("name") != "tanh-128x128-b64":
        raise SystemExit(f"Behaviour-cloning file is {policy.get('name')!r}, not tanh-128x128-b64.")
    if tuple(policy.get("hidden", ())) != HIDDEN or policy.get("activation") != ACTIVATION:
        raise SystemExit("Behaviour-cloning file is not tanh-128x128.")
    if int(policy.get("batch_size", -1)) != BATCH_SIZE:
        raise SystemExit("Behaviour-cloning file was not trained at batch size 64.")
    if abs(float(policy["lr"]) - LR) > 1e-12 or abs(float(policy["alpha"]) - ALPHA) > 1e-12:
        raise SystemExit("Behaviour-cloning file does not use lr 1e-3 and L2 1e-4.")


def ensure_initial_policy():
    """Iteration 0 is the network already trained on the 150 expert episodes."""
    import joblib

    os.makedirs(OUT_DIR, exist_ok=True)
    path = policy_path(0)
    if os.path.isfile(path):
        policy = joblib.load(path)
    else:
        if not os.path.isfile(BC_POLICY):
            raise SystemExit(f"No behaviour-cloning policy at {BC_POLICY}.")
        policy = joblib.load(BC_POLICY)
        _check_bc_policy(policy)
        joblib.dump(policy, path)
    _check_bc_policy(policy)
    return policy


def load_expert_rows():
    """25 ms rows from the 150 expert episodes. Cached so a resume does not reread them."""
    path = expert_cache_path()
    if os.path.isfile(path):
        with np.load(path) as cached:
            if int(cached["n_episodes"]) != N_SOURCE_EPISODES:
                raise SystemExit(f"{path} does not match the {N_SOURCE_EPISODES} expert episodes.")
            return np.asarray(cached["X"]), np.asarray(cached["Y"])
    with np.load(os.path.join(SOURCE_DIR, "dataset_meta.npz")) as meta:
        dt = float(meta["dt"])
    stride = int(round(SAMPLE_STRIDE_S / dt))
    if stride < 1 or abs(stride * dt - SAMPLE_STRIDE_S) > 1e-9:
        raise SystemExit(f"25 ms is not an integer number of steps at dt={dt}.")
    features, labels = [], []
    for index in range(N_SOURCE_EPISODES):
        with np.load(source_episode_path(index)) as episode:
            obs = np.asarray(episode["observation"], dtype=np.float64)[::stride]
            wheels = np.asarray(episode["wheel_cmd"], dtype=np.float64)[::stride]
            torque = np.asarray(episode["torque_arm"], dtype=np.float64)[::stride]
        features.append(obs)
        labels.append(np.hstack((wheels, torque)))
    features = np.vstack(features)
    labels = np.vstack(labels)
    np.savez(path, X=features, Y=labels, n_episodes=np.int64(N_SOURCE_EPISODES))
    print(f"Cached {len(features)} expert samples from {N_SOURCE_EPISODES} episodes.", flush=True)
    return features, labels


def draw_targets(iteration, base_pos, shoulder_z):
    """Seven targets in the 2 m box and three in the near square. Fixed by the iteration."""
    path = targets_path(iteration)
    if os.path.isfile(path):
        return np.load(path)
    wide_lo, wide_hi = box_bounds(base_pos, shoulder_z)
    near_lo, near_hi = near_bounds(base_pos)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 11, iteration]))
    wide = sample_targets(N_WIDE, wide_lo, wide_hi, rng)
    near = sample_targets(N_NEAR, near_lo, near_hi, rng)
    targets = np.vstack((wide, near))
    np.save(path, targets)
    return targets


def rollout(ctx, policy, target, z_max):
    """Run the learner. Store the expert command at the states the learner visits."""
    sim = ctx["sim"]
    client = ctx["client"]
    dyn_model = ctx["dyn_model"]
    target = np.asarray(target, dtype=float)
    time_step = ctx["time_step"]
    stride = int(round(SAMPLE_STRIDE_S / time_step))

    sim.ResetPose()
    client.resetBaseVelocity(ctx["rid"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sim.ReceiveObservation()
    task._place_target_marker(ctx, target)

    expert = task._markov_expert(ctx, target)
    if abs(float(expert.arm_params.z_max) - z_max) > 1e-12:
        raise RuntimeError("Integrator limit changed since the behaviour-cloning policy was saved.")
    stop_criteria = StopCriteria(reach_tolerance=task.REACH_TOLERANCE, max_steps=task.max_steps)
    cmd_plant = MotorCommands()
    z = np.zeros(3)

    observations, expert_wheels, expert_torques = [], [], []
    learner_wheels, learner_torques = [], []
    current_time = 0.0
    step_index = 0
    ee_pos = None
    ee_dist = None
    stop_message = None

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))
        q_arm = np.asarray(q_mes[4:11], dtype=float)
        qd_arm = np.asarray(qd_mes[4:11], dtype=float)
        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ctx["ee_frame_id"]].translation.copy()
        ee_dist = float(np.linalg.norm(target - ee_pos))

        z_in = np.array(z, dtype=float, copy=True)
        target_rel = relative_target(target, base_pos, base_bearing)
        cmd = expert(q_pin, q_arm, qd_arm, target_rel, z_in)
        obs = np.asarray(observation_vector(target_rel, q_pin, q_arm, qd_arm, z_in), dtype=np.float64)
        wheel_label = clean_wheel_pair(cmd.wheel_cmd)
        torque_label = np.array(cmd.torque_arm, dtype=np.float64, copy=True)
        wheel_apply, torque_apply = predict_command(policy, obs)
        if step_index % stride == 0:
            observations.append(obs)
            expert_wheels.append(wheel_label)
            expert_torques.append(torque_label)
            learner_wheels.append(np.array(wheel_apply[:2], dtype=np.float64, copy=True))
            learner_torques.append(np.array(torque_apply, dtype=np.float64, copy=True))
        z = integrator_update(z_in, cmd.cartesian_error, cmd.reach_weight, time_step, z_max)

        cmd_plant.SetControlCmd(
            np.concatenate((wheel_apply, torque_apply, np.zeros(6))),
            ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6,
        )
        sim.Step(cmd_plant, "torque")
        arm_engaged = cmd.reach_weight >= 0.999
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist, arm_engaged)
        if stop_message:
            break
        if int(current_time / 5.0) != int((current_time - time_step) / 5.0) and current_time >= 5.0:
            print(f"    t={current_time:.0f}s  ee {ee_dist:.3f} m", flush=True)
        current_time += time_step
        step_index += 1

    if not observations:
        raise RuntimeError("Rollout ended before any sample was kept.")
    expert_w = np.vstack(expert_wheels)
    expert_t = np.vstack(expert_torques)
    learn_w = np.vstack(learner_wheels)
    learn_t = np.vstack(learner_torques)
    message = stop_message or ""
    return {
        "observation": np.vstack(observations),
        "wheel_cmd": expert_w,
        "torque_arm": expert_t,
        "target": target,
        "reached": bool(message.startswith("Target reached")),
        "final_error": float(ee_dist),
        "final_time": float(current_time),
        "stop_message": message,
        "wheel_disagreement": float(np.mean(np.abs(learn_w - expert_w))),
        "torque_disagreement": float(np.mean(np.abs(learn_t - expert_t))),
    }


def save_episode(path, episode, iteration):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            observation=episode["observation"],
            wheel_cmd=episode["wheel_cmd"],
            torque_arm=episode["torque_arm"],
            target=episode["target"],
            reached=np.bool_(episode["reached"]),
            final_error=np.float64(episode["final_error"]),
            final_time=np.float64(episode["final_time"]),
            stop_message=np.asarray(episode["stop_message"]),
            iteration=np.int64(iteration),
            wheel_disagreement=np.float64(episode["wheel_disagreement"]),
            torque_disagreement=np.float64(episode["torque_disagreement"]),
        )
    os.replace(tmp, path)


def collect_iteration(ctx, policy, iteration, z_max):
    base_pos, shoulder = measure_reset(ctx)
    targets = draw_targets(iteration, base_pos, shoulder[2])
    print(
        f"Iteration {iteration}: {len(targets)} rollouts "
        f"({N_WIDE} in the 2 m box, {N_NEAR} in the near square).",
        flush=True,
    )
    summaries = []
    for index, target in enumerate(targets):
        path = episode_path(iteration, index)
        if os.path.isfile(path):
            with np.load(path) as saved:
                summaries.append({
                    "reached": bool(saved["reached"]),
                    "final_error": float(saved["final_error"]),
                    "n_samples": int(len(saved["observation"])),
                    "wheel_disagreement": float(saved["wheel_disagreement"]),
                    "torque_disagreement": float(saved["torque_disagreement"]),
                })
            print(f"  [{index + 1}/{len(targets)}] already logged", flush=True)
            continue
        print(f"  [{index + 1}/{len(targets)}] target {np.round(target, 3)}", flush=True)
        started = time.perf_counter()
        episode = rollout(ctx, policy, target, z_max)
        save_episode(path, episode, iteration)
        summaries.append({
            "reached": episode["reached"],
            "final_error": episode["final_error"],
            "n_samples": int(len(episode["observation"])),
            "wheel_disagreement": episode["wheel_disagreement"],
            "torque_disagreement": episode["torque_disagreement"],
        })
        print(
            f"    {'reached' if episode['reached'] else 'missed'}  "
            f"error {episode['final_error']:.4f} m at t={episode['final_time']:.2f}s  "
            f"wheel diff {episode['wheel_disagreement']:.3f}  "
            f"torque diff {episode['torque_disagreement']:.2f}  "
            f"({time.perf_counter() - started:.0f}s wall, {len(episode['observation'])} rows)",
            flush=True,
        )
    return summaries


def load_dagger_rows(through):
    features, labels = [], []
    for iteration in range(1, through + 1):
        for index in range(N_WIDE + N_NEAR):
            path = episode_path(iteration, index)
            if not os.path.isfile(path):
                raise SystemExit(f"Missing DAgger rollout {path}.")
            with np.load(path) as episode:
                obs = np.asarray(episode["observation"], dtype=np.float64)
                wheels = np.asarray(episode["wheel_cmd"], dtype=np.float64)
                torque = np.asarray(episode["torque_arm"], dtype=np.float64)
            features.append(obs)
            labels.append(np.hstack((wheels, torque)))
    if not features:
        return np.zeros((0, 22)), np.zeros((0, 9))
    return np.vstack(features), np.vstack(labels)


def fit_policy(features, labels, limits, iteration):
    import joblib

    rng = np.random.default_rng(SEED)
    features, labels = shuffle_rows(features, labels, rng)
    x_scaler = StandardScaler().fit(features)
    y_scaler = StandardScaler().fit(labels)
    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation=ACTIVATION,
        alpha=ALPHA,
        learning_rate_init=LR,
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=N_ITER_NO_CHANGE,
        shuffle=True,
        random_state=SEED,
        solver="adam",
    )
    print(f"Training iteration {iteration} on {len(features)} samples.", flush=True)
    started = time.perf_counter()
    model.fit(x_scaler.transform(features), y_scaler.transform(labels))
    policy = {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "name": "tanh-128x128-b64-dagger",
        "hidden": list(HIDDEN),
        "activation": ACTIVATION,
        "lr": LR,
        "alpha": ALPHA,
        "batch_size": BATCH_SIZE,
        "iteration": int(iteration),
        "n_samples": int(len(features)),
        "wheel_limit": float(limits["wheel_limit"]),
        "torque_limit": float(limits["torque_limit"]),
        "sample_stride_s": SAMPLE_STRIDE_S,
        "n_iter": int(model.n_iter_),
    }
    path = policy_path(iteration)
    joblib.dump(policy, path)
    print(
        f"Saved {path} after {model.n_iter_} iterations ({time.perf_counter() - started:.1f}s).",
        flush=True,
    )
    return policy, float(time.perf_counter() - started)


def main():
    import joblib

    os.makedirs(OUT_DIR, exist_ok=True)
    initial = ensure_initial_policy()
    limits = {"wheel_limit": float(initial["wheel_limit"]), "torque_limit": float(initial["torque_limit"])}
    with np.load(os.path.join(SOURCE_DIR, "dataset_meta.npz")) as meta:
        z_max = float(meta["z_max"])
    expert_x, expert_y = load_expert_rows()
    log = load_log()
    ctx = task.setup_simulation()
    try:
        policy = initial
        for iteration in range(1, N_ITERS + 1):
            summaries = collect_iteration(ctx, policy, iteration, z_max)
            if os.path.isfile(policy_path(iteration)):
                policy = joblib.load(policy_path(iteration))
                print(f"Iteration {iteration} policy already trained.", flush=True)
                if any(row.get("iteration") == iteration for row in log):
                    continue
                extra_count = int(sum(row["n_samples"] for row in summaries))
                train_seconds = None
                n_iter = int(policy["n_iter"])
                n_total = int(policy["n_samples"])
            else:
                extra_x, extra_y = load_dagger_rows(iteration)
                policy, train_seconds = fit_policy(
                    np.vstack((expert_x, extra_x)), np.vstack((expert_y, extra_y)), limits, iteration,
                )
                extra_count = int(sum(row["n_samples"] for row in summaries))
                n_iter = int(policy["n_iter"])
                n_total = int(len(expert_x) + len(extra_x))
            errors = np.array([row["final_error"] for row in summaries], dtype=float)
            record = {
                "iteration": iteration,
                "n_rollouts": len(summaries),
                "n_reached": int(sum(row["reached"] for row in summaries)),
                "median_error": float(np.median(errors)),
                "mean_error": float(errors.mean()),
                "n_new_samples": extra_count,
                "n_total_samples": n_total,
                "mean_wheel_disagreement": float(np.mean([row["wheel_disagreement"] for row in summaries])),
                "mean_torque_disagreement": float(np.mean([row["torque_disagreement"] for row in summaries])),
                "n_iter": n_iter,
                "train_seconds": train_seconds,
            }
            log = [row for row in log if row.get("iteration") != iteration]
            log.append(record)
            log.sort(key=lambda row: row["iteration"])
            save_log(log)
            print(
                f"Iteration {iteration}: reached {record['n_reached']}/{record['n_rollouts']}, "
                f"median error {record['median_error']:.3f} m, "
                f"{record['n_new_samples']} new samples.",
                flush=True,
            )
    finally:
        ctx["client"].disconnect()
    final = joblib.load(policy_path(N_ITERS))
    joblib.dump(final, WINNER_PATH)
    print(f"Saved {WINNER_PATH}", flush=True)


if __name__ == "__main__":
    main()
