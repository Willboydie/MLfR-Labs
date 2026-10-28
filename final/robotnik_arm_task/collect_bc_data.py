"""Log expert demonstrations for behaviour cloning.

The loop mirrors ``robotnik_arm_task.run_episode`` and calls ``arm_controller``
and ``base_controller`` as they are. That function does not record ``base_pos``,
``base_bearing`` or arm joint velocity, so the dataset is written here. The
expert is unchanged: no action noise, one episode per target, each from the
default reset pose.

Targets are the 24-point assessment grid plus half-cell jitter, kept inside
the grid's x/y/z span and at least 0.4 m horizontally from the robot. Re-running
skips episodes that are already on disk.

From the RoboEnv checkout::

    pixi run python ../final/robotnik_arm_task/collect_bc_data.py
"""
import os
import sys

if "--gui" not in sys.argv:
    os.environ.setdefault("ROBOENV_HEADLESS", "1")
else:
    os.environ["ROBOENV_HEADLESS"] = "0"

import argparse
import time

import numpy as np
import pinocchio as pin
from simulation_and_control import MotorCommands

from robotnik_arm_control import (
    BaseNavState,
    StopCriteria,
    arm_controller,
    base_controller,
    check_stop,
    quaternion2bearing,
)
import robotnik_arm_task as task
from bc_features import DATA_DIR, N_TARGETS, SEED, episode_path, meta_path, sample_bc_targets


def parse_args():
    parser = argparse.ArgumentParser(description="Log expert reaching demonstrations.")
    parser.add_argument("--out", default=DATA_DIR, help="Directory for episode npz files.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-targets", type=int, default=N_TARGETS)
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many new episodes (0 = all missing).")
    parser.add_argument("--shard", default="", help="Run every n-th episode, as i/n. The target list is shared.")
    parser.add_argument("--gui", action="store_true", help="Open the PyBullet GUI.")
    return parser.parse_args()


def episode_indices(n, shard):
    indices = list(range(n))
    if not shard:
        return indices
    index, count = (int(part) for part in shard.split("/"))
    if count < 1 or not 0 <= index < count:
        raise SystemExit(f"Bad --shard {shard!r}; expected i/n with 0 <= i < n")
    return indices[index::count]


def ensure_targets(data_dir, n, seed, dt):
    """Create the shared target list, or reload it so a resumed run cannot drift."""
    path = meta_path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    if os.path.isfile(path):
        meta = np.load(path)
        if int(meta["seed"]) != seed or len(meta["targets"]) != n:
            raise SystemExit(
                f"{path} was built with seed={int(meta['seed'])} and {len(meta['targets'])} targets; "
                f"this run asked for seed={seed} and {n}."
            )
        if abs(float(meta["dt"]) - dt) > 1e-12:
            raise SystemExit(f"Simulation dt {dt} does not match the saved dataset dt {float(meta['dt'])}")
        return meta["targets"], meta["anchor_index"]

    targets, anchors, grid = sample_bc_targets(n, seed)
    with open(path, "wb") as handle:
        np.savez(
            handle,
            targets=targets,
            anchor_index=anchors,
            seed=np.int64(seed),
            dt=np.float64(dt),
            grid=grid,
        )
    horiz = np.hypot(targets[:, 0], targets[:, 1])
    print(
        f"Sampled {n} targets from {len(grid)} grid anchors (seed {seed}). "
        f"x [{targets[:, 0].min():.3f}, {targets[:, 0].max():.3f}]  "
        f"y [{targets[:, 1].min():.3f}, {targets[:, 1].max():.3f}]  "
        f"z [{targets[:, 2].min():.3f}, {targets[:, 2].max():.3f}]  "
        f"horizontal radius [{horiz.min():.3f}, {horiz.max():.3f}] m",
        flush=True,
    )
    return targets, anchors


def log_episode(ctx, target_ee_pos):
    """One expert episode. Same control flow as ``run_episode``, with a fuller log.

    The sample is the pre-step state and the command computed from it. A step
    that already meets a stop condition is not logged, matching ``run_episode``.
    """
    sim = ctx["sim"]
    client = ctx["client"]
    dyn_model = ctx["dyn_model"]
    target_ee_pos = np.asarray(target_ee_pos, dtype=float)
    time_step = ctx["time_step"]

    sim.ResetPose()
    client.resetBaseVelocity(ctx["rid"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sim.ReceiveObservation()
    task._place_target_marker(ctx, target_ee_pos)

    base_goal_pos, base_goal_bearing = task._base_goal(target_ee_pos)
    base_state = BaseNavState()
    q_init_arm = ctx["q_init_arm"]
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    stop_criteria = StopCriteria(reach_tolerance=task.REACH_TOLERANCE, max_steps=task.max_steps)
    cmd = MotorCommands()

    time_all = []
    base_pos_all = []
    base_bearing_all = []
    q_arm_all = []
    qd_arm_all = []
    wheel_cmds_all = []
    torque_arm_all = []
    phase_all = []
    q_des_arm_all = []

    current_time = 0.0
    ee_pos = None
    base_pos = None
    stop_message = None
    user_quit = False

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

        wheel_cmds = base_controller(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing,
            current_time, ctx["base_params"],
        )
        base_navigation_done = base_state.done
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target_ee_pos, ee_pos,
            ctx["ee_frame_id"], ctx["shoulder_frame_id"], ctx["arm_lower"], ctx["arm_upper"], time_step,
            ctx["arm_params"], reach_active=base_navigation_done,
        )
        torque_fingers = np.zeros(6)

        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                dyn_model.DisplayModel(sim.GetMotorAngles(index))

        keys = sim.GetPyBulletClient().getKeyboardEvents()
        q_key = ord("q")
        if q_key in keys and keys[q_key] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            user_quit = True
            stop_message = "Stopped by user"
            break

        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, base_navigation_done)
        if stop_message:
            break

        time_all.append(current_time)
        base_pos_all.append(base_pos.copy())
        base_bearing_all.append(float(base_bearing))
        q_arm_all.append(np.asarray(q_mes[4:11], dtype=float).copy())
        qd_arm_all.append(np.asarray(qd_mes[4:11], dtype=float).copy())
        wheel_cmds_all.append(np.asarray(wheel_cmds, dtype=float).copy())
        torque_arm_all.append(np.asarray(torque_arm, dtype=float).copy())
        phase_all.append("done" if base_state.done else base_state.phase)
        q_des_arm_all.append(np.asarray(q_des_arm, dtype=float).copy())

        if time_all and int(current_time / 5.0) != int((current_time - time_step) / 5.0) and current_time >= 5.0:
            phase = phase_all[-1]
            print(
                f"    t={current_time:.0f}s  ee {ee_dist_error:.3f} m  phase {phase}",
                flush=True,
            )

        current_time += time_step

    if not time_all:
        raise RuntimeError("Episode ended before any sample was logged")

    final_error = float(np.linalg.norm(target_ee_pos - ee_pos))
    return {
        "time": np.asarray(time_all, dtype=np.float64),
        "base_pos": np.asarray(base_pos_all, dtype=np.float64),
        "base_bearing": np.asarray(base_bearing_all, dtype=np.float64),
        "q_arm": np.asarray(q_arm_all, dtype=np.float64),
        "qd_arm": np.asarray(qd_arm_all, dtype=np.float64),
        "target": target_ee_pos,
        "wheel_cmds": np.asarray(wheel_cmds_all, dtype=np.float64),
        "torque_arm": np.asarray(torque_arm_all, dtype=np.float64),
        "phase": np.asarray(phase_all, dtype="U16"),
        "q_des_arm": np.asarray(q_des_arm_all, dtype=np.float64),
        "q_init_arm": np.asarray(q_init_arm, dtype=np.float64),
        "start_base_pos": np.asarray(base_pos_all[0], dtype=np.float64),
        "dt": np.float64(time_step),
        "final_error": np.float64(final_error),
        "stop_message": np.asarray(stop_message or ""),
        "user_quit": user_quit,
        "final_time": current_time,
    }


def save_episode(path, episode, anchor_index):
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            time=episode["time"],
            base_pos=episode["base_pos"],
            base_bearing=episode["base_bearing"],
            q_arm=episode["q_arm"],
            qd_arm=episode["qd_arm"],
            target=episode["target"],
            wheel_cmds=episode["wheel_cmds"],
            torque_arm=episode["torque_arm"],
            phase=episode["phase"],
            q_des_arm=episode["q_des_arm"],
            q_init_arm=episode["q_init_arm"],
            start_base_pos=episode["start_base_pos"],
            dt=episode["dt"],
            final_error=episode["final_error"],
            stop_message=episode["stop_message"],
            anchor_index=np.int64(anchor_index),
        )
    os.replace(tmp, path)


def main():
    args = parse_args()
    indices = episode_indices(args.n_targets, args.shard)
    ctx = task.setup_simulation()
    ran = 0
    try:
        targets, anchors = ensure_targets(args.out, args.n_targets, args.seed, ctx["time_step"])
        print(f"Logging {len(indices)} episode slots into {args.out}", flush=True)
        for index in indices:
            path = episode_path(args.out, index)
            if os.path.isfile(path):
                print(f"[{index + 1}/{args.n_targets}] already logged", flush=True)
                continue
            if args.limit and ran >= args.limit:
                break
            target = targets[index]
            print(f"[{index + 1}/{args.n_targets}] target {np.round(target, 3)}", flush=True)
            wall_start = time.perf_counter()
            episode = log_episode(ctx, target)
            save_episode(path, episode, int(anchors[index]))
            ran += 1
            print(
                f"    error {float(episode['final_error']):.4f} m at t={episode['final_time']:.2f}s "
                f"(wall {time.perf_counter() - wall_start:.1f}s, {len(episode['time'])} samples) "
                f"| {episode['stop_message']}",
                flush=True,
            )
            if episode["user_quit"]:
                break
    finally:
        ctx["client"].disconnect()
    print(f"Finished. New episodes this run: {ran}", flush=True)


if __name__ == "__main__":
    main()
