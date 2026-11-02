"""One Markov episode at (-1.2, 1.2, 0.7): twist, attitude, path, wheel limits.

The robot starts at the default pose. Logs are written next to the other figures
for this target, under plots/markov_vs_original/-1.20_1.20_0.70/.

    cd RoboEnv
    pixi run python ../final/robotnik_arm_task/log_markov_base.py
"""
import os

os.environ["ROBOENV_HEADLESS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

import robotnik_arm_task as task
from robotnik_markov_expert import (
    StopCriteria,
    base_errors,
    base_twist,
    brake_scale,
    check_stop,
    integrator_update,
    relative_target,
    wrap_angle,
)

TARGET = np.array([-1.2, 1.2, 0.7])
WHEEL_NAMES = (
    "robot_front_right_wheel_joint",
    "robot_front_left_wheel_joint",
    "robot_back_left_wheel_joint",
    "robot_back_right_wheel_joint",
)
WHEEL_LABELS = ("front right", "front left", "back left", "back right")
# Velocity loop gain set in setup_simulation. Torque = kd * (command - measured speed).
WHEEL_KD = 30.0
SMOOTH_STEPS = 50  # 50 ms at the 1 ms step, for the plotted measured traces


def _rpy(ori):
    quat = pin.Quaternion(float(ori[3]), float(ori[0]), float(ori[1]), float(ori[2]))
    quat.normalize()
    return pin.rpy.matrixToRpy(quat.toRotationMatrix())


def _pin_yaw(q_pin):
    quat = pin.Quaternion(float(q_pin[6]), float(q_pin[3]), float(q_pin[4]), float(q_pin[5]))
    quat.normalize()
    return float(pin.rpy.matrixToRpy(quat.toRotationMatrix())[2])


def _smooth(values, width):
    kernel = np.ones(width) / width
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def _wheels_to_twist(qd, radius, width):
    """Inverse of twist_to_wheels. Right wheels are indices 0 and 3, left are 1 and 2."""
    qd = np.asarray(qd, dtype=float)
    right = 0.5 * (qd[0] + qd[3])
    left = 0.5 * (qd[1] + qd[2])
    forward = radius * 0.5 * (right + left)
    yaw_rate = radius * (right - left) / width
    return forward, yaw_rate


def _link_names(client, rid):
    names = {}
    for joint in range(client.getNumJoints(rid)):
        info = client.getJointInfo(rid, joint)
        names[joint] = info[12].decode()
    return names


def _wheel_limits(client, rid):
    """URDF velocity and effort as PyBullet reports them for the four drive joints."""
    rows = []
    for name in WHEEL_NAMES:
        joint = next(
            j for j in range(client.getNumJoints(rid))
            if client.getJointInfo(rid, j)[1].decode() == name
        )
        info = client.getJointInfo(rid, joint)
        rows.append({
            "joint": name,
            "index": joint,
            "effort": float(info[10]),
            "velocity": float(info[11]),
            "damping": float(info[6]),
            "friction": float(info[7]),
        })
    return rows


def _contacts(client, rid, link_names):
    """Normal force summed per link. Each contact is counted once."""
    found = {}
    seen = set()
    points = list(client.getContactPoints(bodyA=rid))
    points.extend(client.getContactPoints(bodyB=rid))
    for point in points:
        if point[1] == rid:
            link = point[3]
            other = point[2]
            position = point[5]
        elif point[2] == rid:
            link = point[4]
            other = point[1]
            position = point[6]
        else:
            continue
        key = (int(link), int(other), round(position[0], 3), round(position[1], 3), round(position[2], 3))
        if key in seen:
            continue
        seen.add(key)
        name = "base" if link < 0 else link_names.get(int(link), f"link {link}")
        found[name] = found.get(name, 0.0) + float(point[9])
    return found


def _commanded_twist(target_rel, expert):
    """Forward speed and yaw rate after the parking brake, the twist sent to the wheels."""
    params = expert.base_params
    gate = expert.gate_params
    dist_err, bearing = base_errors(target_rel, params)
    v_cmd, w_cmd = base_twist(target_rel, params)
    v_cmd *= brake_scale(abs(dist_err), gate.pos_inner, gate.pos_brake_edge)
    w_cmd *= brake_scale(abs(bearing), gate.ang_inner, gate.ang_brake_edge)
    return float(v_cmd), float(w_cmd), float(bearing)


def run_log(ctx):
    sim = ctx["sim"]
    client = ctx["client"]
    dyn_model = ctx["dyn_model"]
    rid = ctx["rid"]
    time_step = float(ctx["time_step"])
    target = TARGET.copy()
    link_names = _link_names(client, rid)
    limits = _wheel_limits(client, rid)
    torque_limits = sim.bot[0].servo_motor_model._torque_limits

    sim.ResetPose()
    client.resetBaseVelocity(rid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sim.ReceiveObservation()

    expert = task._markov_expert(ctx, target)
    footprint_id = None
    if dyn_model.pin_model.existFrame("robot_base_footprint"):
        footprint_id = dyn_model.pin_model.getFrameId("robot_base_footprint")
    z = np.zeros(3)
    stop_criteria = StopCriteria(reach_tolerance=task.REACH_TOLERANCE, max_steps=task.max_steps)
    radius = expert.base_params.wheel_radius
    width = expert.base_params.wheel_base_width

    columns = {name: [] for name in (
        "time", "x", "y", "z", "roll", "pitch", "yaw", "yaw_pin",
        "v_cmd", "w_cmd", "v_meas", "w_meas", "v_body", "w_body",
        "v_wheels", "w_wheels", "alpha", "alpha_geom", "ee_dist", "reach_w",
        "foot_x", "foot_y",
    )}
    active = list(sim.getNameActiveJoints())
    if active[:4] != list(WHEEL_NAMES):
        raise RuntimeError(f"Expected the first four motors to be the drive wheels, got {active[:4]}")

    wheel_cmd_all = []
    wheel_qd_all = []
    wheel_qd_pre_all = []
    wheel_tau_all = []
    wheel_force_all = []
    other_force_peak = {}
    kink_force_peak = {}

    current_time = 0.0
    step_index = 0
    stop_message = None
    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        roll, pitch, yaw = _rpy(base_ori)
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))
        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ctx["ee_frame_id"]].translation.copy()
        if footprint_id is None:
            foot = base_pos
        else:
            foot = dyn_model.pin_data.oMf[footprint_id].translation

        target_rel = relative_target(target, base_pos, yaw)
        cmd = expert(q_pin, q_mes[4:11], qd_mes[4:11], target_rel, z)
        v_cmd, w_cmd, alpha = _commanded_twist(target_rel, expert)
        delta = target[:2] - base_pos[:2]
        alpha_geom = wrap_angle(np.arctan2(delta[1], delta[0]) - yaw)
        z = integrator_update(
            z, cmd.cartesian_error, cmd.reach_weight, time_step, expert.arm_params.z_max,
        )

        wheel_cmds = np.asarray(cmd.wheel_cmd, dtype=float)
        cmd_all = np.concatenate((wheel_cmds, cmd.torque_arm, np.zeros(6)))
        motor = task.MotorCommands()
        motor.SetControlCmd(cmd_all, ["velocity"] * 4 + ["torque"] * 13)
        sim.Step(motor, "torque")

        lin = np.asarray(sim.GetBaseLinVelocity(), dtype=float)
        ang = np.asarray(sim.GetBaseAngVelocity(), dtype=float)
        lin_body = np.asarray(sim.GetBaseLinVelocityBodyFrame(), dtype=float)
        ang_body = np.asarray(sim.GetBaseAngVelocityBodyFrame(), dtype=float)
        c, s = np.cos(yaw), np.sin(yaw)
        v_meas = float(c * lin[0] + s * lin[1])
        w_meas = float(ang[2])
        qd_now = np.asarray(sim.GetMotorVelocities(0)[:4], dtype=float)
        v_wheels, w_wheels = _wheels_to_twist(qd_now, radius, width)
        tau = np.asarray(sim.bot[0].applied_motor_commands[:4], dtype=float)

        sample_contacts = step_index % 10 == 0
        if sample_contacts:
            forces = _contacts(client, rid, link_names)
            # Link names drop the joint suffix: robot_front_right_wheel, and so on.
            wheel_force = np.array([
                forces.get(name[: -len("_joint")], 0.0) for name in WHEEL_NAMES
            ])
            for name, force in forces.items():
                if name.endswith("_wheel"):
                    continue
                other_force_peak[name] = max(other_force_peak.get(name, 0.0), force)
                if 14.0 <= current_time <= 20.0:
                    kink_force_peak[name] = max(kink_force_peak.get(name, 0.0), force)
        else:
            wheel_force = wheel_force_all[-1] if wheel_force_all else np.zeros(4)

        columns["time"].append(current_time)
        columns["x"].append(float(base_pos[0]))
        columns["y"].append(float(base_pos[1]))
        columns["z"].append(float(base_pos[2]))
        columns["roll"].append(float(roll))
        columns["pitch"].append(float(pitch))
        columns["yaw"].append(float(yaw))
        columns["yaw_pin"].append(_pin_yaw(q_pin))
        columns["v_cmd"].append(v_cmd)
        columns["w_cmd"].append(w_cmd)
        columns["v_meas"].append(v_meas)
        columns["w_meas"].append(w_meas)
        columns["v_body"].append(float(lin_body[0]))
        columns["w_body"].append(float(ang_body[2]))
        columns["v_wheels"].append(float(v_wheels))
        columns["w_wheels"].append(float(w_wheels))
        columns["alpha"].append(alpha)
        columns["alpha_geom"].append(float(alpha_geom))
        columns["ee_dist"].append(float(np.linalg.norm(target - ee_pos)))
        columns["reach_w"].append(float(cmd.reach_weight))
        columns["foot_x"].append(float(foot[0]))
        columns["foot_y"].append(float(foot[1]))
        wheel_cmd_all.append(wheel_cmds.copy())
        wheel_qd_all.append(qd_now.copy())
        wheel_qd_pre_all.append(np.asarray(qd_mes[:4], dtype=float).copy())
        wheel_tau_all.append(tau.copy())
        wheel_force_all.append(np.asarray(wheel_force, dtype=float).copy())

        arm_engaged = cmd.reach_weight >= 0.999
        stop_message = check_stop(
            stop_criteria, current_time, time_step, columns["ee_dist"][-1], arm_engaged
        )
        if stop_message:
            break
        current_time += time_step
        step_index += 1
        if step_index % 5000 == 0:
            print(f"  t={current_time:.1f}s  EE {columns['ee_dist'][-1]:.3f} m", flush=True)

    data = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    data["wheel_cmd"] = np.asarray(wheel_cmd_all)
    data["wheel_qd"] = np.asarray(wheel_qd_all)
    data["wheel_qd_pre"] = np.asarray(wheel_qd_pre_all)
    data["wheel_tau"] = np.asarray(wheel_tau_all)
    data["wheel_force"] = np.asarray(wheel_force_all)
    data["stop_message"] = stop_message
    data["limits"] = limits
    data["torque_limits_enforced"] = torque_limits
    data["other_force_peak"] = other_force_peak
    data["kink_force_peak"] = kink_force_peak
    data["radius"] = radius
    data["width"] = width
    data["kd"] = float(sim.bot[0].servo_motor_model._kd)
    return data


def _median_ratio(mask, numerator, denominator, floor):
    selected = mask & (np.abs(denominator) > floor)
    if not np.any(selected):
        return None, 0
    ratio = numerator[selected] / denominator[selected]
    return float(np.median(ratio)), int(np.count_nonzero(selected))


def _fmt_ratio(value, count):
    if value is None:
        return "no samples"
    return f"{value:.2f}  (n={count})"


def summarise(data):
    time = data["time"]
    v_cmd = _smooth(data["v_cmd"], SMOOTH_STEPS)
    w_cmd = _smooth(data["w_cmd"], SMOOTH_STEPS)
    v_meas = _smooth(data["v_meas"], SMOOTH_STEPS)
    w_meas = _smooth(data["w_meas"], SMOOTH_STEPS)
    w_wheels = _smooth(data["w_wheels"], SMOOTH_STEPS)
    spin = (np.abs(v_cmd) < 0.05) & (np.abs(w_cmd) > 0.3)
    arc = (np.abs(v_cmd) > 0.10) & (np.abs(w_cmd) > 0.05)
    lines = []
    lines.append(f"stop: {data['stop_message']}")
    lines.append(f"duration {time[-1]:.2f} s, {len(time)} samples, dt implicit from the config")
    yaw_err = wrap_angle(data["yaw_pin"] - data["yaw"])
    alpha_err = wrap_angle(data["alpha"] - data["alpha_geom"])
    lines.append(
        f"yaw pinocchio minus pybullet: max {np.max(np.abs(yaw_err)):.2e} rad"
    )
    lines.append(
        f"alpha expert minus atan2(target - base) - yaw: max {np.max(np.abs(alpha_err)):.2e} rad"
    )
    foot_err = np.hypot(data["foot_x"] - data["x"], data["foot_y"] - data["y"])
    lines.append(
        f"pinocchio footprint minus pybullet base, horizontal: max {np.max(foot_err):.4f} m"
    )
    lines.append("yaw achieved / yaw commanded, and body yaw / yaw implied by the wheel speeds:")
    for label, mask in (("spin |v|<0.05 and |w|>0.3", spin), ("arc |v|>0.10 and |w|>0.05", arc)):
        body_over_cmd, n_body = _median_ratio(mask, w_meas, w_cmd, 0.05)
        wheels_over_cmd, n_wheels = _median_ratio(mask, w_wheels, w_cmd, 0.05)
        body_over_wheels, n_slip = _median_ratio(mask, w_meas, w_wheels, 0.05)
        if np.any(mask):
            span = f"t {time[mask][0]:.1f}–{time[mask][-1]:.1f} s"
        else:
            span = "absent"
        lines.append(f"  {label} ({span})")
        lines.append(f"    body / command {_fmt_ratio(body_over_cmd, n_body)}")
        lines.append(f"    wheels / command {_fmt_ratio(wheels_over_cmd, n_wheels)}")
        lines.append(f"    body / wheels {_fmt_ratio(body_over_wheels, n_slip)}")
    after = (time > 8.0) & (time < 25.0)
    kink_index = int(np.argmin(np.where(after, data["ee_dist"], np.inf)))
    kink_t = float(time[kink_index])
    window = (time >= kink_t - 2.0) & (time <= kink_t + 2.0)
    lines.append(
        f"EE minimum in 8–25 s is {data['ee_dist'][kink_index]:.3f} m at t={kink_t:.2f} s"
    )
    lines.append(
        "roll / pitch peak |deg| in that ±2 s: "
        f"{np.degrees(np.max(np.abs(data['roll'][window]))):.2f} / "
        f"{np.degrees(np.max(np.abs(data['pitch'][window]))):.2f}"
    )
    lines.append(
        "roll / pitch peak |deg| over the run: "
        f"{np.degrees(np.max(np.abs(data['roll']))):.2f} / "
        f"{np.degrees(np.max(np.abs(data['pitch']))):.2f}"
    )
    lines.append(
        f"base height in the ±2 s window: {data['z'][window].min():.4f}–{data['z'][window].max():.4f} m"
    )
    lost = data["wheel_force"][window] < 1.0
    for index, label in enumerate(WHEEL_LABELS):
        gap = float(np.mean(lost[:, index])) if len(time[window]) else 0.0
        lines.append(
            f"  {label} normal force in the window: "
            f"min {data['wheel_force'][window, index].min():.1f} N, "
            f"median {np.median(data['wheel_force'][window, index]):.1f} N, "
            f"fraction below 1 N: {gap:.2f}"
        )
    if data["kink_force_peak"]:
        touched = ", ".join(
            f"{name} {force:.1f} N" for name, force in sorted(data["kink_force_peak"].items(), key=lambda item: -item[1])
        )
        lines.append(f"non-wheel contacts at 14–20 s: {touched}")
    else:
        lines.append("non-wheel contacts at 14–20 s: none")
    if data["other_force_peak"]:
        touched = ", ".join(
            f"{name} {force:.1f} N" for name, force in sorted(data["other_force_peak"].items(), key=lambda item: -item[1])
        )
        lines.append(f"non-wheel contacts over the run: {touched}")
    lines.append(f"velocity-loop kd = {data['kd']:g}; ServoMotorModel torque limit = {data['torque_limits_enforced']}")
    for row in data["limits"]:
        lines.append(
            f"  {row['joint']}: URDF effort {row['effort']:.0f} N·m, "
            f"velocity {row['velocity']:.0f} rad/s, damping {row['damping']:g}, friction {row['friction']:g}"
        )
    tau = data["wheel_tau"]
    qd = data["wheel_qd"]
    cmd = data["wheel_cmd"]
    lines.append(
        f"peak |commanded wheel speed| {np.max(np.abs(cmd)):.2f} rad/s, "
        f"peak |measured| {np.max(np.abs(qd)):.2f} rad/s, "
        f"peak |torque| {np.max(np.abs(tau)):.1f} N·m"
    )
    # The velocity loop torques the speed it read before the step, not the speed after it.
    expected = data["kd"] * (cmd - data["wheel_qd_pre"])
    lines.append(
        f"torque minus kd*(command - speed seen by the loop): max abs {np.max(np.abs(tau - expected)):.2f} N·m"
    )
    # Outer wheel: the side whose |command| is larger. Differential shortfall is command minus measurement.
    right_cmd = cmd[:, 0]
    left_cmd = cmd[:, 1]
    right_qd = 0.5 * (qd[:, 0] + qd[:, 3])
    left_qd = 0.5 * (qd[:, 1] + qd[:, 2])
    outer_is_right = np.abs(right_cmd) >= np.abs(left_cmd)
    outer_short = np.where(
        outer_is_right,
        np.abs(right_cmd) - np.abs(right_qd),
        np.abs(left_cmd) - np.abs(left_qd),
    )
    lines.append(
        f"outer-wheel |command| - |measured|: median {np.median(outer_short):.3f} rad/s, "
        f"95th percentile {np.percentile(outer_short, 95):.3f} rad/s"
    )
    return kink_t, lines


def _mark(ax, kink_t):
    ax.axvline(kink_t, color="0.4", ls=":", lw=0.9, label=f"EE minimum t={kink_t:.1f}s")


def plot_all(data, kink_t, out_dir):
    time = data["time"]
    v_cmd = _smooth(data["v_cmd"], SMOOTH_STEPS)
    w_cmd = _smooth(data["w_cmd"], SMOOTH_STEPS)
    v_meas = _smooth(data["v_meas"], SMOOTH_STEPS)
    w_meas = _smooth(data["w_meas"], SMOOTH_STEPS)
    v_wheels = _smooth(data["v_wheels"], SMOOTH_STEPS)
    w_wheels = _smooth(data["w_wheels"], SMOOTH_STEPS)
    saved = []

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
    axes[0].plot(time, v_cmd, color="C1", lw=1.3, label="Commanded")
    axes[0].plot(time, v_meas, color="C0", lw=1.15, label="Measured, heading frame")
    axes[0].plot(time, v_wheels, color="C2", lw=1.0, ls="--", label="From measured wheel speeds")
    axes[0].set_ylabel("Forward speed v (m/s)")
    axes[0].grid(True, alpha=0.3)
    _mark(axes[0], kink_t)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(time, w_cmd, color="C1", lw=1.3, label="Commanded")
    axes[1].plot(time, w_meas, color="C0", lw=1.15, label="Measured yaw rate")
    axes[1].plot(time, w_wheels, color="C2", lw=1.0, ls="--", label="From measured wheel speeds")
    axes[1].set_ylabel("Yaw rate ω (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    _mark(axes[1], kink_t)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Markov (−1.2, 1.2, 0.7) — body twist\n"
        "Measured traces are a 50 ms average. Wheel curve uses the calibrated radius and track.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out_dir, "twist.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)

    # The comparison figure's kink is the departure after the closest approach, near 16 s,
    # not the instant of minimum error (about 12 s on this target).
    window = (time >= 12.0) & (time <= 20.0)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
    axes[0].plot(time[window], np.degrees(data["roll"][window]), color="C0", lw=1.15, label="Roll")
    axes[0].plot(time[window], np.degrees(data["pitch"][window]), color="C1", lw=1.15, label="Pitch")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(16.0, color="0.4", ls=":", lw=0.9)
    axes[0].legend(loc="upper right", fontsize=8)
    for index, label in enumerate(WHEEL_LABELS):
        axes[1].plot(time[window], data["wheel_force"][window, index], lw=1.05, label=label)
    axes[1].axhline(1.0, color="k", ls=":", lw=0.8, label="1 N")
    axes[1].set_ylabel("Wheel normal force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(16.0, color="0.4", ls=":", lw=0.9, label="t = 16 s")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(
        "Markov (−1.2, 1.2, 0.7) — base attitude around the 16 s departure\n"
        "Normal force is sampled every 10 ms and held. A wheel near 0 N has left the ground. "
        f"Closest approach was {data['ee_dist'][int(np.argmin(np.abs(time - kink_t)))]:.3f} m at t={kink_t:.1f}s.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out_dir, "attitude.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    sc = ax.scatter(
        data["x"], data["y"], c=time, s=3, cmap="viridis", linewidths=0, zorder=2,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("Time (s)")
    stride = max(1, int(round(1.0 / np.median(np.diff(time)))))
    idx = np.arange(0, len(time), stride)
    heading = np.stack([np.cos(data["yaw"][idx]), np.sin(data["yaw"][idx])], axis=1)
    aim = data["yaw"][idx] + data["alpha"][idx]
    aim_xy = np.stack([np.cos(aim), np.sin(aim)], axis=1)
    ax.quiver(
        data["x"][idx], data["y"][idx], heading[:, 0], heading[:, 1],
        angles="xy", scale_units="xy", scale=5.0, width=0.004, color="C0",
        label="Body heading", zorder=3,
    )
    ax.quiver(
        data["x"][idx], data["y"][idx], aim_xy[:, 0], aim_xy[:, 1],
        angles="xy", scale_units="xy", scale=5.0, width=0.003, color="C1",
        label="Expert α, direction of the target", zorder=3,
    )
    ax.scatter([TARGET[0]], [TARGET[1]], c="red", s=40, zorder=4, label="Target")
    kink_i = int(np.argmin(np.abs(time - kink_t)))
    ax.scatter([data["x"][kink_i]], [data["y"][kink_i]], c="k", s=28, zorder=4, label=f"t={kink_t:.1f}s")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    alpha_err = np.max(np.abs(wrap_angle(data["alpha"] - data["alpha_geom"])))
    ax.set_title(
        "Markov (−1.2, 1.2, 0.7) — path, heading, and expert α\n"
        f"Arrows every 1 s. max |α − (world bearing − yaw)| = {alpha_err:.1e} rad"
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "path_alpha.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)

    effort = data["limits"][0]["effort"]
    velocity_limit = data["limits"][0]["velocity"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    for index, label in enumerate(WHEEL_LABELS):
        axes[0].plot(time, data["wheel_cmd"][:, index], color=f"C{index}", lw=1.15, label=f"{label} command")
        axes[0].plot(time, data["wheel_qd"][:, index], color=f"C{index}", lw=0.9, ls="--")
    axes[0].set_ylabel("Wheel speed (rad/s)")
    axes[0].grid(True, alpha=0.3)
    _mark(axes[0], kink_t)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    for index, label in enumerate(WHEEL_LABELS):
        axes[1].plot(time, data["wheel_tau"][:, index], lw=1.0, label=label)
    axes[1].axhline(effort, color="k", ls=":", lw=0.8)
    axes[1].axhline(-effort, color="k", ls=":", lw=0.8, label=f"URDF effort ±{effort:.0f} N·m")
    axes[1].set_ylabel("Applied torque (N·m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    _mark(axes[1], kink_t)
    axes[1].legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(
        "Markov (−1.2, 1.2, 0.7) — wheel speed and torque\n"
        f"Solid is the command, dashed is measured. URDF velocity limit is ±{velocity_limit:.0f} rad/s "
        "and is not drawn: every command is far below it.\n"
        "PyBullet is in torque mode and the motor model does not apply the URDF effort as a clamp.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = os.path.join(out_dir, "wheel_limits.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)
    return saved


def main():
    print("Opening a headless simulation for the Markov expert at (-1.2, 1.2, 0.7)", flush=True)
    ctx = task.setup_simulation()
    try:
        data = run_log(ctx)
    finally:
        ctx["client"].disconnect()
    kink_t, lines = summarise(data)
    out_dir = task.plots_dir("markov_vs_original", "-1.20_1.20_0.70")
    np.savez(
        os.path.join(out_dir, "base_log.npz"),
        **{key: value for key, value in data.items() if isinstance(value, np.ndarray)},
        effort=np.array([row["effort"] for row in data["limits"]]),
        velocity_limit=np.array([row["velocity"] for row in data["limits"]]),
    )
    saved = plot_all(data, kink_t, out_dir)
    print("\n".join(lines), flush=True)
    for path in saved:
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
