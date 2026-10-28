import numpy as np
import time
import os
import sys

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import mpl_toolkits.mplot3d  # noqa: F401  registers the 3d projection
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)
from robotnik_markov_expert import (
    ArmControllerParams as MarkovArmParams,
    BaseControllerParams as MarkovBaseParams,
    MarkovExpert,
    check_stop as markov_check_stop,
    integrator_update,
    relative_target,
)


# ==============================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the Kinova end-effector in the world frame:
# [X (forward/meters), Y (lateral/meters), Z (height/meters)]
TARGET_EE_POS = np.array([0.0, 1.1, 0.5])

# Sparse world-frame grid the expert is scored on. The column through the robot
# (horizontal radius below GRID_MIN_RADIUS) is omitted — those points start inside the base.
GRID_X = np.array([-1.2, 0.0, 1.2])
GRID_Y = np.array([-1.2, 0.0, 1.2])
GRID_Z = np.array([0.25, 0.70, 1.15])
GRID_MIN_RADIUS = 0.4

# Fixed gripper finger configuration (angles in radians, held fixed during simulation):
# [finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3, finger_tip_3]
# Set to None to use initial configuration from config/URDF, or an array of 6 angles.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Max time (s) allowed for the base to reach its standoff pose before the arm starts reaching.
# Targets behind the robot need a large turn, so this must be generous.
BASE_NAV_TIMEOUT = 25.0
# Max EE distance from the shoulder the arm is asked to reach (m). Targets further away are projected
# onto this sphere, so the arm settles at the closest pose instead of fully stretching (singularity).
ARM_REACH_LIMIT = 0.85
# Base-to-target horizontal standoff (m) as a function of target height: larger for low targets,
# default for mid heights, smaller for high targets (far above the shoulder, which sits ~0.49 m high).
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
# Stop once the EE is this close to the target (m); max iterations set via ROBOENV_MAX_STEPS
REACH_TOLERANCE = 0.01
# ==============================================================================

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(TASK_DIR, "plots")


def plots_dir(*parts):
    """Directory for figures, created if needed. `plots_dir("bc")` is `plots/bc`."""
    path = os.path.join(PLOTS_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def sparse_target_grid():
    """Cartesian targets on GRID_X × GRID_Y × GRID_Z, excluding the robot footprint."""
    targets = [
        np.array([x, y, z], dtype=float)
        for z in GRID_Z
        for y in GRID_Y
        for x in GRID_X
        if np.hypot(x, y) >= GRID_MIN_RADIUS
    ]
    return targets


def simulation_root():
    """Directory containing configs/ and models/.

    The task folder is used when those assets have been copied next to this script.
    Otherwise the RoboEnv checkout two levels up is used.
    """
    if os.path.isdir(os.path.join(TASK_DIR, "configs")) and os.path.isdir(os.path.join(TASK_DIR, "models")):
        return TASK_DIR
    roboenv = os.path.abspath(os.path.join(TASK_DIR, os.pardir, os.pardir, "RoboEnv"))
    if os.path.isdir(os.path.join(roboenv, "configs")) and os.path.isdir(os.path.join(roboenv, "models")):
        return roboenv
    raise FileNotFoundError(
        "Need configs/ and models/ next to this script (copy them from RoboEnv), "
        f"or a RoboEnv checkout at {roboenv}"
    )


def _place_target_marker(ctx, target_ee_pos):
    """Show a single red target dot, replacing whatever was drawn before.

    A massless sphere keeps its previous visual when its pose is reset, which
    leaves a trail as the target changes. Delete that body and create it again,
    and move the existing debug point instead of adding another one.
    """
    client = ctx["client"]
    pos = np.asarray(target_ee_pos, dtype=float).tolist()

    if ctx["target_body"] is not None:
        client.removeBody(ctx["target_body"])
    if ctx["target_visual"] is None:
        ctx["target_visual"] = client.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[1.0, 0.0, 0.0, 0.85]
        )
    ctx["target_body"] = client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=ctx["target_visual"],
        basePosition=pos
    )

    if not headless:
        previous_point = ctx["target_debug_point"]
        ctx["target_debug_point"] = client.addUserDebugPoints(
            pointPositions=[pos],
            pointColorsRGB=[[1.0, 0.0, 0.0]],
            pointSize=10,
            replaceItemUniqueId=-1 if previous_point is None else previous_point,
        )


def setup_simulation():
    """Create the simulator, fix the gripper, and build the kinematic model."""
    conf_file_name = "robotnik_arm.json"
    cur_dir = simulation_root()
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    # Configure realistic ground contact and skid-steer friction dynamics
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0]
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    # Gripper fingers stay at a fixed configuration for the whole process.
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    controlled_frame_name = "robot_j2s7s300_end_effector"
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    init_joint_angles = np.array(sim.GetMotorAngles(0))

    return {
        "sim": sim,
        "client": client,
        "rid": rid,
        "dyn_model": dyn_model,
        "ee_frame_id": dyn_model.pin_model.getFrameId(controlled_frame_name),
        "shoulder_frame_id": dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2"),
        "target_finger_angles": target_finger_angles,
        "q_init_arm": init_joint_angles[4:11].copy(),
        "arm_lower": np.array(lower_limits[4:11]),
        "arm_upper": np.array(upper_limits[4:11]),
        "time_step": sim.GetTimeStep(),
        "arm_params": ArmControllerParams(reach_limit=ARM_REACH_LIMIT),
        "base_params": BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT),
        "target_body": None,
        "target_visual": None,
        "target_debug_point": None,
    }


def _standoff_distance(target_ee_pos):
    """Horizontal distance from the parked base origin to the target.

    This is the distance the phased controller's goal pose sits from the target.
    The Markov base law takes that distance directly and does not use a goal pose.
    """
    return float(np.interp(target_ee_pos[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))


def _base_goal(target_ee_pos):
    """Standoff pose the phased base controller drives to before the arm reaches."""
    standoff = _standoff_distance(target_ee_pos)
    target_xy = target_ee_pos[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    return base_goal_pos, target_bearing


def _markov_expert(ctx, target_ee_pos):
    """Markov expert for this target. Only the standoff depends on the target.

    The cartesian integrator z is not kept here. The episode loop updates it with
    integrator_update and passes it in on each call.
    """
    return MarkovExpert(
        ctx["dyn_model"],
        ctx["ee_frame_id"],
        ctx["shoulder_frame_id"],
        ctx["arm_lower"],
        ctx["arm_upper"],
        ctx["q_init_arm"],
        MarkovArmParams(reach_limit=ARM_REACH_LIMIT),
        MarkovBaseParams(standoff=_standoff_distance(target_ee_pos)),
    )


def run_episode(ctx, target_ee_pos, *, record=True, verbose=True, realtime=False, on_step=None,
                controller="phased"):
    """Reset the robot and run one expert until a stop condition.

    `controller` is `"phased"` (the original base phases plus joint-space integrator) or
    `"markov"` (a call on the heading-frame observation and the cartesian integrator z).
    Both start from the same reset pose. `target_ee_pos` is a world-frame Cartesian point (m).

    Returns final error, the stop message, and (optionally) the time series used by the
    single-target plots.

    `on_step(current_time, ee_dist_error)` is called about 20 times a second after the
    physics step. Return False from it to stop the episode early.
    """
    if controller not in ("phased", "markov"):
        raise ValueError(f"Unknown controller {controller!r}; expected 'phased' or 'markov'")
    sim = ctx["sim"]
    client = ctx["client"]
    dyn_model = ctx["dyn_model"]
    target_ee_pos = np.asarray(target_ee_pos, dtype=float)
    target_finger_angles = ctx["target_finger_angles"]
    time_step = ctx["time_step"]

    sim.ResetPose()
    client.resetBaseVelocity(ctx["rid"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sim.ReceiveObservation()
    _place_target_marker(ctx, target_ee_pos)

    q_init_arm = ctx["q_init_arm"]
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)
    if controller == "markov":
        markov_expert = _markov_expert(ctx, target_ee_pos)
        # z integrates the end-effector error outside the expert. It starts at rest.
        z = np.zeros(3)
        base_state = None
        base_goal_pos = base_goal_bearing = None
    else:
        markov_expert = None
        base_goal_pos, base_goal_bearing = _base_goal(target_ee_pos)
        base_state = BaseNavState()
    cmd = MotorCommands()

    time_all, ee_pos_all, ee_error_all = [], [], []
    q_mes_all, q_d_all = [], []
    torque_arm_all, wheel_cmds_all = [], []
    # Effective base phase: once navigation finishes, `done` stays true while `phase` does not change.
    phase_all, q_des_arm_all = [], []

    # One kinematic pass so the banner matches the state the loop starts from.
    base_pos = np.array(sim.GetBasePosition())
    base_ori = np.array(sim.GetBaseOrientation())
    q_mes = np.array(sim.GetMotorAngles(0))
    q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ctx["ee_frame_id"]].translation.copy()
    if verbose:
        print("==================================================================")
        print(f"Controller                   : {controller}")
        print(f"Target Cartesian EE Position : {target_ee_pos}")
        print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
        print(f"Initial Cartesian Error (m)  : {np.linalg.norm(target_ee_pos - init_ee_pos):.4f}")
        print("==================================================================")

    current_time = 0.0
    ee_pos = init_ee_pos.copy()
    stop_message = None
    user_quit = False
    step_index = 0
    base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

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
        ee_error = target_ee_pos - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)

        if controller == "markov":
            # World pose enters only through the heading-frame displacement.
            # The command uses the current z; the update below is z for the next step.
            target_rel = relative_target(target_ee_pos, base_pos, base_bearing)
            cmd_exp = markov_expert(q_pin, q_mes[4:11], qd_mes[4:11], target_rel, z)
            wheel_cmds = cmd_exp.wheel_cmd
            torque_arm = cmd_exp.torque_arm
            q_des_arm = cmd_exp.q_ref_arm
            arm_engaged = cmd_exp.reach_weight >= 0.999
            stop_ee = cmd_exp.ee_error
            z = integrator_update(
                z, cmd_exp.cartesian_error, cmd_exp.reach_weight, time_step, markov_expert.arm_params.z_max,
            )
        else:
            wheel_cmds = base_controller(
                base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, ctx["base_params"]
            )
            arm_engaged = base_state.done
            torque_arm, q_des_arm, qd_des_arm = arm_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target_ee_pos, ee_pos,
                ctx["ee_frame_id"], ctx["shoulder_frame_id"], ctx["arm_lower"], ctx["arm_upper"], time_step,
                ctx["arm_params"], reach_active=arm_engaged,
            )
            stop_ee = ee_dist_error
        torque_fingers = np.zeros(6)

        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            user_quit = True
            stop_message = "Stopped by user"
            break

        if controller == "markov":
            stop_message = markov_check_stop(
                stop_criteria, current_time, time_step, stop_ee, arm_engaged
            )
        else:
            stop_message = check_stop(stop_criteria, current_time, time_step, stop_ee, arm_engaged)
        if stop_message:
            if verbose:
                print(stop_message)
            break

        if record:
            time_all.append(current_time)
            ee_pos_all.append(ee_pos)
            ee_error_all.append(ee_error)
            current_desired_q = np.concatenate((q_mes[:4], q_des_arm, target_finger_angles))
            q_mes_all.append(q_mes)
            q_d_all.append(current_desired_q)
            torque_arm_all.append(np.asarray(torque_arm, dtype=float).copy())
            wheel_cmds_all.append(np.asarray(wheel_cmds, dtype=float).copy())
            if controller == "markov":
                phase_all.append("reach" if arm_engaged else "hold")
            else:
                phase_all.append("done" if base_state.done else base_state.phase)
            q_des_arm_all.append(np.asarray(q_des_arm, dtype=float).copy())

        if on_step is not None and step_index % 50 == 0:
            if on_step(current_time, float(ee_dist_error)) is False:
                user_quit = True
                stop_message = "Stopped from the playback window"
                if verbose:
                    print(stop_message)
                break

        if realtime and not headless:
            time.sleep(time_step)

        current_time += time_step
        step_index += 1
        if verbose and int(current_time / time_step) % 500 == 0:
            if controller == "markov":
                status = (
                    f"reach_w {cmd_exp.reach_weight:.2f} engaged {arm_engaged} "
                    f"|z| {np.linalg.norm(z):.4f}"
                )
            else:
                status = f"phase {base_state.phase} nav_done {arm_engaged}"
            print(
                f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | "
                f"EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | "
                f"yaw {base_bearing:.2f} {status}"
            )

    final_error = float(np.linalg.norm(target_ee_pos - ee_pos))
    if verbose:
        print("==================================================================")
        print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
        print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
        print(f"Target End-Effector Position : {target_ee_pos}")
        print(f"Final Cartesian Error (m)    : {final_error:.4f}")
        print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
        print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
        print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
        print("==================================================================")

    return {
        "controller": controller,
        "target": target_ee_pos,
        "final_error": final_error,
        "final_ee_pos": np.asarray(ee_pos, dtype=float).copy(),
        "stop_message": stop_message,
        "time": current_time,
        "user_quit": user_quit,
        "time_all": np.asarray(time_all) if record else None,
        "ee_pos_all": np.asarray(ee_pos_all) if record else None,
        "ee_error_all": np.asarray(ee_error_all) if record else None,
        "q_mes_all": q_mes_all if record else None,
        "q_d_all": q_d_all if record else None,
        "torque_arm_all": np.asarray(torque_arm_all) if record else None,
        "wheel_cmds_all": np.asarray(wheel_cmds_all) if record else None,
        "phase_all": np.asarray(phase_all) if record else None,
        "q_des_arm_all": np.asarray(q_des_arm_all) if record else None,
    }


def plot_episode(result):
    """Time-series plots for a single expert episode."""
    target = result["target"]
    time_all = result["time_all"]
    ee_pos_all = result["ee_pos_all"]
    ee_error_all = result["ee_error_all"]
    if time_all is None or len(time_all) == 0:
        print("No samples logged; skipping episode plots.")
        return

    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=target[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in result["q_mes_all"]], label=f'Measured J{i+1}')
        plt.plot(time_all, [q[4 + i] for q in result["q_d_all"]], label=f'Desired J{i+1}', linestyle='--')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()


WHEEL_NAMES = ["front right", "front left", "back left", "back right"]
PHASE_ORDER = ["align", "drive", "orient", "done"]


def _target_tag(target):
    target = np.asarray(target, dtype=float)
    return f"target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) m"


def plot_command_signals(result, out_dir=None):
    """Arm torques, wheel commands, and the expert's carried internal state.

    The base phase and the arm joint reference `q_des` are memory the expert
    keeps between steps. They are not in the task's measurement list.
    """
    time_all = result["time_all"]
    if time_all is None or len(time_all) == 0:
        print("No samples logged; skipping command plots.")
        return []

    out_dir = plots_dir("expert") if out_dir is None else out_dir
    tag = _target_tag(result["target"])
    joint_colors = plt.cm.tab10(np.linspace(0.0, 0.7, 7))
    saved = []

    fig, ax = plt.subplots(figsize=(10, 4.6))
    torque = result["torque_arm_all"]
    for i in range(torque.shape[1]):
        ax.plot(time_all, torque[:, i], color=joint_colors[i], label=f"J{i + 1}", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title(f"Arm joint torques — {tag}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=7, loc="upper right", fontsize=8)
    fig.tight_layout()
    torque_path = os.path.join(out_dir, "expert_arm_torques.png")
    fig.savefig(torque_path, dpi=140, bbox_inches="tight")
    saved.append(torque_path)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    wheels = result["wheel_cmds_all"]
    for i, name in enumerate(WHEEL_NAMES):
        ax.plot(time_all, wheels[:, i], label=name, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wheel command (rad/s)")
    ax.set_title(f"Wheel velocity commands — {tag}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    wheel_path = os.path.join(out_dir, "expert_wheel_cmds.png")
    fig.savefig(wheel_path, dpi=140, bbox_inches="tight")
    saved.append(wheel_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)
    phase_index = np.array([PHASE_ORDER.index(phase) for phase in result["phase_all"]])
    axes[0].step(time_all, phase_index, where="post", color="C0", linewidth=1.6)
    axes[0].set_yticks(range(len(PHASE_ORDER)), PHASE_ORDER)
    axes[0].set_ylim(-0.3, len(PHASE_ORDER) - 0.7)
    axes[0].set_ylabel("Base phase")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Expert internal state — {tag}")

    q_des = result["q_des_arm_all"]
    for i in range(q_des.shape[1]):
        axes[1].plot(time_all, q_des[:, i], color=joint_colors[i], label=f"J{i + 1}", linewidth=1.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Arm reference (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=7, loc="upper right", fontsize=8)
    fig.tight_layout()
    state_path = os.path.join(out_dir, "expert_internal_state.png")
    fig.savefig(state_path, dpi=140, bbox_inches="tight")
    saved.append(state_path)

    for path in saved:
        print(f"Saved {path}")
    if headless:
        plt.close("all")
    else:
        plt.show()
    return saved


def run_random_trajectory():
    """Run the expert once on a random sparse-grid target and save the three signal plots."""
    targets = sparse_target_grid()
    index = int(np.random.default_rng().integers(len(targets)))
    target = targets[index]
    print(f"Random trajectory {index + 1}/{len(targets)}: {np.round(target, 3)}", flush=True)
    ctx = setup_simulation()
    try:
        result = run_episode(ctx, target, record=True, verbose=True, realtime=False)
    finally:
        ctx["client"].disconnect()
    plot_command_signals(result)
    return result


def plot_targets_by_error(targets, errors, title=None, vmax=None):
    """Scatter every grid target, coloured on a linear scale by final EE error.

    Left: the 3D grid. Right: one horizontal slice per height, with the error in metres
    next to each point. The colour scale starts at 0 so equal error differences are equal
    colour differences.
    """
    targets = np.asarray(targets, dtype=float)
    errors = np.asarray(errors, dtype=float)
    hi = float(errors.max()) if vmax is None else float(vmax)
    norm = Normalize(vmin=0.0, vmax=max(hi, 1e-6))
    cmap = plt.cm.viridis
    heights = np.unique(np.round(targets[:, 2], 5))

    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(len(heights), 2, width_ratios=[1.15, 1.0], wspace=0.32, hspace=0.55)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax3.scatter(
        targets[:, 0], targets[:, 1], targets[:, 2],
        c=errors, cmap=cmap, norm=norm, s=70, depthshade=False,
        edgecolors="k", linewidths=0.4,
    )
    ax3.scatter([0.0], [0.0], [0.0], c="k", marker="+", s=60, depthshade=False)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.set_title("All targets")
    ax3.view_init(elev=22, azim=-58)
    spans = np.ptp(targets, axis=0)
    spans[spans < 1e-6] = 1.0
    ax3.set_box_aspect(spans)

    pad = 0.35
    xlim = (targets[:, 0].min() - pad, targets[:, 0].max() + pad)
    ylim = (targets[:, 1].min() - pad, targets[:, 1].max() + pad)
    slice_axes = []
    for row, z in enumerate(heights):
        ax = fig.add_subplot(gs[row, 1])
        mask = np.isclose(targets[:, 2], z)
        ax.scatter(
            targets[mask, 0], targets[mask, 1],
            c=errors[mask], cmap=cmap, norm=norm, s=90,
            edgecolors="k", linewidths=0.4, zorder=3,
        )
        for xyz, err in zip(targets[mask], errors[mask]):
            ax.annotate(
                f"{err:.3f}",
                (xyz[0], xyz[1]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
        ax.scatter([0.0], [0.0], c="k", marker="+", s=50, zorder=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"z = {z:.2f} m")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        slice_axes.append(ax)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(errors)
    cbar = fig.colorbar(sm, ax=slice_axes, fraction=0.046, pad=0.04)
    cbar.set_label("Final end-effector error (m)")
    fig.suptitle(
        title
        or (
            "Expert on a sparse target grid — colour is final EE error (linear scale)\n"
            "Black + is the robot start. Slice labels are the error in metres."
        ),
        fontsize=11,
    )
    return fig


def _report_and_plot(targets, errors):
    within = int(np.sum(errors <= REACH_TOLERANCE))
    print("==================================================================")
    print(f"Targets within {REACH_TOLERANCE * 1000:.0f} mm : {within}/{len(errors)}")
    print(f"Error min / median / max (m)  : {errors.min():.4f} / {np.median(errors):.4f} / {errors.max():.4f}")
    order = np.argsort(errors)
    for idx in order:
        xyz = targets[idx]
        print(f"  ({xyz[0]:6.2f}, {xyz[1]:6.2f}, {xyz[2]:5.2f})  {errors[idx]:.4f} m")
    print("==================================================================")

    fig = plot_targets_by_error(targets, errors)
    out_path = os.path.join(plots_dir("expert"), "expert_grid_error.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved {out_path}")
    if headless:
        plt.close(fig)
    else:
        plt.show()


def plot_saved_grid():
    """Combine expert_grid_part_*.npz shards and draw the error plot."""
    part_paths = sorted(
        os.path.join(TASK_DIR, name)
        for name in os.listdir(TASK_DIR)
        if name.startswith("expert_grid_part_") and name.endswith(".npz")
    )
    if not part_paths:
        raise FileNotFoundError(f"No expert_grid_part_*.npz files in {TASK_DIR}")
    chunks = [np.load(path, allow_pickle=True) for path in part_paths]
    targets = np.vstack([chunk["targets"] for chunk in chunks])
    errors = np.concatenate([chunk["errors"] for chunk in chunks])
    _report_and_plot(targets, errors)


def evaluate_sparse_grid():
    """Run the expert once per sparse-grid target and plot the final errors.

    Set ROBOENV_GRID_SHARD=i/n to run only every n-th target, starting at i, and write
    expert_grid_part_i.npz instead of plotting. Combine the parts with --grid-plot.
    """
    targets = sparse_target_grid()
    shard_spec = os.environ.get("ROBOENV_GRID_SHARD", "").strip()
    shard_index = None
    if shard_spec:
        shard_index, shard_count = (int(part) for part in shard_spec.split("/"))
        targets = targets[shard_index::shard_count]
        print(f"Shard {shard_index}/{shard_count}: {len(targets)} targets", flush=True)
    else:
        print(f"Sparse grid: {len(targets)} targets "
              f"(x={GRID_X.tolist()}, y={GRID_Y.tolist()}, z={GRID_Z.tolist()})", flush=True)
    ctx = setup_simulation()
    results = []
    try:
        for i, target in enumerate(targets):
            print(f"[{i + 1}/{len(targets)}] target {np.round(target, 3)}", flush=True)
            wall_start = time.perf_counter()
            result = run_episode(ctx, target, record=False, verbose=False, realtime=False)
            results.append(result)
            print(
                f"    error {result['final_error']:.4f} m at t={result['time']:.2f}s "
                f"(wall {time.perf_counter() - wall_start:.1f}s) | {result['stop_message']}",
                flush=True,
            )
            if result["user_quit"]:
                break
    finally:
        ctx["client"].disconnect()

    if not results:
        return

    errors = np.array([r["final_error"] for r in results])
    scored_targets = np.vstack([r["target"] for r in results])
    messages = np.array([r["stop_message"] or "" for r in results])
    if shard_index is not None:
        part_path = os.path.join(TASK_DIR, f"expert_grid_part_{shard_index}.npz")
        np.savez(
            part_path,
            targets=scored_targets,
            errors=errors,
            messages=messages,
        )
        print(f"Saved {part_path}")
        return
    grid_path = os.path.join(TASK_DIR, "expert_grid.npz")
    np.savez(grid_path, targets=scored_targets, errors=errors, messages=messages)
    print(f"Saved {grid_path}")
    _report_and_plot(scored_targets, errors)


# Ten sparse-grid targets: front, side, behind and a diagonal at each height band.
COMPARE_TARGETS = [
    np.array([1.2, 0.0, 0.25]),
    np.array([0.0, 1.2, 0.25]),
    np.array([-1.2, 0.0, 0.25]),
    np.array([1.2, 1.2, 0.25]),
    np.array([1.2, 0.0, 0.70]),
    np.array([0.0, -1.2, 0.70]),
    np.array([-1.2, 0.0, 0.70]),
    np.array([-1.2, 1.2, 0.70]),
    np.array([1.2, 0.0, 1.15]),
    np.array([-1.2, -1.2, 1.15]),
]
# The 1 ms config used a 40000-step cap, which is 40 s of simulation.
COMPARE_SIM_SECONDS = 40.0


def _target_panel_title(result_a, result_b):
    target = np.asarray(result_a["target"], dtype=float)
    return (
        f"({target[0]:.1f}, {target[1]:.1f}, {target[2]:.2f})  "
        f"orig {result_a['final_error']:.3f} m / markov {result_b['final_error']:.3f} m"
    )


def plot_controller_comparison(phased_results, markov_results):
    """Overlay original and Markov wheel commands, arm torques and EE error.

    One panel per target. Wheel commands show the two distinct sides (the back
    wheel on each side copies the front). Torque panels use one colour per joint;
    solid is the original, dashed is Markov.
    """
    n = len(phased_results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    saved = []

    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 2.35 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax, phased, markov in zip(flat, phased_results, markov_results):
        if len(phased["time_all"]):
            ax.plot(
                phased["time_all"], np.linalg.norm(phased["ee_error_all"], axis=1),
                color="C0", lw=1.3, label="Original",
            )
        if len(markov["time_all"]):
            ax.plot(
                markov["time_all"], np.linalg.norm(markov["ee_error_all"], axis=1),
                color="C1", lw=1.2, label="Markov",
            )
        ax.axhline(REACH_TOLERANCE, color="k", ls=":", lw=0.8)
        ax.set_title(_target_panel_title(phased, markov), fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EE error (m)")
        ax.grid(True, alpha=0.3)
    for ax in flat[n:]:
        ax.axis("off")
    flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("End-effector error — original vs Markov", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    compare_dir = plots_dir("markov_vs_original")
    ee_path = os.path.join(compare_dir, "ee.png")
    fig.savefig(ee_path, dpi=120, bbox_inches="tight")
    saved.append(ee_path)

    fig, axes = plt.subplots(n, 2, figsize=(10.5, 1.55 * n), sharex=True, squeeze=False)
    for row, (phased, markov) in enumerate(zip(phased_results, markov_results)):
        for col, index, name in ((0, 0, "Right wheels"), (1, 1, "Left wheels")):
            ax = axes[row, col]
            if len(phased["time_all"]):
                ax.plot(phased["time_all"], phased["wheel_cmds_all"][:, index], color="C0", lw=1.15, label="Original")
            if len(markov["time_all"]):
                ax.plot(markov["time_all"], markov["wheel_cmds_all"][:, index], color="C1", lw=1.05, label="Markov")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(_target_tag(phased["target"]), fontsize=7)
            if row == n - 1:
                ax.set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Wheel commands (rad/s) — original vs Markov\nBack wheel on each side copies the front",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    wheel_path = os.path.join(compare_dir, "wheels.png")
    fig.savefig(wheel_path, dpi=120, bbox_inches="tight")
    saved.append(wheel_path)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 2.6 * nrows), squeeze=False)
    flat = axes.ravel()
    joint_colors = plt.cm.tab10(np.linspace(0.0, 0.7, 7))
    for ax, phased, markov in zip(flat, phased_results, markov_results):
        for i in range(7):
            if len(phased["time_all"]):
                ax.plot(
                    phased["time_all"], phased["torque_arm_all"][:, i],
                    color=joint_colors[i], lw=1.0, label=f"J{i + 1}",
                )
            if len(markov["time_all"]):
                ax.plot(
                    markov["time_all"], markov["torque_arm_all"][:, i],
                    color=joint_colors[i], lw=1.0, ls="--",
                )
        ax.set_title(_target_panel_title(phased, markov), fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque (N·m)")
        ax.grid(True, alpha=0.3)
    for ax in flat[n:]:
        ax.axis("off")
    flat[0].legend(ncol=7, loc="upper right", fontsize=7)
    fig.suptitle("Arm torques — solid original, dashed Markov, colour is the joint", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    torque_path = os.path.join(compare_dir, "torques.png")
    fig.savefig(torque_path, dpi=120, bbox_inches="tight")
    saved.append(torque_path)

    for path in saved:
        print(f"Saved {path}")
    if headless:
        plt.close("all")
    else:
        plt.show()
    return saved


def compare_controllers():
    """Run the phased expert and the Markov expert on the same ten targets."""
    global max_steps
    targets = [np.asarray(target, dtype=float) for target in COMPARE_TARGETS]
    print(f"Comparing controllers on {len(targets)} targets", flush=True)
    ctx = setup_simulation()
    dt = float(ctx["time_step"])
    max_steps = int(round(COMPARE_SIM_SECONDS / dt))
    print(f"Physics dt {dt:.4f} s, step cap {max_steps} ({COMPARE_SIM_SECONDS:.0f} s)", flush=True)
    phased_results = []
    markov_results = []
    try:
        for i, target in enumerate(targets):
            print(f"[{i + 1}/{len(targets)}] target {np.round(target, 3)}", flush=True)
            for name, bucket in (("phased", phased_results), ("markov", markov_results)):
                wall_start = time.perf_counter()
                result = run_episode(ctx, target, record=True, verbose=False, realtime=False, controller=name)
                bucket.append(result)
                print(
                    f"    {name:6}  error {result['final_error']:.4f} m at t={result['time']:.2f}s "
                    f"(wall {time.perf_counter() - wall_start:.1f}s) | {result['stop_message']}",
                    flush=True,
                )
                if result["user_quit"]:
                    break
            if phased_results and phased_results[-1]["user_quit"]:
                break
    finally:
        ctx["client"].disconnect()

    if len(phased_results) != len(markov_results) or not phased_results:
        print("Comparison stopped before a full pair was recorded.")
        return

    scored = np.vstack([result["target"] for result in phased_results])
    out_path = os.path.join(plots_dir("markov_vs_original"), "markov_vs_original.npz")
    np.savez(
        out_path,
        targets=scored,
        dt=np.float64(dt),
        phased_errors=np.array([result["final_error"] for result in phased_results]),
        markov_errors=np.array([result["final_error"] for result in markov_results]),
        phased_times=np.array([result["time"] for result in phased_results]),
        markov_times=np.array([result["time"] for result in markov_results]),
        phased_messages=np.array([result["stop_message"] or "" for result in phased_results]),
        markov_messages=np.array([result["stop_message"] or "" for result in markov_results]),
        phased_t=np.array([result["time_all"] for result in phased_results], dtype=object),
        markov_t=np.array([result["time_all"] for result in markov_results], dtype=object),
        phased_ee=np.array([np.linalg.norm(result["ee_error_all"], axis=1) for result in phased_results], dtype=object),
        markov_ee=np.array([np.linalg.norm(result["ee_error_all"], axis=1) for result in markov_results], dtype=object),
        phased_wheels=np.array([result["wheel_cmds_all"] for result in phased_results], dtype=object),
        markov_wheels=np.array([result["wheel_cmds_all"] for result in markov_results], dtype=object),
        phased_torques=np.array([result["torque_arm_all"] for result in phased_results], dtype=object),
        markov_torques=np.array([result["torque_arm_all"] for result in markov_results], dtype=object),
    )
    print(f"Saved {out_path}")
    plot_controller_comparison(phased_results, markov_results)


def main():
    ctx = setup_simulation()
    try:
        result = run_episode(ctx, TARGET_EE_POS, record=True, verbose=True, realtime=not headless)
    finally:
        if headless:
            ctx["client"].disconnect()
    if headless:
        print(f"Headless run finished at t={result['time']:.2f}s")
        return
    plot_episode(result)


if __name__ == '__main__':
    if "--compare" in sys.argv:
        compare_controllers()
    elif "--grid-plot" in sys.argv:
        plot_saved_grid()
    elif os.environ.get("ROBOENV_GRID", "").strip().lower() in ("1", "true", "yes") or "--grid" in sys.argv:
        evaluate_sparse_grid()
    elif "--signals" in sys.argv:
        run_random_trajectory()
    else:
        main()
