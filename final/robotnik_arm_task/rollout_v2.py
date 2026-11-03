"""One simulation episode for collection, DAgger, and the frozen test.

The plant command can come from the expert, from a policy, or from one half
of each. That is the 2×2 swap. DAgger mixing is by half-second blocks rather
than by single steps, so the torque does not chatter at 1 kHz.

Scoring always uses the true end-effector position. Observation noise, when
requested, is added only to the measurements the controller sees. A joint that
winds past one revolution, or a base that leaves the workspace, stops the
episode and is reported as unstable instead of as a large average error.
"""
import os

os.environ.setdefault("ROBOENV_HEADLESS", "1")

import numpy as np
import pinocchio as pin
from simulation_and_control import MotorCommands

from collect_markov_data import apply_dart, clean_wheel_pair
from features_v2 import SPIN_Q, apply_initial_state
from policy_v2 import command_from_policy
from robotnik_arm_control import (
    BaseNavState,
    StopCriteria as PhasedStop,
    arm_controller,
    base_controller,
    check_stop as phased_check_stop,
)
from robotnik_markov_expert import (
    StopCriteria as MarkovStop,
    check_stop as markov_check_stop,
    integrator_update,
    observation_vector,
    relative_target,
)
import robotnik_arm_task as task

TOLERANCES_M = (0.010, 0.025, 0.050, 0.100)
TAIL_S = 0.5


class _Tails:
    def __init__(self):
        self.time_to = {tol: None for tol in TOLERANCES_M}
        self.min_error = float("inf")
        self.t = []
        self.xy = []
        self.qd = []
        self.wheel = []

    def update(self, step, time, distance, base_xy, qd, wheels):
        if distance < self.min_error:
            self.min_error = float(distance)
        for tol in self.time_to:
            if self.time_to[tol] is None and distance <= tol:
                self.time_to[tol] = float(time)
        if step % 20 == 0:
            self.t.append(float(time))
            self.xy.append(np.asarray(base_xy, dtype=np.float64).reshape(2).copy())
            self.qd.append(np.asarray(qd, dtype=np.float64).reshape(7).copy())
            self.wheel.append(np.asarray(wheels, dtype=np.float64).reshape(4).copy())

    def finish(self, time):
        times = np.asarray(self.t, dtype=float)
        speed = None
        qd_abs = None
        wheel_abs = None
        if len(times) >= 2:
            selected = times >= float(time) - TAIL_S
            if int(np.count_nonzero(selected)) < 2:
                selected = np.ones(len(times), dtype=bool)
            xy = np.vstack(self.xy)[selected]
            step_t = times[selected]
            delta = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            speed = float(np.median(delta / np.maximum(np.diff(step_t), 1e-6)))
            qd_abs = float(np.mean(np.abs(np.vstack(self.qd)[selected])))
            wheel_abs = float(np.mean(np.abs(np.vstack(self.wheel)[selected])))
        return {
            "min_error": None if not np.isfinite(self.min_error) else float(self.min_error),
            "time_to_m": {str(tol): self.time_to[tol] for tol in TOLERANCES_M},
            "base_speed_tail": speed,
            "qd_abs_tail": qd_abs,
            "wheel_abs_tail": wheel_abs,
        }


def _unstable(ee_pos, base_pos, q_arm, qd_arm, distance, time, spin_qd):
    spun = bool(
        max(abs(float(qd_arm[0])), abs(float(qd_arm[2]))) > spin_qd
        or max(abs(float(q_arm[0])), abs(float(q_arm[2]))) > SPIN_Q
    )
    finite = bool(np.isfinite(ee_pos).all() and np.isfinite(base_pos).all())
    escaped = bool(
        (not finite)
        or ee_pos[2] < -0.3
        or ee_pos[2] > 2.5
        or np.linalg.norm(ee_pos[:2]) > 4.0
        or np.linalg.norm(base_pos[:2]) > 4.0
        or (distance > 3.0 and time > 2.0)
    )
    return escaped, spun


def _payload(ctx, extra_kg):
    client = ctx["client"]
    rid = ctx["rid"]
    chosen = None
    for joint in range(client.getNumJoints(rid)):
        name = client.getJointInfo(rid, joint)[12].decode()
        if "end_effector" in name:
            chosen = (joint, name)
            break
        if name.endswith("link_7"):
            chosen = (joint, name)
    if chosen is None:
        raise RuntimeError("No end-effector link for the payload change")
    joint, name = chosen
    mass = float(client.getDynamicsInfo(rid, joint)[0])
    client.changeDynamics(rid, joint, mass=mass + float(extra_kg))
    return joint, name, mass


def _measure(ctx):
    sim = ctx["sim"]
    dyn = ctx["dyn_model"]
    base_pos = np.array(sim.GetBasePosition(), dtype=float)
    base_ori = np.array(sim.GetBaseOrientation(), dtype=float)
    q_mes = np.array(sim.GetMotorAngles(0), dtype=float)
    qd_mes = np.array(sim.GetMotorVelocities(0), dtype=float)
    q_pin = dyn.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
    pin.forwardKinematics(dyn.pin_model, dyn.pin_data, q_pin)
    pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
    ee_pos = dyn.pin_data.oMf[ctx["ee_frame_id"]].translation.copy()
    bearing = float(task.quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2]))
    return base_pos, base_ori, bearing, q_mes, qd_mes, q_pin, ee_pos


def _learned_command(policy, expert, cmd, q_pin, q_arm, qd_arm, target_rel, z, base, arm):
    """Wheels, torque, cartesian error, and reach weight for the learned half."""
    if policy.get("kind") in ("split", "mono"):
        feedback_tuple = None if cmd is None else (cmd.cartesian_error, cmd.reach_weight, cmd.tau_gravity)
        return command_from_policy(
            policy, q_pin, q_arm, qd_arm, target_rel, z, expert,
            feedback_tuple=feedback_tuple,
            use_base=base == "learned",
            use_arm=arm == "learned",
        )
    from markov_mlp import predict_command

    obs = observation_vector(target_rel, q_pin, q_arm, qd_arm, z)
    wheels, torque = predict_command(policy, obs)
    if cmd is not None:
        error = np.asarray(cmd.cartesian_error, dtype=np.float64)
        reach_w = float(cmd.reach_weight)
    else:
        error, reach_w = expert.cartesian_feedback(q_pin, target_rel)
        error = np.asarray(error, dtype=np.float64)
        reach_w = float(reach_w)
    if base != "learned":
        wheels = np.zeros(4, dtype=np.float64)
    if arm != "learned":
        torque = np.zeros(7, dtype=np.float64)
    return wheels, torque, error, reach_w


def _control_state(ctx, base_pos, base_ori, bearing, q_mes, qd_mes, q_pin, target, obs_noise):
    """Measurements the controller sees. Noise stays out of the scored pose."""
    q_arm = np.asarray(q_mes[4:11], dtype=float).copy()
    qd_arm = np.asarray(qd_mes[4:11], dtype=float).copy()
    base_ctrl = np.array(base_pos, dtype=float, copy=True)
    bearing_ctrl = float(bearing)
    q_pin_ctrl = q_pin
    if obs_noise is not None:
        rng = obs_noise["rng"]
        q_arm = q_arm + rng.normal(0.0, obs_noise["q"], size=7)
        qd_arm = qd_arm + rng.normal(0.0, obs_noise["qd"], size=7)
        base_ctrl[:2] = base_ctrl[:2] + rng.normal(0.0, obs_noise["xy"], size=2)
        bearing_ctrl = bearing_ctrl + float(rng.normal(0.0, obs_noise["yaw"]))
        q_mes_ctrl = np.array(q_mes, dtype=float, copy=True)
        q_mes_ctrl[4:11] = q_arm
        q_pin_ctrl = ctx["dyn_model"].ReoderJoints2PinVec(
            np.concatenate((base_ctrl, base_ori, q_mes_ctrl)), "pos",
        )
    target_rel = relative_target(target, base_ctrl, bearing_ctrl)
    return q_arm, qd_arm, q_pin_ctrl, target_rel


def _empty_rows():
    return {
        "time": [],
        "observation": [],
        "wheel_cmd": [],
        "torque_arm": [],
        "wheel_applied": [],
        "cartesian_error": [],
        "reach_weight": [],
        "tau_gravity": [],
        "ee_dist": [],
    }


def _stack_rows(rows):
    if not rows["time"]:
        return None
    return {
        "time": np.asarray(rows["time"], dtype=np.float64),
        "observation": np.vstack(rows["observation"]),
        "wheel_cmd": np.vstack(rows["wheel_cmd"]),
        "torque_arm": np.vstack(rows["torque_arm"]),
        "wheel_applied": np.vstack(rows["wheel_applied"]),
        "cartesian_error": np.vstack(rows["cartesian_error"]),
        "reach_weight": np.asarray(rows["reach_weight"], dtype=np.float64),
        "tau_gravity": np.vstack(rows["tau_gravity"]),
        "ee_dist": np.asarray(rows["ee_dist"], dtype=np.float64),
    }


def run_episode(ctx, target, *, policy=None, base="expert", arm="expert", beta=0.0, mix_rng=None,
                initial=None, dart=None, obs_noise=None, push=None, payload_kg=0.0,
                spin_qd=8.0, log_every=0, verbose=False):
    """Run one target. ``base`` and ``arm`` are ``expert`` or ``learned``.

    ``beta`` is the probability that a half-second block executes the expert
    instead of the learned command. It is how early DAgger iterations stay near
    the expert. ``log_every`` is in simulation steps; 0 keeps no rows.
    """
    if base not in ("expert", "learned") or arm not in ("expert", "learned"):
        raise ValueError("base and arm must be 'expert' or 'learned'")
    if (base == "learned" or arm == "learned") and policy is None:
        raise ValueError("A learned command needs a policy")
    target = np.asarray(target, dtype=float).reshape(3)
    payload = None
    if payload_kg:
        payload = _payload(ctx, payload_kg)
    try:
        return _run(ctx, target, policy, base, arm, beta, mix_rng, initial, dart, obs_noise,
                    push, spin_qd, log_every, verbose, payload)
    finally:
        if payload is not None:
            joint, _name, mass = payload
            ctx["client"].changeDynamics(ctx["rid"], joint, mass=mass)


def _run(ctx, target, policy, base, arm, beta, mix_rng, initial, dart, obs_noise, push,
         spin_qd, log_every, verbose, payload):
    sim = ctx["sim"]
    client = ctx["client"]
    apply_initial_state(ctx, initial)
    task._place_target_marker(ctx, target)
    expert = task._markov_expert(ctx, target)
    stop = MarkovStop(reach_tolerance=task.REACH_TOLERANCE, max_steps=task.max_steps)
    plant = MotorCommands()
    z = np.zeros(3)
    z_max = float(expert.arm_params.z_max)
    dt = float(ctx["time_step"])
    block = max(1, int(round(0.5 / dt)))
    tails = _Tails()
    rows = _empty_rows()
    execute_expert_block = False
    expert_steps = 0
    current_time = 0.0
    step = 0
    stop_message = None
    escaped = False
    spun = False
    ee_pos = None
    base_pos = None

    while True:
        base_pos, base_ori, bearing, q_mes, qd_mes, q_pin, ee_pos = _measure(ctx)
        distance = float(np.linalg.norm(target - ee_pos))
        q_arm_true = np.asarray(q_mes[4:11], dtype=float)
        qd_arm_true = np.asarray(qd_mes[4:11], dtype=float)
        escaped, spun = _unstable(ee_pos, base_pos, q_arm_true, qd_arm_true, distance, current_time, spin_qd)
        if escaped or spun:
            stop_message = "Unstable: workspace escape" if escaped and not spun else "Unstable: joint spin"
            if escaped and spun:
                stop_message = "Unstable: joint spin and workspace escape"
            break

        if step % block == 0:
            execute_expert_block = bool(beta > 0.0 and mix_rng is not None and mix_rng.random() < beta)
        q_arm, qd_arm, q_pin_ctrl, target_rel = _control_state(
            ctx, base_pos, base_ori, bearing, q_mes, qd_mes, q_pin, target, obs_noise,
        )
        need_expert = execute_expert_block or base == "expert" or arm == "expert" or log_every > 0
        cmd = expert(q_pin_ctrl, q_arm, qd_arm, target_rel, z) if need_expert else None
        learned = None
        if not execute_expert_block and (base == "learned" or arm == "learned"):
            learned = _learned_command(
                policy, expert, cmd, q_pin_ctrl, q_arm, qd_arm, target_rel, z, base, arm,
            )
        if execute_expert_block or (base == "expert" and arm == "expert"):
            wheel_plant = np.asarray(cmd.wheel_cmd, dtype=np.float64)
            torque_plant = np.asarray(cmd.torque_arm, dtype=np.float64)
            used_expert = True
        else:
            wheel_plant = np.asarray(cmd.wheel_cmd if base == "expert" else learned[0], dtype=np.float64)
            torque_plant = np.asarray(cmd.torque_arm if arm == "expert" else learned[1], dtype=np.float64)
            used_expert = False
        if used_expert:
            expert_steps += 1
        if dart is not None and used_expert:
            wheel_plant, torque_plant = apply_dart(
                wheel_plant, torque_plant, dart["rng"], dart["wheel_std"], dart["torque_std"],
                dart["wheel_limit"], dart["torque_limit"],
            )
        if cmd is not None:
            error = np.asarray(cmd.cartesian_error, dtype=np.float64)
            reach_w = float(cmd.reach_weight)
        else:
            error = np.asarray(learned[2], dtype=np.float64)
            reach_w = float(learned[3])
        if log_every and step % log_every == 0:
            if cmd is None:
                raise RuntimeError("Row logging needs the expert command")
            obs = observation_vector(target_rel, q_pin_ctrl, q_arm, qd_arm, z)
            rows["time"].append(current_time)
            rows["observation"].append(np.asarray(obs, dtype=np.float64))
            rows["wheel_cmd"].append(clean_wheel_pair(cmd.wheel_cmd))
            rows["torque_arm"].append(np.asarray(cmd.torque_arm, dtype=np.float64).copy())
            rows["wheel_applied"].append(np.asarray(wheel_plant, dtype=np.float64).copy())
            rows["cartesian_error"].append(error.copy())
            rows["reach_weight"].append(reach_w)
            rows["tau_gravity"].append(np.asarray(cmd.tau_gravity, dtype=np.float64).copy())
            rows["ee_dist"].append(distance)
        z = integrator_update(z, error, reach_w, dt, z_max)

        if push is not None and push[0] <= current_time < push[1]:
            client.applyExternalForce(
                ctx["rid"], -1, list(push[2]), [0.0, 0.0, 0.0], client.WORLD_FRAME,
            )
        plant.SetControlCmd(
            np.concatenate((wheel_plant, torque_plant, np.zeros(6))),
            ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6,
        )
        sim.Step(plant, "torque")
        step += 1
        tails.update(step, current_time, distance, base_pos[:2], qd_arm_true, wheel_plant)
        engaged = reach_w >= 0.999
        stop_message = markov_check_stop(stop, current_time, dt, distance, engaged)
        if stop_message:
            break
        if verbose and current_time > 0.0 and int(current_time / 5.0) != int((current_time - dt) / 5.0):
            print(f"    t={current_time:.0f}s  ee {distance:.3f} m  reach_w {reach_w:.2f}", flush=True)
        current_time += dt

    # The pose that the stop decision saw. One more read would be the pose after
    # an extra, uncommanded step, so the score stays on this one.
    if ee_pos is None:
        raise RuntimeError("Episode ended before a measurement")
    final_error = float(np.linalg.norm(target - ee_pos))
    message = stop_message or ""
    unstable = bool(escaped or spun)
    reached = bool(message.startswith("Target reached")) and not unstable
    tail = tails.finish(current_time)
    return {
        "final_error": final_error,
        "final_ee_pos": np.asarray(ee_pos, dtype=float).copy(),
        "final_base_xy": np.asarray(base_pos[:2], dtype=float).copy(),
        "time": float(current_time),
        "stop_message": message,
        "reached": reached,
        "diverged": bool(escaped),
        "spun": bool(spun),
        "unstable": unstable,
        "expert_fraction": float(expert_steps / max(step, 1)),
        "payload_link": None if payload is None else payload[1],
        "rows": _stack_rows(rows),
        **tail,
    }


def run_phased(ctx, target, *, verbose=False):
    """The original phased expert, from the same reset the Markov expert uses."""
    target = np.asarray(target, dtype=float).reshape(3)
    sim = ctx["sim"]
    apply_initial_state(ctx, None)
    task._place_target_marker(ctx, target)
    base_state = BaseNavState()
    goal_xy, goal_bearing = task._base_goal(target)
    q_des = np.array(ctx["q_init_arm"], dtype=float, copy=True)
    qd_des = np.zeros(7)
    stop = PhasedStop(reach_tolerance=task.REACH_TOLERANCE, max_steps=task.max_steps)
    plant = MotorCommands()
    dt = float(ctx["time_step"])
    tails = _Tails()
    current_time = 0.0
    step = 0
    stop_message = None
    escaped = False
    spun = False
    ee_pos = None
    base_pos = None
    while True:
        base_pos, _ori, bearing, q_mes, qd_mes, q_pin, ee_pos = _measure(ctx)
        distance = float(np.linalg.norm(target - ee_pos))
        q_arm = np.asarray(q_mes[4:11], dtype=float)
        qd_arm = np.asarray(qd_mes[4:11], dtype=float)
        escaped, spun = _unstable(ee_pos, base_pos, q_arm, qd_arm, distance, current_time, spin_qd=8.0)
        if escaped or spun:
            stop_message = "Unstable: joint spin" if spun else "Unstable: workspace escape"
            break
        wheels = base_controller(
            base_state, base_pos, bearing, goal_xy, goal_bearing, current_time, ctx["base_params"],
        )
        engaged = bool(base_state.done)
        torque, q_des, qd_des = arm_controller(
            ctx["dyn_model"], q_pin, q_arm, qd_arm, q_des, ctx["q_init_arm"], target, ee_pos,
            ctx["ee_frame_id"], ctx["shoulder_frame_id"], ctx["arm_lower"], ctx["arm_upper"], dt,
            ctx["arm_params"], reach_active=engaged,
        )
        plant.SetControlCmd(
            np.concatenate((wheels, torque, np.zeros(6))),
            ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6,
        )
        sim.Step(plant, "torque")
        tails.update(step, current_time, distance, base_pos[:2], qd_arm, wheels)
        stop_message = phased_check_stop(stop, current_time, dt, distance, engaged)
        if stop_message:
            break
        if verbose and current_time > 0.0 and int(current_time / 5.0) != int((current_time - dt) / 5.0):
            print(f"    t={current_time:.0f}s  ee {distance:.3f} m  phase {base_state.phase}", flush=True)
        current_time += dt
        step += 1
    final_error = float(np.linalg.norm(target - ee_pos))
    message = stop_message or ""
    unstable = bool(escaped or spun)
    return {
        "final_error": final_error,
        "final_ee_pos": np.asarray(ee_pos, dtype=float).copy(),
        "final_base_xy": np.asarray(base_pos[:2], dtype=float).copy(),
        "time": float(current_time),
        "stop_message": message,
        "reached": bool(message.startswith("Target reached")) and not unstable,
        "diverged": bool(escaped),
        "spun": bool(spun),
        "unstable": unstable,
        "expert_fraction": 1.0,
        "payload_link": None,
        "rows": None,
        **tails.finish(current_time),
    }
