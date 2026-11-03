"""Markov expert for the Robotnik mobile base + Kinova arm reaching task.

Every command is a pure function of the current observation and the integrator state z.
There are no phases, latches or clocks inside the expert. z is not hidden: `integrator_update`
is the same few lines at data collection and at deployment, outside the network, so the expert
can label any (observation, z) pair (behaviour cloning, DAgger).

Target convention
-----------------
Targets are relative displacements: the target position minus the base origin, expressed in the
base heading frame (x forward, y left, z up, i.e. the world frame rotated by the base yaw). Build it
each step with `relative_target`. World x, y and yaw never enter the controller, so the commands are
the same wherever the robot is in the world.

What the commands depend on
---------------------------
    target_rel           (3)  target displacement, heading frame
    base roll, pitch     (2)  read from q_pin; they change the gravity compensation
    q_arm, qd_arm        (7 + 7)
    z                    (3)  integrated end-effector error, same frame
plus constant configuration (gains, q_hold, joint limits). `observation_vector` returns exactly
these 22 numbers.

Integrator
----------
The original joint reference q_des was memory of this kind, but it integrated the expert's own
IK velocity, which a learner would have had to reproduce. This one integrates the end-effector
error, which forward kinematics computes from the observation:

    e       = target_rel - ee_rel                         (3,)  see cartesian_error
    z       <- clip(w * (z + e * dt), +-z_max)                  integrator_update
    v_c     = sat(k_c * e + k_i * z)                            Cartesian velocity request

The command uses the z passed in; the update runs afterwards and produces the next z.
Multiplying by the reach weight w clears z whenever the arm is not engaged and scales it in
between, so no latch is needed. The clip is the anti-windup, in place of the old +-0.15 rad
lead limit. At rest z grows until e = 0, and k_i * z supplies the holding velocity the gravity
model gets wrong, which is zero steady-state error.

How it replaces the phased expert
---------------------------------
* Base: one continuous law drives to the set of poses `standoff` metres from the target and facing
  it. It turns on the spot when the target is off to the side, drives when roughly facing it, and
  settles facing it, without switching. A skid-steer base cannot converge smoothly to one exact
  pose (Brockett's condition), but it can converge to this set. Inside the parking band each
  wheel command fades to zero across a shoulder and is exactly zero deeper in, forward speed
  and yaw rate separately. The shoulder lies inside the band: the command is still the full
  unbraked twist at the edge, which is what carries the base in. Fading before the edge stops
  the base short of it. At zero the motors hold the base still, as the phased controller's done
  state did.
* Reach weight w in [0, 1]: rises smoothly as the base nears that set and blends the arm from holding
  q_hold (w = 0) to reaching (w = 1). The hold spring is weighted by (1 - w)^3, so a slightly
  incomplete park barely pulls the arm back toward q_hold. The IK velocity still ramps with w,
  which removes the kd * qd_des torque kick at the old phase switch.
* Arm: while reaching, the joint PD tracks the IK velocity of v_c. Position error is closed by
  the Cartesian PI above; the integral state is an input, not memory inside the expert.

Typical loop
------------
    expert = MarkovExpert(dyn_model, ee_frame_id, shoulder_frame_id, arm_lower, arm_upper,
                          q_hold_arm, ArmControllerParams(reach_limit=...),
                          BaseControllerParams(standoff=...))
    z = np.zeros(3)
    target_rel = relative_target(target_world, base_pos, base_bearing)
    cmd = expert(q_pin, q_arm, qd_arm, target_rel, z)
    z = integrator_update(z, cmd.cartesian_error, cmd.reach_weight, dt, expert.arm_params.z_max)
    # apply cmd.wheel_cmd and cmd.torque_arm
    # World-frame distance, not cmd.ee_error: the heading-frame one differs by a
    # few millimetres once the base rolls or pitches.
    ee_dist = np.linalg.norm(target_world - ee_world)
    stop = check_stop(criteria, t, dt, ee_dist, arm_engaged=cmd.reach_weight >= 0.999)
"""
from dataclasses import dataclass

import numpy as np
import pinocchio as pin


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def wrap_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def smoothstep(x):
    """C1 ramp: 0 for x <= 0, 1 for x >= 1, smooth in between."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def quaternion2bearing(q_w, q_x, q_y, q_z):
    """Convert quaternion orientation to planar bearing (yaw angle)."""
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()
    base_euler = pin.rpy.matrixToRpy(quat.toRotationMatrix())
    return base_euler[2]


def relative_target(target_world, base_pos, base_bearing):
    """World target -> displacement from the base origin in the base heading frame.

    base_pos (3,) and base_bearing must describe the free-flyer frame of q_pin, i.e. q_pin[:3]
    and its yaw. If your targets are specified relative to the start pose, convert them to world
    points once at the start of the episode and call this every step.
    """
    d = np.asarray(target_world, dtype=float) - np.asarray(base_pos, dtype=float)[:3]
    c, s = np.cos(base_bearing), np.sin(base_bearing)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])


def heading_frame_configuration(q_pin):
    """Re-express q_pin in the base heading frame: base at the origin, yaw removed.

    Roll and pitch are kept, so gravity compensation still sees the base tilt. Joint entries are
    passed through unchanged. Assumes the first joint is a free-flyer: q_pin = [x, y, z, qx, qy, qz, qw, ...].
    """
    q = np.array(q_pin, dtype=float, copy=True)
    quat = pin.Quaternion(q[6], q[3], q[4], q[5])
    quat.normalize()
    rot = quat.toRotationMatrix()
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    rz_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    q[:3] = 0.0
    q[3:7] = pin.Quaternion(rz_inv @ rot).coeffs()   # (x, y, z, w)
    return q


def observation_vector(target_rel, q_pin, q_arm, qd_arm, z):
    """The 22 numbers the expert's commands depend on: [target_rel, roll, pitch, q_arm, qd_arm, z].

    z is the cartesian integrator (see integrator_update), not a hidden state of the network.
    Continuous (unlimited) joints can wind past +-pi; for a network input, consider feeding
    sin/cos of those angles instead of the raw values.
    """
    quat = pin.Quaternion(q_pin[6], q_pin[3], q_pin[4], q_pin[5])
    quat.normalize()
    roll, pitch, _ = pin.rpy.matrixToRpy(quat.toRotationMatrix())
    return np.concatenate([np.asarray(target_rel, dtype=float), [roll, pitch],
                           np.asarray(q_arm, dtype=float), np.asarray(qd_arm, dtype=float),
                           np.asarray(z, dtype=float)])


# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------

@dataclass
class ArmControllerParams:
    """Tuning parameters of the arm reaching controller."""
    reach_limit: float                   # m, max EE distance from the shoulder the arm is asked to reach
    kp: float = 100.0                    # joint PD position gain (posture hold and limit protection)
    kd: float = 30.0                     # joint PD velocity gain (velocity tracking while reaching)
    max_joint_vel: float = 1.0           # rad/s
    max_torque: float = 30.0             # N*m
    cartesian_kp: float = 1.5            # k_c, 1/s, proportional gain on the cartesian error
    cartesian_ki: float = 1.5            # k_i, 1/s^2, integral gain. With k_c, a held 1 cm error
                                         # makes the I term match the P term after about a second
    z_max: float = 0.15                  # m*s, element-wise clip. k_i*z_max = 0.225 m/s, inside the
                                         # cartesian speed cap; this is the anti-windup that replaces
                                         # the phased expert's +-0.15 rad joint-lead limit
    max_cartesian_speed: float = 0.25    # m/s
    min_ik_damping: float = 0.01
    max_ik_damping: float = 0.10
    singularity_threshold: float = 0.15
    limit_margin: float = 0.05           # rad, joints this close to a limit are treated as saturated


@dataclass
class BaseControllerParams:
    """Tuning parameters of the mobile-base law."""
    standoff: float                      # m, horizontal distance from base origin to target when parked
    kp_pos: float = 1.0
    kp_ori: float = 3.0
    max_lin_vel: float = 0.25            # m/s
    max_ang_vel: float = 1.0             # rad/s, turning on the spot
    max_ang_vel_driving: float = 0.3     # rad/s at full speed: skid-steer slips when turning hard while driving
    wheel_radius: float = 0.11
    # Effective track width for skid-steer (calibrated in simulation: yaw rate ~0.34 rad/s per rad/s of wheel speed)
    wheel_base_width: float = 0.66


@dataclass
class ReachGateParams:
    """Smoothstep bands over which the arm blends from holding its posture to reaching.

    The arm reaches fully once |rho - standoff| < pos_inner and |bearing error| < ang_inner, and only
    holds its posture once either exceeds its outer value. ang_inner is 0.20 rad (~11 deg). With the
    yaw command left on, the base does get closer than the old 0.10 rad gate, but it does not hold
    that tighter heading once the arm is out, and the leftover yaw command is what scrubs the wheels.
    J1 absorbs the 0.20 rad. The wheel brake keeps the full unbraked command up to these inner
    edges, fades to zero across pos_brake_edge / ang_brake_edge inside them, and is exactly zero
    deeper in. A fade outside the edge stalls the base just short of the band. Keep pos_inner at
    or above where wheel friction stops the base (a few cm short of the standoff). Wider bands
    are easier to regress; narrower bands approach the old hard switch.
    """
    pos_inner: float = 0.08              # m
    pos_outer: float = 0.40              # m
    ang_inner: float = 0.20              # rad (~11 deg)
    ang_outer: float = 0.50              # rad
    pos_brake_edge: float = 0.02         # m, soft shoulder inside pos_inner
    ang_brake_edge: float = 0.05         # rad, soft shoulder inside ang_inner


# ----------------------------------------------------------------------------------------------
# Base
# ----------------------------------------------------------------------------------------------

def base_errors(target_rel, params):
    """(rho - standoff, bearing of the target in the base frame) for a heading-frame displacement."""
    rho = np.hypot(target_rel[0], target_rel[1])
    bearing = np.arctan2(target_rel[1], target_rel[0])
    return rho - params.standoff, bearing


def base_twist(target_rel, params):
    """Forward speed (m/s) and yaw rate (rad/s) for the unbraked base law.

    v is proportional to the standoff error and gated by how well the base faces the target, so it
    turns on the spot when the target is off to the side and reverses if it ends up too close.
    The turn-rate cap shrinks smoothly with forward speed to limit skid-steer slip.
    Smooth everywhere except with the target exactly behind (bearing = +-pi) or exactly on the
    base origin (rho = 0, not visited because standoff > 0).
    """
    dist_err, bearing_err = base_errors(target_rel, params)
    heading_gate = max(0.0, np.cos(bearing_err)) ** 2
    v_cmd = params.max_lin_vel * np.tanh(params.kp_pos * dist_err / params.max_lin_vel) * heading_gate
    speed_frac = min(1.0, abs(v_cmd) / params.max_lin_vel)
    w_cap = params.max_ang_vel + (params.max_ang_vel_driving - params.max_ang_vel) * speed_frac
    w_cmd = w_cap * np.tanh(params.kp_ori * bearing_err / w_cap)
    return v_cmd, w_cmd


def twist_to_wheels(v_cmd, w_cmd, params):
    """Wheel speeds [front_right, front_left, back_left, back_right] (rad/s)."""
    lw = (v_cmd - w_cmd * params.wheel_base_width / 2) / params.wheel_radius
    rw = (v_cmd + w_cmd * params.wheel_base_width / 2) / params.wheel_radius
    return np.array([rw, lw, lw, rw])


def base_controller(target_rel, params):
    """Unbraked wheel speeds. `parked_wheel_cmd` is what the expert applies."""
    v_cmd, w_cmd = base_twist(target_rel, params)
    return twist_to_wheels(v_cmd, w_cmd, params)


def brake_scale(abs_err, inner, edge):
    """1 at and outside the parking edge, 0 a shoulder's width inside it, smooth between.

    The shoulder sits inside the window, so the command is still the full unbraked twist
    when the error is at `inner`. Fading outside that edge leaves the base short of the band.
    """
    if edge <= 0.0:
        return 0.0 if abs_err <= inner else 1.0
    return float(smoothstep((abs_err - (inner - edge)) / edge))


def parked_wheel_cmd(target_rel, params, gate):
    """Wheel speeds faded to zero across a shoulder inside the parking band, per axis.

    Forward speed is the unbraked command outside pos_inner, falls with a smoothstep across
    pos_brake_edge inside that edge, and is exactly zero deeper in. Yaw rate does the same
    about ang_inner. The two fades are separate, so a heading already inside its deadzone
    is not turned again while the base closes the last few centimetres.

    The shoulder is inside the window. Full authority up to the edge is what carries the
    base in. A fade outside the edge — including scaling the wheels by (1 - reach weight),
    which blends over the whole approach — is weaker than the command for a short distance
    before the band, and the base stops there with the wheels creeping. Once inside the
    deadzone a zero velocity command holds the base the way the phased controller's done
    state does.
    """
    dist_err, bearing_err = base_errors(target_rel, params)
    v_cmd, w_cmd = base_twist(target_rel, params)
    v_cmd *= brake_scale(abs(dist_err), gate.pos_inner, gate.pos_brake_edge)
    w_cmd *= brake_scale(abs(bearing_err), gate.ang_inner, gate.ang_brake_edge)
    return twist_to_wheels(v_cmd, w_cmd, params)


def reach_weight(target_rel, base_params, gate):
    """Arm weight in [0, 1]: 0 while the base is far from its parking set, 1 once it is there."""
    dist_err, bearing_err = base_errors(target_rel, base_params)
    w_pos = smoothstep((gate.pos_outer - abs(dist_err)) / (gate.pos_outer - gate.pos_inner))
    w_ang = smoothstep((gate.ang_outer - abs(bearing_err)) / (gate.ang_outer - gate.ang_inner))
    return float(w_pos * w_ang)


# ----------------------------------------------------------------------------------------------
# Arm
# ----------------------------------------------------------------------------------------------

def cartesian_error(target_rel, ee_rel, shoulder_rel, reach_limit):
    """End-effector error e = target_rel - ee_rel, in the heading frame.

    A target further than reach_limit from the shoulder is pulled back onto that sphere first,
    so inside the reachable set this is exactly target_rel - ee_rel, and beyond it e is zero at
    the closest pose the arm is asked to hold. Both the integrator and the Cartesian loop use
    this vector: it is forward kinematics of the observation plus a constant limit, not the IK.
    """
    target_rel = np.asarray(target_rel, dtype=float)
    ee_rel = np.asarray(ee_rel, dtype=float)
    shoulder_rel = np.asarray(shoulder_rel, dtype=float)
    offset = target_rel - shoulder_rel
    offset_norm = np.linalg.norm(offset)
    if offset_norm > reach_limit:
        offset = offset * (reach_limit / offset_norm)
    return shoulder_rel + offset - ee_rel


def cartesian_velocity(ee_error, z, params):
    """v_c = sat(k_c * e + k_i * z). The saturation is the cartesian speed limit."""
    v = params.cartesian_kp * np.asarray(ee_error, dtype=float) + params.cartesian_ki * np.asarray(z, dtype=float)
    speed = np.linalg.norm(v)
    if speed > params.max_cartesian_speed:
        v = v * (params.max_cartesian_speed / speed)
    return v


def integrator_update(z, ee_error, reach_weight, dt, z_max):
    """z <- clip(w * (z + e * dt), +-z_max).

    Run this after the expert call, with the z that call was given and the error it returns
    (`cmd.cartesian_error`). The command depends on the current z; this value is the next one.
    w = 0 clears the state. The clip is the anti-windup.
    """
    z_next = float(reach_weight) * (np.asarray(z, dtype=float) + np.asarray(ee_error, dtype=float) * dt)
    return np.clip(z_next, -z_max, z_max)


def reach_velocity(model, data, q_canon, q_arm, v_cartesian, ee_frame_id,
                   arm_lower, arm_upper, params, arm_v):
    """Joint velocity (7,) that realises a Cartesian velocity request.

    Singularity-damped, joint-limit aware differential IK (same as the original expert).
    Everything is in the heading frame; q_canon comes from heading_frame_configuration.
    v_cartesian is v_c from cartesian_velocity.
    """
    pin.computeJointJacobians(model, data, q_canon)
    jac = pin.computeFrameJacobian(model, data, q_canon, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    jac_arm = jac[:3, arm_v]

    # Increase damping near singular poses to keep joint velocities bounded
    sigma_min = np.linalg.svd(jac_arm, compute_uv=False)[-1]
    damping = np.interp(sigma_min, [0.0, params.singularity_threshold],
                        [params.max_ik_damping, params.min_ik_damping])

    # Saturation-aware solve: joints at a limit that the motion would push further into are frozen,
    # and their share of the motion is redistributed to the remaining joints
    has_limits = arm_lower < arm_upper
    free = np.ones(7, dtype=bool)
    qd_ik = np.zeros(7)
    for _ in range(7):
        jac_free = jac_arm[:, free]
        regularized = jac_free @ jac_free.T + damping**2 * np.eye(3)
        qd_ik = np.zeros(7)
        qd_ik[free] = jac_free.T @ np.linalg.solve(regularized, v_cartesian)
        pushing_lower = has_limits & (q_arm <= arm_lower + params.limit_margin) & (qd_ik < 0)
        pushing_upper = has_limits & (q_arm >= arm_upper - params.limit_margin) & (qd_ik > 0)
        newly_blocked = (pushing_lower | pushing_upper) & free
        if not newly_blocked.any():
            break
        free &= ~newly_blocked
        qd_ik[:] = 0.0
    max_joint_speed = np.max(np.abs(qd_ik))
    if max_joint_speed > params.max_joint_vel:
        qd_ik *= params.max_joint_vel / max_joint_speed
    return qd_ik


def arm_references(q_arm, q_hold_arm, qd_ik, w, arm_lower, arm_upper, params):
    """Joint references (q_ref, qd_ref) for the blended hold/reach law.

    The hold spring kp * (q_ref - q) is weighted by (1 - w)^3. w = 0 still holds q_hold
    exactly and w = 1 is still pure IK velocity tracking, but a slightly incomplete park
    (w near 1) no longer pulls the arm back toward the tucked posture.
    q_ref is kept inside the limit margins, so a joint that drifts into a margin is pushed back.
    """
    has_limits = arm_lower < arm_upper
    hold = (1.0 - w) ** 3
    q_ref = np.asarray(q_arm, dtype=float) + hold * (np.asarray(q_hold_arm, dtype=float) - q_arm)
    q_ref = np.where(has_limits,
                     np.clip(q_ref, arm_lower + params.limit_margin, arm_upper - params.limit_margin),
                     q_ref)
    return q_ref, w * qd_ik


def arm_torque(q_arm, qd_arm, q_ref, qd_ref, tau_gravity, params):
    """Gravity-compensated joint PD law. Kept separate so it can sit under a learned reference policy."""
    torque = params.kp * (q_ref - q_arm) + params.kd * (qd_ref - qd_arm) + tau_gravity
    return np.clip(torque, -params.max_torque, params.max_torque)


# ----------------------------------------------------------------------------------------------
# Expert
# ----------------------------------------------------------------------------------------------

@dataclass
class ExpertCommand:
    torque_arm: np.ndarray        # (7,) N*m
    wheel_cmd: np.ndarray         # (4,) rad/s, [front_right, front_left, back_left, back_right]
    q_ref_arm: np.ndarray         # (7,) rad, reference the joint PD tracks
    qd_ref_arm: np.ndarray        # (7,) rad/s
    reach_weight: float           # 0 = arm holds posture, 1 = arm reaching
    ee_error: float               # m, end-effector to target distance (true target, not the sphere)
    cartesian_error: np.ndarray   # (3,) e used by the PI and by integrator_update
    dist_error: float             # m, rho - standoff
    bearing_error: float          # rad, bearing of the target in the base frame
    tau_gravity: np.ndarray       # (7,) N*m, gravity term added inside arm_torque, before the clip


class MarkovExpert:
    """Expert that is Markov in the observation and the integrator state z.

    The object holds constant configuration only. z is an argument of __call__; this object does
    not store or update it. Its pinocchio Data is a scratch buffer that is fully overwritten on
    every call, so nothing carries over between calls, and the caller's dyn_model.pin_data
    (world-frame kinematics) is left untouched.
    """

    def __init__(self, dyn_model, ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, q_hold_arm,
                 arm_params, base_params, gate_params=None, arm_v=slice(6, 13)):
        self.model = dyn_model.pin_model
        if self.model.joints[1].shortname() != "JointModelFreeFlyer":
            raise ValueError("Expected the first joint of the pinocchio model to be a free-flyer")
        self._data = self.model.createData()
        self.ee_frame_id = ee_frame_id
        self.shoulder_frame_id = shoulder_frame_id
        self.arm_lower = np.array(arm_lower, dtype=float)
        self.arm_upper = np.array(arm_upper, dtype=float)
        self.q_hold_arm = np.array(q_hold_arm, dtype=float)
        self.arm_params = arm_params
        self.base_params = base_params
        self.gate_params = gate_params if gate_params is not None else ReachGateParams()
        self.arm_v = arm_v

    def cartesian_feedback(self, q_pin, target_rel):
        """Reach weight and cartesian error for ``integrator_update``.

        Forward kinematics only. The learned policy does not emit these, and the
        integrator outside the network needs the same two numbers the expert uses.
        """
        target_rel = np.asarray(target_rel, dtype=float)
        reach_w = reach_weight(target_rel, self.base_params, self.gate_params)
        q_canon = heading_frame_configuration(q_pin)
        pin.forwardKinematics(self.model, self._data, q_canon)
        pin.updateFramePlacements(self.model, self._data)
        ee_rel = self._data.oMf[self.ee_frame_id].translation.copy()
        shoulder_rel = self._data.oMf[self.shoulder_frame_id].translation.copy()
        ee_error_vec = cartesian_error(target_rel, ee_rel, shoulder_rel, self.arm_params.reach_limit)
        return ee_error_vec, reach_w

    def __call__(self, q_pin, q_arm, qd_arm, target_rel, z):
        """Commands for one (observation, z) pair.

        q_pin: full pinocchio configuration (world x, y and yaw are ignored);
        q_arm, qd_arm: (7,) arm joint angles and velocities;
        target_rel: (3,) target displacement in the base heading frame (see relative_target);
        z: (3,) cartesian error integrator in that same frame. Updated by integrator_update,
        not by this call.
        """
        model, data = self.model, self._data
        q_arm = np.asarray(q_arm, dtype=float)
        qd_arm = np.asarray(qd_arm, dtype=float)
        target_rel = np.asarray(target_rel, dtype=float)
        z = np.asarray(z, dtype=float)
        if z.shape != (3,):
            raise ValueError(f"integrator z must have shape (3,), got {z.shape}")

        dist_err, bearing_err = base_errors(target_rel, self.base_params)
        w = reach_weight(target_rel, self.base_params, self.gate_params)
        wheel_cmd = parked_wheel_cmd(target_rel, self.base_params, self.gate_params)

        q_canon = heading_frame_configuration(q_pin)
        pin.forwardKinematics(model, data, q_canon)
        pin.updateFramePlacements(model, data)
        ee_rel = data.oMf[self.ee_frame_id].translation.copy()
        shoulder_rel = data.oMf[self.shoulder_frame_id].translation.copy()
        ee_error_vec = cartesian_error(target_rel, ee_rel, shoulder_rel, self.arm_params.reach_limit)

        if w > 0.0:
            v_c = cartesian_velocity(ee_error_vec, z, self.arm_params)
            qd_ik = reach_velocity(model, data, q_canon, q_arm, v_c,
                                   self.ee_frame_id, self.arm_lower, self.arm_upper,
                                   self.arm_params, self.arm_v)
        else:
            qd_ik = np.zeros(7)

        q_ref, qd_ref = arm_references(q_arm, self.q_hold_arm, qd_ik, w,
                                       self.arm_lower, self.arm_upper, self.arm_params)
        tau_g = pin.computeGeneralizedGravity(model, data, q_canon)[self.arm_v]
        torque = arm_torque(q_arm, qd_arm, q_ref, qd_ref, tau_g, self.arm_params)

        return ExpertCommand(
            torque_arm=torque,
            wheel_cmd=wheel_cmd,
            q_ref_arm=q_ref,
            qd_ref_arm=qd_ref,
            reach_weight=w,
            ee_error=float(np.linalg.norm(target_rel - ee_rel)),
            cartesian_error=ee_error_vec,
            dist_error=float(dist_err),
            bearing_error=float(bearing_err),
            tau_gravity=np.asarray(tau_g, dtype=float),
        )


# ----------------------------------------------------------------------------------------------
# Episode termination (not part of the policy; it may keep state)
# ----------------------------------------------------------------------------------------------

@dataclass
class StopCriteria:
    """Thresholds and running state for the simulation stop conditions."""
    reach_tolerance: float                     # m
    max_steps: int                             # 0 disables the step cap
    min_time: float = 8.0                      # s before reached / stalled can trigger
    stall_window: float = 3.0                  # s without meaningful improvement of the EE error
    stall_improvement: float = 0.002           # m
    dwell_time: float = 0.50                   # s the EE must stay inside the tolerance
    flythrough_rise: float = 0.02              # m above the best that voids that best
    best_ee_error: float = np.inf
    best_ee_error_time: float = 0.0
    inside_since: float = -1.0                 # < 0 when the EE is outside the tolerance


def check_stop(criteria, current_time, time_step, ee_dist_error, arm_engaged):
    """Stop when the arm has settled, not when the end-effector sweeps past the target.

    Returns a message describing why the simulation should stop, or None to continue.
    `arm_engaged` replaces base_navigation_done; pass `cmd.reach_weight >= 0.999`.
    That gate opens while the base is still yawing into its wheel deadzone, so the
    distance can dive through the tolerance and bounce straight back out. Pass the
    world-frame distance: it is what the episode is scored on.

    A one-sample crossing is not a reach. The distance has to stay inside the
    tolerance for `dwell_time`. The minimum of a dive is not a pose the arm is
    holding: a later rise of `flythrough_rise` voids it, and the recovery is timed
    as a new approach. Time before `min_time` does not count toward the stall
    window, so a best recorded on the way in cannot expire on the first step that
    is allowed to stop. Stall tracking only runs while the arm is engaged. A base
    that never reaches its parking set is caught by max_steps.
    """
    if criteria.max_steps and current_time / time_step >= criteria.max_steps:
        return f"Reached step cap {criteria.max_steps} at t={current_time:.2f}s"

    if not arm_engaged:
        criteria.best_ee_error = np.inf
        criteria.best_ee_error_time = current_time
        criteria.inside_since = -1.0
        return None

    # The recorded best was a pass through the target. Time the recovery instead.
    if ee_dist_error > criteria.best_ee_error + criteria.flythrough_rise:
        criteria.best_ee_error = np.inf
        criteria.best_ee_error_time = current_time

    if ee_dist_error < criteria.best_ee_error - criteria.stall_improvement:
        criteria.best_ee_error = ee_dist_error
        criteria.best_ee_error_time = current_time

    if ee_dist_error < criteria.reach_tolerance:
        if criteria.inside_since < 0.0:
            criteria.inside_since = current_time
        held = current_time - criteria.inside_since
        if held >= criteria.dwell_time and current_time > criteria.min_time:
            return (f"Target reached within {criteria.reach_tolerance*1000:.0f}mm tolerance "
                    f"at t={current_time:.2f}s!")
        return None

    criteria.inside_since = -1.0
    # The window starts at the later of the last real improvement and min_time.
    settled_for = current_time - max(criteria.best_ee_error_time, criteria.min_time)
    if settled_for > criteria.stall_window and current_time > criteria.min_time:
        return (f"Target not reachable: closest pose reached at t={current_time:.2f}s "
                f"(best EE error {criteria.best_ee_error:.4f} m, current {ee_dist_error:.4f} m)")
    return None


# ----------------------------------------------------------------------------------------------
# Gripper setup (unchanged)
# ----------------------------------------------------------------------------------------------

FINGER_JOINT_NAMES = [
    "robot_j2s7s300_joint_finger_1",
    "robot_j2s7s300_joint_finger_tip_1",
    "robot_j2s7s300_joint_finger_2",
    "robot_j2s7s300_joint_finger_tip_2",
    "robot_j2s7s300_joint_finger_3",
    "robot_j2s7s300_joint_finger_tip_3",
]


def configure_gripper_fingers(sim, client, rid, fixed_angles=None):
    """Reset the gripper fingers to fixed angles and hold them there.

    Intercepts the motor torque commands so wheels (0..3) and arm (4..10) keep their
    normal torque control while fingers (11..16) are position-held.
    Returns the finger target angles (initial config angles if `fixed_angles` is None).
    """
    finger_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if client.getJointInfo(rid, j)[1].decode() in FINGER_JOINT_NAMES
    ]
    raw_init_angles = np.array(sim.GetInitMotorAngles())
    target_finger_angles = fixed_angles if fixed_angles is not None else raw_init_angles[11:17]
    for i, jid in enumerate(finger_joint_ids):
        client.resetJointState(rid, jid, targetValue=float(target_finger_angles[i]), targetVelocity=0.0)
    sim.ReceiveObservation()

    orig_set_torque = sim._SetMotorTorqueByIds

    def custom_set_torque(motor_ids, commands, index=0):
        orig_set_torque(motor_ids[:11], commands[:11], index=index)
        sim.pybullet_client.setJointMotorControlArray(
            bodyIndex=sim.bot[index].bot_pybullet,
            jointIndices=motor_ids[11:17],
            controlMode=sim.pybullet_client.POSITION_CONTROL,
            targetPositions=np.asarray(target_finger_angles).tolist(),
            forces=[5.0] * 6
        )

    sim._SetMotorTorqueByIds = custom_set_torque
    return target_finger_angles
