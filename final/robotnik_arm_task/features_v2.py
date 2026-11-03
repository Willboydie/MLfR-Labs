"""Features, target draws, and row filters for the split base/arm policy.

The frozen test is the original 24-point assessment grid plus 20 targets drawn
once and then never redrawn. Training draws come from the band the arm can
actually reach, and they are rejected if they land on a frozen target.

Joints 1, 3, 5 and 7 are continuous on this arm. Their raw angles wind (q1
reached about -33 rad in the old data), so the network sees sin and cos.
The other three joints stay in radians.

Rows in which joint 1 or 3 is spinning are dropped, together with the quarter
second before the spin starts. Those rows carried the largest torques in the
old set and pulled mean squared error away from the small commands that
actually hold the target.

Run ``python features_v2.py`` for the algebra checks. They do not start a simulator.
"""
import os

os.environ.setdefault("ROBOENV_HEADLESS", "1")

import numpy as np

from robotnik_markov_expert import (
    BaseControllerParams,
    ReachGateParams,
    base_errors,
    parked_wheel_cmd,
)

# Same numbers as robotnik_arm_task.STANDOFF_*. Checked against that module below.
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]

# World-frame box for new targets, metres. XY is centred on the reset base.
XY_HALF = 1.30
MIN_RADIUS = 0.45
Z_LO = 0.25
Z_HI = 1.25
BEYOND_LO = (0.08, 0.25)
BEYOND_HI = (1.25, 1.45)
BEYOND_FRAC = 0.125
MIN_SEPARATION = 0.08

# Joints 1, 3, 5, 7 in the 7-vector. Confirmed against the simulator limits at collection.
CONTINUOUS_JOINTS = (0, 2, 4, 6)
SPIN_QD = 3.0          # rad/s on joint 1 or 3. The expert caps IK at 1 rad/s.
SPIN_Q = 2.0 * np.pi   # rad. Past one revolution the joint has wound up.
SPIN_MARGIN_S = 0.25

PARK_WHEEL = 1e-8      # rad/s. The expert's deadzone is an exact zero, not a small number.
SHOULDER_WHEEL = 0.15  # rad/s. Upweight the fade into that deadzone, not only the zero itself.

N_HELDOUT = 20
HELDOUT_SEED = 17
TRAIN_SEED = 1

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def standoff_for_height(z):
    """Horizontal park distance (m) for a world-frame target height."""
    return float(np.interp(float(z), STANDOFF_Z_POINTS, STANDOFF_POINTS))


def continuous_mask_from_limits(lower, upper):
    """True where the joint is unlimited. Pinocchio marks those with lower >= upper."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return ~(lower < upper)


def encode_q(q, continuous):
    """Replace each continuous angle with sin and cos. Limited joints stay in radians.

    ``q`` is (7,) or (N, 7). ``continuous`` is a length-7 boolean mask.
    """
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q.reshape(1, -1)
    continuous = np.asarray(continuous, dtype=bool).reshape(7)
    cols = []
    for index in range(7):
        if continuous[index]:
            cols.append(np.sin(q[:, index]))
            cols.append(np.cos(q[:, index]))
        else:
            cols.append(q[:, index])
    encoded = np.column_stack(cols)
    return encoded[0] if single else encoded


def q_feature_names(continuous):
    continuous = np.asarray(continuous, dtype=bool).reshape(7)
    names = []
    for index in range(7):
        joint = index + 1
        if continuous[index]:
            names.append(f"sin_q{joint}")
            names.append(f"cos_q{joint}")
        else:
            names.append(f"q{joint}")
    return names


def arm_feature_names(continuous):
    names = ["target_rel_x", "target_rel_y", "target_rel_z", "roll", "pitch"]
    names += q_feature_names(continuous)
    names += [f"qd{i}" for i in range(1, 8)]
    names += ["z_x", "z_y", "z_z", "e_x", "e_y", "e_z", "reach_w"]
    names += [f"tau_g{i}" for i in range(1, 8)]
    return names


def base_feature_names():
    return ["target_rel_x", "target_rel_y", "target_rel_z", "standoff", "dist_err", "bearing_err", "abs_dist_err", "abs_bearing_err"]


def base_feature_matrix(target_rel, standoff):
    """(N, 6). ``standoff`` is scalar or (N,)."""
    target_rel = np.asarray(target_rel, dtype=np.float64)
    single = target_rel.ndim == 1
    if single:
        target_rel = target_rel.reshape(1, 3)
    standoff = np.asarray(standoff, dtype=np.float64)
    if standoff.ndim == 0:
        standoff = np.full(len(target_rel), float(standoff))
    dist = np.hypot(target_rel[:, 0], target_rel[:, 1]) - standoff
    bearing = np.arctan2(target_rel[:, 1], target_rel[:, 0])
    matrix = np.column_stack([target_rel, standoff, dist, bearing, np.abs(dist), np.abs(bearing)])
    return matrix[0] if single else matrix


def arm_feature_matrix(target_rel, roll_pitch, q, qd, z, error, reach_w, tau_g, continuous):
    """(N, D) or (D,) if a single row is passed. D is 37 when four joints are continuous."""
    target_rel = np.atleast_2d(np.asarray(target_rel, dtype=np.float64))
    roll_pitch = np.atleast_2d(np.asarray(roll_pitch, dtype=np.float64))
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    qd = np.atleast_2d(np.asarray(qd, dtype=np.float64))
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    error = np.atleast_2d(np.asarray(error, dtype=np.float64))
    tau_g = np.atleast_2d(np.asarray(tau_g, dtype=np.float64))
    n = len(target_rel)
    reach = np.asarray(reach_w, dtype=np.float64).reshape(-1)
    if reach.size == 1 and n != 1:
        reach = np.full(n, float(reach[0]))
    if reach.size != n:
        raise ValueError(f"reach_w has length {reach.size}, expected {n}")
    encoded = encode_q(q, continuous)
    matrix = np.column_stack([
        target_rel, roll_pitch, encoded, qd, z, error, reach, tau_g,
    ])
    if np.asarray(target_rel).shape[0] == 1 and np.asarray(q).ndim == 2 and len(q) == 1:
        # atleast_2d hides whether the caller passed a row. Return a row when n == 1
        # only if every input was a single sample. Callers that want a matrix pass n >= 1
        # explicitly through the batch path below. A length-1 batch still returns (1, D)
        # when ``as_row`` is not requested. The single-row path is arm_features_row.
        return matrix
    return matrix


def arm_features_row(target_rel, roll_pitch, q, qd, z, error, reach_w, tau_g, continuous):
    """One arm feature vector, shape (D,)."""
    matrix = arm_feature_matrix(
        np.asarray(target_rel, dtype=np.float64).reshape(1, 3),
        np.asarray(roll_pitch, dtype=np.float64).reshape(1, 2),
        np.asarray(q, dtype=np.float64).reshape(1, 7),
        np.asarray(qd, dtype=np.float64).reshape(1, 7),
        np.asarray(z, dtype=np.float64).reshape(1, 3),
        np.asarray(error, dtype=np.float64).reshape(1, 3),
        [float(reach_w)],
        np.asarray(tau_g, dtype=np.float64).reshape(1, 7),
        continuous,
    )
    return matrix[0]


def spin_mask(q, qd):
    """True on rows where joint 1 or 3 is spinning or has wound past one revolution."""
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    speed = np.maximum(np.abs(qd[:, 0]), np.abs(qd[:, 2]))
    wind = np.maximum(np.abs(q[:, 0]), np.abs(q[:, 2]))
    return (speed > SPIN_QD) | (wind > SPIN_Q)


def filter_logged(rows, spun, abort_time):
    """Drop a spin and the quarter second before it.

    The aborting sample itself is not in ``rows``: the episode stops when that
    sample is measured, before it is logged. ``abort_time`` is that sample's time.
    """
    if rows is None or len(rows["time"]) == 0:
        return rows
    time = np.asarray(rows["time"], dtype=np.float64)
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1e-3
    observation = np.asarray(rows["observation"], dtype=np.float64)
    mask = keep_mask(observation[:, 5:12], observation[:, 12:19], dt)
    if spun:
        mask = mask & (time < float(abort_time) - SPIN_MARGIN_S)
    if not np.any(mask):
        return {key: value[:0] for key, value in rows.items()}
    return {key: value[mask] for key, value in rows.items()}


def keep_mask(q, qd, dt):
    """False on a spin and on the quarter second that leads into it, and on everything after."""
    bad = spin_mask(q, qd)
    if not np.any(bad):
        return ~bad
    first = int(np.flatnonzero(bad)[0])
    margin = int(round(SPIN_MARGIN_S / dt))
    start = max(0, first - margin)
    bad = np.zeros(len(bad), dtype=bool)
    bad[start:] = True
    return ~bad


def parked_labels(wheels):
    """Exact wheel-deadzone rows. ``wheels`` is (N, 2)."""
    wheels = np.asarray(wheels, dtype=np.float64)
    return np.max(np.abs(wheels), axis=1) < PARK_WHEEL


def sample_weights(wheels, reach_w, error):
    """Make the park and the last centimetres count in a plain squared error.

    Exact zeros are rare along a trajectory, and a 1 N·m holding torque is small
    next to a saturated joint. Both get a higher weight. Weights are relative.
    """
    wheels = np.asarray(wheels, dtype=np.float64)
    reach = np.asarray(reach_w, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64)
    magnitude = np.max(np.abs(wheels), axis=1)
    parked = magnitude < PARK_WHEEL
    shoulder = (~parked) & (magnitude < SHOULDER_WHEEL)
    near = (reach >= 0.999) & (np.linalg.norm(error, axis=1) < 0.05)
    reaching = reach >= 0.999
    weights = np.ones(len(wheels), dtype=np.float64)
    weights[reaching] = 2.0
    weights[near] = 8.0
    weights[shoulder] = np.maximum(weights[shoulder], 5.0)
    weights[parked] = np.maximum(weights[parked], 20.0)
    return weights


def _too_close(candidate, others, separation):
    if others is None or len(others) == 0:
        return False
    delta = np.asarray(others, dtype=float) - np.asarray(candidate, dtype=float)
    return bool(np.any(np.linalg.norm(delta, axis=1) < separation))


def sample_targets(n, origin_xy, rng, avoid=None, beyond_frac=BEYOND_FRAC, separation=MIN_SEPARATION):
    """``n`` targets in the reachable band, with about ``beyond_frac`` slightly outside it.

    Nothing inside the base footprint, and nothing within ``separation`` of ``avoid``
    or of another draw. ``origin_xy`` is the reset base position.
    """
    if n < 1:
        raise ValueError("n must be positive")
    origin_xy = np.asarray(origin_xy, dtype=float).reshape(2)
    avoid = np.zeros((0, 3)) if avoid is None else np.asarray(avoid, dtype=float).reshape(-1, 3)
    chosen = []
    tries = 0
    limit = max(100000, 5000 * n)
    while len(chosen) < n:
        tries += 1
        if tries > limit:
            raise RuntimeError(f"Could only place {len(chosen)} of {n} targets")
        beyond = float(rng.random()) < beyond_frac
        xy = origin_xy + rng.uniform(-XY_HALF, XY_HALF, size=2)
        if np.hypot(xy[0] - origin_xy[0], xy[1] - origin_xy[1]) < MIN_RADIUS:
            continue
        if beyond:
            band = BEYOND_LO if float(rng.random()) < 0.5 else BEYOND_HI
            z = float(rng.uniform(band[0], band[1]))
        else:
            z = float(rng.uniform(Z_LO, Z_HI))
        candidate = np.array([xy[0], xy[1], z], dtype=float)
        if _too_close(candidate, avoid, separation) or _too_close(candidate, chosen, separation):
            continue
        chosen.append(candidate)
    return np.vstack(chosen)


def beyond_band(z):
    """True when a height is outside [0.25, 1.25] m."""
    z = np.asarray(z, dtype=float)
    return (z < Z_LO) | (z > Z_HI)


def wheel_pair_from_expert(target_rel, standoff, gate=None):
    """``[right, left]`` from the expert base law at this displacement and standoff."""
    params = BaseControllerParams(standoff=float(standoff))
    gate = ReachGateParams() if gate is None else gate
    wheels = parked_wheel_cmd(np.asarray(target_rel, dtype=float), params, gate)
    return np.array([wheels[0], wheels[1]], dtype=np.float64), params, gate


def synthetic_base_samples(n_park, n_shoulder, n_drive, base_z, rng):
    """Labels from the closed-form base law, which does not look at the arm.

    Logged trajectories barely touch the exact-zero deadzone (the yaw command
    falls to zero across 0.05 rad). These rows put that deadzone, the shoulder
    just outside it, and ordinary driving into the base fit in known proportions.
    """
    gate = ReachGateParams()
    # Zero when |dist_err| <= pos_inner - pos_brake_edge and |bearing| <= ang_inner - ang_brake_edge.
    dist_zero = gate.pos_inner - gate.pos_brake_edge
    bearing_zero = gate.ang_inner - gate.ang_brake_edge

    def pack(dist_err, bearing, world_z):
        standoff = standoff_for_height(world_z)
        rho = standoff + dist_err
        if rho <= 0.02:
            return None
        target_rel = np.array([
            rho * np.cos(bearing),
            rho * np.sin(bearing),
            world_z - base_z,
        ], dtype=np.float64)
        wheels, params, _ = wheel_pair_from_expert(target_rel, standoff, gate)
        # Reject a draw whose recovered errors do not match the request. That is a geometry bug.
        got_dist, got_bearing = base_errors(target_rel, params)
        if abs(got_dist - dist_err) > 1e-6 or abs(got_bearing - bearing) > 1e-6:
            raise RuntimeError("base feature round-trip failed")
        return base_feature_matrix(target_rel, standoff), wheels

    features = []
    labels = []

    def accept(row):
        if row is None:
            return
        feature, wheels = row
        features.append(feature)
        labels.append(wheels)

    made = 0
    while made < n_park:
        row = pack(
            float(rng.uniform(-dist_zero, dist_zero)),
            float(rng.uniform(-bearing_zero, bearing_zero)),
            float(rng.uniform(0.10, 1.40)),
        )
        if row is None:
            continue
        if np.max(np.abs(row[1])) > 1e-8:
            raise RuntimeError("A sample inside the deadzone did not get zero wheel speed")
        accept(row)
        made += 1

    made = 0
    while made < n_shoulder:
        # One axis in the fade, the other still inside its zero, so exactly one brake is active.
        if float(rng.random()) < 0.5:
            dist = float(rng.uniform(dist_zero, gate.pos_inner + 0.02)) * float(rng.choice([-1.0, 1.0]))
            bearing = float(rng.uniform(-bearing_zero * 0.5, bearing_zero * 0.5))
        else:
            dist = float(rng.uniform(-dist_zero * 0.5, dist_zero * 0.5))
            bearing = float(rng.uniform(bearing_zero, gate.ang_inner + 0.02)) * float(rng.choice([-1.0, 1.0]))
        row = pack(dist, bearing, float(rng.uniform(0.10, 1.40)))
        if row is None:
            continue
        accept(row)
        made += 1

    made = 0
    while made < n_drive:
        dist = float(rng.uniform(-0.25, 1.6))
        bearing = float(rng.uniform(-np.pi, np.pi))
        if abs(dist) <= dist_zero and abs(bearing) <= bearing_zero:
            continue
        row = pack(dist, bearing, float(rng.uniform(0.10, 1.40)))
        if row is None:
            continue
        accept(row)
        made += 1

    return np.vstack(features), np.vstack(labels)


def initial_state(rng, q_hold, lower, upper, canonical_prob=0.25):
    """Default pose with probability ``canonical_prob``, otherwise a perturbed start.

    The perturbation is the generalisation the brief asks for. The canonical pose
    stays in the set because the frozen test always starts there. Returns None
    for a canonical start.
    """
    if float(rng.random()) < canonical_prob:
        return None
    q = np.asarray(q_hold, dtype=float).copy()
    q = q + rng.uniform(-0.45, 0.45, size=7)
    limited = np.asarray(lower) < np.asarray(upper)
    margin = 0.12
    q[limited] = np.clip(q[limited], lower[limited] + margin, upper[limited] - margin)
    return {
        "dx": float(rng.uniform(-0.30, 0.30)),
        "dy": float(rng.uniform(-0.30, 0.30)),
        "yaw": float(rng.uniform(-np.pi, np.pi)),
        "q_arm": q.astype(np.float64),
    }


def apply_initial_state(ctx, initial):
    """Reset to the default pose, then apply ``initial`` when it is not None.

    The default path is the same three calls as ``robotnik_arm_task.run_episode``,
    so the frozen grid starts from the pose the phased expert was scored on.
    """
    sim = ctx["sim"]
    client = ctx["client"]
    rid = ctx["rid"]
    sim.ResetPose()
    client.resetBaseVelocity(rid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    if initial is not None:
        base = np.array(sim.GetBasePosition(), dtype=float)
        quat = client.getQuaternionFromEuler([0.0, 0.0, float(initial["yaw"])])
        client.resetBasePositionAndOrientation(
            rid,
            [base[0] + float(initial["dx"]), base[1] + float(initial["dy"]), base[2]],
            quat,
        )
        client.resetBaseVelocity(rid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        bot = sim.bot[0]
        for index in range(7):
            name = f"robot_j2s7s300_joint_{index + 1}"
            joint_id = bot.joint_name_to_id[name]
            client.resetJointState(rid, joint_id, float(initial["q_arm"][index]), targetVelocity=0.0)
    sim.ReceiveObservation()


def check():
    """Sampler, encoding, spin filter, and the deadzone of the base law. No simulator."""
    continuous = np.zeros(7, dtype=bool)
    continuous[list(CONTINUOUS_JOINTS)] = True
    names = arm_feature_names(continuous)
    q = np.zeros(7)
    q[0] = -33.0
    q[1] = 0.4
    encoded = encode_q(q, continuous)
    if encoded.shape != (11,):
        raise AssertionError(f"expected 11 angle features, got {encoded.shape}")
    if abs(encoded[0] - np.sin(-33.0)) > 1e-12 or abs(encoded[1] - np.cos(-33.0)) > 1e-12:
        raise AssertionError("joint 1 was not encoded as sin/cos")
    if abs(encoded[2] - 0.4) > 1e-12:
        raise AssertionError("joint 2 should stay in radians")
    if len(names) != 3 + 2 + 11 + 7 + 3 + 3 + 1 + 7:
        raise AssertionError(f"arm feature width {len(names)}")
    row = arm_features_row(
        [0.2, -0.1, 0.4], [0.0, 0.01], q, np.arange(7) * 0.01,
        [0.0, 0.0, 0.0], [0.01, 0.0, -0.02], 1.0, np.ones(7), continuous,
    )
    if row.shape != (len(names),) or not np.isfinite(row).all():
        raise AssertionError("arm feature row has the wrong shape")
    base = base_feature_matrix([0.65, 0.0, 0.2], 0.65)
    if base.shape != (8,) or abs(base[4]) > 1e-9 or abs(base[6]) > 1e-9:
        raise AssertionError("dist_err should be zero on the standoff ring, straight ahead")

    q_log = np.zeros((30, 7))
    qd_log = np.zeros((30, 7))
    q_log[20:, 0] = -7.0
    qd_log[18:, 0] = 10.0
    kept = keep_mask(q_log, qd_log, dt=0.05)
    # First bad speed is row 18. Margin 0.25/0.05 = 5 rows, so rows 13.. are dropped.
    if kept.sum() != 13:
        raise AssertionError(f"spin filter kept {kept.sum()} rows, expected 13")

    rng = np.random.default_rng(0)
    grid = np.array([[1.2, 0.0, 0.7], [-1.2, -1.2, 0.25]])
    draw = sample_targets(30, [0.0, 0.0], rng, avoid=grid, beyond_frac=0.0)
    if np.any(np.hypot(draw[:, 0], draw[:, 1]) < MIN_RADIUS - 1e-9):
        raise AssertionError("a target landed in the footprint")
    if np.any(draw[:, 2] < Z_LO) or np.any(draw[:, 2] > Z_HI):
        raise AssertionError("a non-beyond draw left the band")
    if _too_close(grid[0], draw, MIN_SEPARATION):
        raise AssertionError("a draw landed on a frozen target")
    again = sample_targets(30, [0.0, 0.0], np.random.default_rng(0), avoid=grid, beyond_frac=0.0)
    if not np.allclose(draw, again):
        raise AssertionError("target draw is not determined by the seed")
    mixed = sample_targets(80, [0.0, 0.0], np.random.default_rng(1), beyond_frac=BEYOND_FRAC)
    frac = float(np.mean(beyond_band(mixed[:, 2])))
    if not 0.02 <= frac <= 0.30:
        raise AssertionError(f"beyond fraction {frac:.2f} is outside the expected range")

    features, labels = synthetic_base_samples(40, 20, 40, base_z=0.08, rng=np.random.default_rng(2))
    if features.shape[1] != 8 or labels.shape[1] != 2:
        raise AssertionError("synthetic base arrays have the wrong width")
    dead = np.max(np.abs(labels[:40]), axis=1)
    if np.any(dead > 1e-8):
        raise AssertionError("deadzone synthetic labels are not exact zeros")
    import robotnik_arm_task as task
    if list(task.STANDOFF_Z_POINTS) != STANDOFF_Z_POINTS or list(task.STANDOFF_POINTS) != STANDOFF_POINTS:
        raise AssertionError("standoff table does not match robotnik_arm_task")
    print(
        f"features_v2 ok ({len(names)} arm inputs, {len(base_feature_names())} base inputs, "
        f"deadzone exact on {len(dead)} synthetic rows)",
        flush=True,
    )


if __name__ == "__main__":
    check()
