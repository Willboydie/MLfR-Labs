"""Input features for behaviour cloning the robotnik reaching expert.

Instantaneous features are expressed in the base frame. The base-goal pose uses
the same geometry as ``robotnik_arm_task._base_goal``. A short history is four
samples, 25 ms apart, covering the last 100 ms. Each history entry is the
change from that past sample to now (a velocity proxy): base translation in the
current base frame, wrapped bearing change, and the change in ``q_arm``.
Samples that would fall before the reset are the initial state, so the first
100 ms measures the change since the start rather than a zero pad.

Run ``python bc_features.py`` to check the feature algebra without a rollout.
"""
import os

os.environ.setdefault("ROBOENV_HEADLESS", "1")

import numpy as np
from simulation_and_control import wrap_angle

import robotnik_arm_task as task

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "bc_data")
MODEL_DIR = os.path.join(TASK_DIR, "bc_model")

N_TARGETS = 100
N_TRAIN_TRAJECTORIES = 80
N_VAL_TRAJECTORIES = 20
SEED = 0

ARM_DOF = 7
HISTORY_LAGS_S = (0.025, 0.05, 0.075, 0.1)
# xyz change, wrapped bearing change, q_arm change
HISTORY_DIM = 3 + 1 + ARM_DOF

OUTPUT_NAMES = ["wheel_right", "wheel_left"] + [f"torque_j{i}" for i in range(1, ARM_DOF + 1)]
PHASE_ORDER = list(task.PHASE_ORDER)
# "done" is the logged name of the arm-reaching phase: the base controller has
# finished and no longer changes its phase variable.
PHASE_LABELS = {"align": "align", "drive": "drive", "orient": "orient", "done": "done (reach)"}


def _feature_names():
    names = [
        "target_x_base", "target_y_base", "target_z_base",
        "goal_x_base", "goal_y_base",
        "dist_to_goal",
        "heading_error",
        "bearing_error",
        "sin_base_bearing", "cos_base_bearing",
        "dist_from_start",
    ]
    names += [f"q{i}" for i in range(1, ARM_DOF + 1)]
    names += [f"qd{i}" for i in range(1, ARM_DOF + 1)]
    names += [f"q{i}_minus_init" for i in range(1, ARM_DOF + 1)]
    names.append("time")
    instant = len(names)
    for lag_s in HISTORY_LAGS_S:
        ms = int(round(lag_s * 1000))
        names += [
            f"d_base_x_{ms}ms", f"d_base_y_{ms}ms", f"d_base_z_{ms}ms", f"d_bearing_{ms}ms",
        ]
        names += [f"dq{i}_{ms}ms" for i in range(1, ARM_DOF + 1)]
    return names, instant


FEATURE_NAMES, INSTANT_DIM = _feature_names()


def history_steps(dt):
    """Step lags for the four history samples. 25 ms must divide the time step."""
    steps = []
    for lag_s in HISTORY_LAGS_S:
        count = lag_s / dt
        if abs(count - round(count)) > 1e-6:
            raise ValueError(f"{lag_s * 1000:.0f} ms is not an integer number of steps at dt={dt}")
        steps.append(int(round(count)))
    return tuple(steps)


def episode_path(data_dir, index):
    return os.path.join(data_dir, f"episode_{index:03d}.npz")


def meta_path(data_dir):
    return os.path.join(data_dir, "dataset_meta.npz")


def grid_span():
    """Axis-aligned bounds of the assessment grid, in metres."""
    lo = np.array([task.GRID_X.min(), task.GRID_Y.min(), task.GRID_Z.min()], dtype=float)
    hi = np.array([task.GRID_X.max(), task.GRID_Y.max(), task.GRID_Z.max()], dtype=float)
    return lo, hi


def jitter_half_widths():
    """Half the grid spacing on each axis, so jittered cells cover the span."""
    widths = []
    for values in (task.GRID_X, task.GRID_Y, task.GRID_Z):
        step = np.diff(np.sort(np.unique(np.asarray(values, dtype=float))))
        widths.append(0.5 * float(np.min(step)))
    return np.asarray(widths, dtype=float)


def sample_bc_targets(n=N_TARGETS, seed=SEED):
    """``n`` targets, each a sparse-grid point plus uniform half-cell jitter.

    Every assessment-grid point is an anchor (so unreachable grid cells are
    included), then the draw is shuffled. Jitter that leaves the grid's x/y/z
    span, or that lands within ``GRID_MIN_RADIUS`` of the robot, is redrawn.
    """
    grid = np.stack([np.asarray(point, dtype=float) for point in task.sparse_target_grid()])
    if n < len(grid):
        raise ValueError(f"Need at least {len(grid)} targets to cover every grid point")
    lo, hi = grid_span()
    half = jitter_half_widths()
    rng = np.random.default_rng(seed)
    repeats = int(np.ceil(n / len(grid)))
    anchors = np.tile(np.arange(len(grid)), repeats)[:n]
    anchors = anchors[rng.permutation(n)]

    targets = np.empty((n, 3), dtype=float)
    for i, anchor in enumerate(anchors):
        base = grid[int(anchor)]
        placed = False
        for _ in range(10000):
            candidate = base + rng.uniform(-half, half)
            if np.any(candidate < lo) or np.any(candidate > hi):
                continue
            if np.hypot(candidate[0], candidate[1]) < task.GRID_MIN_RADIUS:
                continue
            targets[i] = candidate
            placed = True
            break
        if not placed:
            raise RuntimeError(f"Could not place a target around grid point {base}")
    return targets, anchors.astype(np.int64), grid


def yaw_to_base(dx, dy, bearing):
    """Rotate a world-frame xy displacement into the yaw-only base frame."""
    c = np.cos(bearing)
    s = np.sin(bearing)
    return c * dx + s * dy, -s * dx + c * dy


def _as_columns(parts):
    shaped = []
    for part in parts:
        arr = np.asarray(part, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        shaped.append(arr)
    return np.hstack(shaped)


def _instant_block(pos, bearing, q, qd, time, target, goal_xy, goal_bearing, q_init, start):
    """Rows are samples. Target and base-goal position are in the base frame."""
    dx = target[0] - pos[:, 0]
    dy = target[1] - pos[:, 1]
    tx, ty = yaw_to_base(dx, dy, bearing)
    gdx = goal_xy[0] - pos[:, 0]
    gdy = goal_xy[1] - pos[:, 1]
    gx, gy = yaw_to_base(gdx, gdy, bearing)
    heading_err = wrap_angle(np.arctan2(gdy, gdx) - bearing)
    bearing_err = wrap_angle(goal_bearing - bearing)
    dist_start = np.hypot(pos[:, 0] - start[0], pos[:, 1] - start[1])
    return _as_columns([
        tx, ty, target[2] - pos[:, 2],
        gx, gy,
        np.hypot(gdx, gdy),
        heading_err,
        bearing_err,
        np.sin(bearing), np.cos(bearing),
        dist_start,
        q, qd, q - np.asarray(q_init, dtype=np.float64).reshape(1, -1),
        time,
    ])


def _delta_arrays(pos_now, bearing_now, q_now, pos_then, bearing_then, q_then):
    """Change from a past sample to now, with xy in the current base frame."""
    hx, hy = yaw_to_base(pos_now[:, 0] - pos_then[:, 0], pos_now[:, 1] - pos_then[:, 1], bearing_now)
    return _as_columns([
        hx, hy, pos_now[:, 2] - pos_then[:, 2],
        wrap_angle(bearing_now - bearing_then),
        q_now - q_then,
    ])


def _history_block(pos, bearing, q, lag):
    src = np.maximum(0, np.arange(pos.shape[0]) - lag)
    return _delta_arrays(pos, bearing, q, pos[src], bearing[src], q[src])


def episode_feature_matrix(time, base_pos, base_bearing, q_arm, qd_arm, target, q_init, start_base_pos, dt):
    """Feature matrix for one episode, shape ``(T, len(FEATURE_NAMES))``."""
    pos = np.asarray(base_pos, dtype=np.float64)
    bearing = np.asarray(base_bearing, dtype=np.float64)
    q = np.asarray(q_arm, dtype=np.float64)
    qd = np.asarray(qd_arm, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    q_init = np.asarray(q_init, dtype=np.float64).reshape(ARM_DOF)
    start = np.asarray(start_base_pos, dtype=np.float64).reshape(3)
    goal_xy, goal_bearing = task._base_goal(target)
    instant = _instant_block(
        pos, bearing, q, qd, time, target,
        np.asarray(goal_xy, dtype=np.float64), float(goal_bearing), q_init, start,
    )
    history = [_history_block(pos, bearing, q, lag) for lag in history_steps(dt)]
    matrix = np.hstack([instant, *history])
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"{matrix.shape[1]} feature columns but {len(FEATURE_NAMES)} names")
    return matrix


class FeatureBuffer:
    """Online features. ``push`` matches one row of ``episode_feature_matrix``."""

    def __init__(self, dt, q_init, start_base_pos, target):
        self.lags = history_steps(dt)
        self.q_init = np.asarray(q_init, dtype=np.float64).reshape(ARM_DOF)
        self.start = np.asarray(start_base_pos, dtype=np.float64).reshape(3)
        self.target = np.asarray(target, dtype=np.float64).reshape(3)
        goal_xy, goal_bearing = task._base_goal(self.target)
        self.goal_xy = np.asarray(goal_xy, dtype=np.float64)
        self.goal_bearing = float(goal_bearing)
        self.pos = []
        self.bearing = []
        self.q = []

    def push(self, time, base_pos, base_bearing, q_arm, qd_arm):
        pos = np.asarray(base_pos, dtype=np.float64).reshape(3).copy()
        q = np.asarray(q_arm, dtype=np.float64).reshape(ARM_DOF).copy()
        qd = np.asarray(qd_arm, dtype=np.float64).reshape(ARM_DOF)
        bearing = float(base_bearing)
        self.pos.append(pos)
        self.bearing.append(bearing)
        self.q.append(q)
        i = len(self.pos) - 1
        instant = _instant_block(
            pos.reshape(1, 3), np.array([bearing]), q.reshape(1, ARM_DOF), qd.reshape(1, ARM_DOF),
            np.array([float(time)]), self.target, self.goal_xy, self.goal_bearing, self.q_init, self.start,
        )
        deltas = []
        for lag in self.lags:
            j = 0 if i < lag else i - lag
            deltas.append(_delta_arrays(
                pos.reshape(1, 3), np.array([bearing]), q.reshape(1, ARM_DOF),
                self.pos[j].reshape(1, 3), np.array([self.bearing[j]]), self.q[j].reshape(1, ARM_DOF),
            ))
        return np.hstack([instant, *deltas])[0]


def action_labels(wheel_cmds, torque_arm):
    """Two wheel labels plus seven torques.

    The expert always commands ``[right, left, left, right]``. The two distinct
    values are the regression targets; expand them again at run time.
    """
    wheels = np.asarray(wheel_cmds, dtype=np.float64)
    torque = np.asarray(torque_arm, dtype=np.float64)
    if wheels.ndim != 2 or wheels.shape[1] != 4:
        raise ValueError(f"Expected wheel commands (T, 4), got {wheels.shape}")
    right, left = wheels[:, 0], wheels[:, 1]
    if not (np.allclose(wheels[:, 2], left) and np.allclose(wheels[:, 3], right)):
        raise ValueError("Expert wheel command is not [right, left, left, right]")
    return np.column_stack([right, left, torque])


def expand_wheel_commands(right, left):
    """Copy the two wheel outputs back to the four-wheel layout the plant expects."""
    return np.array([right, left, left, right], dtype=np.float64)


def _self_test():
    names, instant = _feature_names()
    assert instant == INSTANT_DIM
    assert (len(names) - instant) == len(HISTORY_LAGS_S) * HISTORY_DIM
    assert history_steps(0.001) == (25, 50, 75, 100)
    hx, hy = yaw_to_base(1.0, 0.0, np.pi / 2)
    assert abs(hx) < 1e-12 and abs(hy + 1.0) < 1e-12

    lo, hi = grid_span()
    targets, anchors, grid = sample_bc_targets(N_TARGETS, seed=SEED)
    assert targets.shape == (N_TARGETS, 3)
    assert np.all(targets >= lo - 1e-9) and np.all(targets <= hi + 1e-9)
    assert np.all(np.hypot(targets[:, 0], targets[:, 1]) >= task.GRID_MIN_RADIUS - 1e-9)
    counts = np.bincount(anchors, minlength=len(grid))
    assert np.all(counts >= N_TARGETS // len(grid))
    again, _, _ = sample_bc_targets(N_TARGETS, seed=SEED)
    assert np.allclose(targets, again)

    rng = np.random.default_rng(1)
    dt = 0.001
    n = 180
    time = np.arange(n) * dt
    bearing = np.cumsum(rng.normal(0, 0.02, size=n))
    bearing[75] = 3.1
    bearing[100] = -3.1
    pos = np.cumsum(rng.normal(0, 0.002, size=(n, 3)), axis=0)
    pos[:, 2] += 0.08
    q = np.cumsum(rng.normal(0, 0.01, size=(n, ARM_DOF)), axis=0)
    qd = rng.normal(0, 0.1, size=(n, ARM_DOF))
    target = np.array([1.0, -0.5, 0.8])
    q_init = rng.normal(0, 0.2, size=ARM_DOF)
    start = pos[0].copy()
    batch = episode_feature_matrix(time, pos, bearing, q, qd, target, q_init, start, dt)
    buffer = FeatureBuffer(dt, q_init, start, target)
    online = np.vstack([
        buffer.push(time[i], pos[i], bearing[i], q[i], qd[i]) for i in range(n)
    ])
    assert batch.shape == online.shape == (n, len(FEATURE_NAMES))
    assert np.allclose(batch, online, rtol=0, atol=1e-9)
    assert np.allclose(batch[0, INSTANT_DIM:], 0.0)

    # Before 100 ms every lookback is the initial state, so the four blocks match.
    first_block = batch[10, INSTANT_DIM:INSTANT_DIM + HISTORY_DIM]
    for k in range(1, len(HISTORY_LAGS_S)):
        block = batch[10, INSTANT_DIM + k * HISTORY_DIM:INSTANT_DIM + (k + 1) * HISTORY_DIM]
        assert np.allclose(block, first_block)
    expected_from_start = _delta_arrays(
        pos[10:11], bearing[10:11], q[10:11], pos[0:1], bearing[0:1], q[0:1]
    )[0]
    assert np.allclose(first_block, expected_from_start)

    # At 100 ms the oldest sample is the reset and the 25 ms sample is index 75.
    oldest = batch[100, INSTANT_DIM + 3 * HISTORY_DIM:INSTANT_DIM + 4 * HISTORY_DIM]
    from_reset = _delta_arrays(pos[100:101], bearing[100:101], q[100:101], pos[0:1], bearing[0:1], q[0:1])[0]
    assert np.allclose(oldest, from_reset)
    newest = batch[100, INSTANT_DIM:INSTANT_DIM + HISTORY_DIM]
    from_25 = _delta_arrays(pos[100:101], bearing[100:101], q[100:101], pos[75:76], bearing[75:76], q[75:76])[0]
    assert np.allclose(newest, from_25)
    wrapped = wrap_angle(bearing[100] - bearing[75])
    assert abs(wrapped) < np.pi
    assert abs(newest[3] - wrapped) < 1e-9
    assert abs(newest[3]) < 1.0  # the branch cut, not the raw -6.2 rad jump

    heading_col = FEATURE_NAMES.index("heading_error")
    bearing_col = FEATURE_NAMES.index("bearing_error")
    assert np.all(np.abs(batch[:, heading_col]) <= np.pi + 1e-9)
    assert np.all(np.abs(batch[:, bearing_col]) <= np.pi + 1e-9)

    wheels = np.array([[1.0, -0.5, -0.5, 1.0], [0.2, 0.2, 0.2, 0.2]])
    torque = np.ones((2, ARM_DOF))
    labels = action_labels(wheels, torque)
    assert labels.shape == (2, 9)
    assert np.allclose(expand_wheel_commands(labels[0, 0], labels[0, 1]), wheels[0])
    print(f"bc_features ok ({len(FEATURE_NAMES)} features, {len(grid)} grid anchors)")


if __name__ == "__main__":
    _self_test()
