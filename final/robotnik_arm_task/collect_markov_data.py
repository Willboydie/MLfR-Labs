"""Log Markov-expert demonstrations for a later behaviour-cloning run.

One episode is one target. The robot always starts at the default reset pose.
Targets are drawn independently and uniformly from a 2 m × 2 m × 2 m box.
Nothing is rejected or redrawn.

* x and y are centered on the base position at the reset pose
* z starts at the shoulder height of that pose (pinocchio frame
  ``robot_j2s7s300_link_2``, the frame the expert treats as the shoulder)
  and extends 2 m upward

Of the 100 episodes, 85 are training and 15 are validation. The validation
episodes are a random subset of the same box, stored in ``dataset_meta.npz``.

Each logged row is the expert call on that state. The model input is
``observation_vector``: heading-frame target displacement, base roll and pitch,
arm position and velocity, and the cartesian integrator ``z``. ``z`` is the
value passed into the expert, before ``integrator_update``. The label is the
clean command: the two distinct wheel speeds ``[right, left]`` and the seven
arm torques.

DART adds Gaussian noise to the command that is sent to the simulator and
keeps the clean command as the label. Wheel noise and torque noise have
separate standard deviations, both defaulting to 0, so the applied command
is the expert command. Noise is drawn per episode from a seed that depends
on the episode index, which keeps a sharded run identical to a full run.
The noisy command is clipped to the expert's own limits: arm torque to
``±max_torque``, and each wheel speed to the largest speed the base twist
limits can produce.

Every episode is saved, including ones that stall or hit the step cap. The
file records the outcome: ``reached``, ``final_error``, ``final_ee_pos``,
``final_time``, and ``stop_message``. The last logged row is the state that
produced that outcome. Re-running skips episodes already on disk.

``--near`` appends 50 further episodes, numbered after the original 100.
Those targets are uniform in the square where ``|x|`` and ``|y|`` are within
0.5 m of the reset base and the world-frame height is in ``[0.25, 0.75)`` m.
They are extra training episodes (``near_index`` in the metadata). The
original 85/15 split is left unchanged. Nothing in that square is rejected.

``--check`` tests the box sampler, the split, and the noise clip without
starting a rollout.

From the RoboEnv checkout::

    pixi run python ../final/robotnik_arm_task/collect_markov_data.py --check
    pixi run python ../final/robotnik_arm_task/collect_markov_data.py
    pixi run python ../final/robotnik_arm_task/collect_markov_data.py --near
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

from robotnik_arm_control import quaternion2bearing
from robotnik_markov_expert import (
    BaseControllerParams,
    StopCriteria,
    check_stop,
    integrator_update,
    observation_vector,
    relative_target,
)
import robotnik_arm_task as task


N_EPISODES = 100
N_TRAIN = 85
N_VAL = 15
SEED = 0

# World-frame box, metres. XY is centered on the reset base. Z = 0 of the box
# is the shoulder height; the interval extends upward.
BOX_XY = 2.0
BOX_Z = 2.0

# Extra episodes. XY is centered on the reset base; Z is world-frame height.
NEAR_EPISODES = 50
NEAR_XY = 0.5
NEAR_Z_LO = 0.25
NEAR_Z_HI = 0.75

WHEEL_NOISE_STD = 0.0
TORQUE_NOISE_STD = 0.0

ARM_DOF = 7
# Column order of robotnik_markov_expert.observation_vector.
OBSERVATION_NAMES = (
    ["target_rel_x", "target_rel_y", "target_rel_z", "roll", "pitch"]
    + [f"q{i}" for i in range(1, ARM_DOF + 1)]
    + [f"qd{i}" for i in range(1, ARM_DOF + 1)]
    + ["z_x", "z_y", "z_z"]
)
TARGET_REL_SLICE = slice(0, 3)
ROLL_PITCH_SLICE = slice(3, 5)
Q_SLICE = slice(5, 12)
QD_SLICE = slice(12, 19)
Z_SLICE = slice(19, 22)
LABEL_NAMES = ["wheel_right", "wheel_left"] + [f"torque_j{i}" for i in range(1, ARM_DOF + 1)]

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "markov_data")
SHOULDER_FRAME = "robot_j2s7s300_link_2"


def parse_args():
    parser = argparse.ArgumentParser(description="Log Markov-expert reaching demonstrations.")
    parser.add_argument("--out", default=DATA_DIR, help="Directory for episode npz files.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wheel-noise-std", type=float, default=WHEEL_NOISE_STD,
                        help="DART std on each of the two wheel speeds (rad/s).")
    parser.add_argument("--torque-noise-std", type=float, default=TORQUE_NOISE_STD,
                        help="DART std on each arm torque (N*m).")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many new episodes (0 = all missing).")
    parser.add_argument("--shard", default="", help="Run every n-th episode, as i/n. The target list is shared.")
    parser.add_argument("--gui", action="store_true", help="Open the PyBullet GUI.")
    parser.add_argument("--check", action="store_true", help="Check the sampler and the noise clip, then exit.")
    parser.add_argument(
        "--near",
        action="store_true",
        help="Append 50 episodes with |x| and |y| < 0.5 m about the reset base and 0.25 m <= z < 0.75 m.",
    )
    return parser.parse_args()


def episode_path(data_dir, index):
    return os.path.join(data_dir, f"episode_{index:03d}.npz")


def meta_path(data_dir):
    return os.path.join(data_dir, "dataset_meta.npz")


def episode_indices(n, shard):
    indices = list(range(n))
    if not shard:
        return indices
    index, count = (int(part) for part in shard.split("/"))
    if count < 1 or not 0 <= index < count:
        raise SystemExit(f"Bad --shard {shard!r}; expected i/n with 0 <= i < n")
    return indices[index::count]


def box_bounds(base_pos, shoulder_z):
    """Axis-aligned world box. ``shoulder_z`` is the bottom face."""
    half_xy = BOX_XY / 2.0
    origin = np.asarray(base_pos, dtype=float)
    lo = np.array([origin[0] - half_xy, origin[1] - half_xy, float(shoulder_z)])
    hi = np.array([origin[0] + half_xy, origin[1] + half_xy, float(shoulder_z) + BOX_Z])
    return lo, hi


def near_bounds(base_pos):
    """Square about the reset base, and a fixed world-frame height band.

    ``|x|`` and ``|y|`` stay under ``NEAR_XY``. ``z`` runs from ``NEAR_Z_LO``
    up to, but not including, ``NEAR_Z_HI``.
    """
    origin = np.asarray(base_pos, dtype=float)
    lo = np.array([origin[0] - NEAR_XY, origin[1] - NEAR_XY, NEAR_Z_LO])
    hi = np.array([origin[0] + NEAR_XY, origin[1] + NEAR_XY, NEAR_Z_HI])
    return lo, hi


def sample_targets(n, lo, hi, rng):
    """``n`` independent uniform draws in ``[lo, hi)``. No rejection."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return rng.uniform(lo, hi, size=(n, 3))


def train_val_split(n, rng):
    """Random 85/15 subset. Returned indices are sorted; membership is the random part."""
    if n != N_TRAIN + N_VAL:
        raise ValueError(f"Expected {N_TRAIN + N_VAL} episodes, got {n}")
    perm = rng.permutation(n)
    train = np.sort(perm[:N_TRAIN]).astype(np.int64)
    val = np.sort(perm[N_TRAIN:]).astype(np.int64)
    return train, val


def wheel_speed_limit(params):
    """Largest |wheel speed| the saturated base twist can command, in rad/s.

    ``w_cap`` falls linearly as forward speed rises, and the wheel map is linear
    in ``(v, w)``, so the maximum is at one of the two endpoints: spinning in
    place, or driving at full speed with the reduced yaw cap.
    """
    half = params.wheel_base_width / 2.0
    spin = params.max_ang_vel * half
    drive = params.max_lin_vel + params.max_ang_vel_driving * half
    return max(spin, drive) / params.wheel_radius


def apply_dart(wheel_cmd, torque_arm, rng, wheel_std, torque_std, wheel_limit, torque_limit):
    """Expert command plus independent Gaussian noise, then the expert clips.

    Noise is added to the two distinct wheel speeds and to the seven torques.
    The plant command is expanded back to ``[right, left, left, right]``.
    Both standard deviations at zero leave the expert command unchanged.
    """
    wheel_cmd = np.asarray(wheel_cmd, dtype=np.float64)
    torque = np.array(torque_arm, dtype=np.float64, copy=True)
    if wheel_std == 0.0 and torque_std == 0.0:
        return np.array(wheel_cmd, copy=True), torque
    right = float(wheel_cmd[0])
    left = float(wheel_cmd[1])
    if wheel_std > 0.0:
        right += float(rng.normal(0.0, wheel_std))
        left += float(rng.normal(0.0, wheel_std))
    if torque_std > 0.0:
        torque = torque + rng.normal(0.0, torque_std, size=torque.shape)
    right = float(np.clip(right, -wheel_limit, wheel_limit))
    left = float(np.clip(left, -wheel_limit, wheel_limit))
    torque = np.clip(torque, -torque_limit, torque_limit)
    applied = np.array([right, left, left, right], dtype=np.float64)
    return applied, torque


def clean_wheel_pair(wheel_cmd):
    """``[right, left]`` from the expert layout ``[right, left, left, right]``."""
    wheel_cmd = np.asarray(wheel_cmd, dtype=float)
    right, left = float(wheel_cmd[0]), float(wheel_cmd[1])
    if abs(wheel_cmd[2] - left) > 1e-9 or abs(wheel_cmd[3] - right) > 1e-9:
        raise RuntimeError("Markov wheel command is not [right, left, left, right]")
    return np.array([right, left], dtype=np.float64)


def assert_observation_layout(obs, target_rel, q_arm, qd_arm, z):
    if obs.shape != (len(OBSERVATION_NAMES),):
        raise RuntimeError(f"observation has {obs.shape[-1]} columns, expected {len(OBSERVATION_NAMES)}")
    if not (
        np.allclose(obs[TARGET_REL_SLICE], target_rel)
        and np.allclose(obs[Q_SLICE], q_arm)
        and np.allclose(obs[QD_SLICE], qd_arm)
        and np.allclose(obs[Z_SLICE], z)
    ):
        raise RuntimeError("observation_vector column order does not match OBSERVATION_NAMES")


def measure_reset(ctx):
    """Base position and shoulder position at the default pose, in the world frame."""
    sim = ctx["sim"]
    dyn = ctx["dyn_model"]
    sim.ResetPose()
    ctx["client"].resetBaseVelocity(ctx["rid"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sim.ReceiveObservation()
    base_pos = np.array(sim.GetBasePosition(), dtype=float)
    base_ori = np.array(sim.GetBaseOrientation(), dtype=float)
    q_mes = np.array(sim.GetMotorAngles(0), dtype=float)
    q_pin = dyn.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
    pin.forwardKinematics(dyn.pin_model, dyn.pin_data, q_pin)
    pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
    shoulder = np.array(dyn.pin_data.oMf[ctx["shoulder_frame_id"]].translation, dtype=float)
    return base_pos, shoulder


def _limits(ctx):
    """Wheel-speed clip, torque clip, and integrator clip. Standoff does not enter."""
    expert = task._markov_expert(ctx, np.zeros(3))
    return (
        float(wheel_speed_limit(expert.base_params)),
        float(expert.arm_params.max_torque),
        float(expert.arm_params.z_max),
    )


def _as_float(meta, key):
    return float(np.asarray(meta[key]))


def _check_meta(meta, path, seed, dt, lo, hi, wheel_std, torque_std, wheel_limit, torque_limit, z_max):
    problems = []
    if int(meta["seed"]) != seed or len(meta["targets"]) != N_EPISODES:
        problems.append(
            f"seed={int(meta['seed'])} and {len(meta['targets'])} targets; this run asked for seed={seed} and {N_EPISODES}"
        )
    if abs(_as_float(meta, "dt") - dt) > 1e-12:
        problems.append(f"simulation dt {dt} does not match saved dt {_as_float(meta, 'dt')}")
    if not (np.allclose(meta["box_lo"], lo) and np.allclose(meta["box_hi"], hi)):
        problems.append(f"reset box {lo} .. {hi} does not match saved {meta['box_lo']} .. {meta['box_hi']}")
    if abs(_as_float(meta, "wheel_noise_std") - wheel_std) > 1e-12 or abs(_as_float(meta, "torque_noise_std") - torque_std) > 1e-12:
        problems.append(
            f"noise std wheels={_as_float(meta, 'wheel_noise_std')} torques={_as_float(meta, 'torque_noise_std')}; "
            f"this run asked for {wheel_std} and {torque_std}. Use a new --out directory."
        )
    if (
        abs(_as_float(meta, "wheel_limit") - wheel_limit) > 1e-9
        or abs(_as_float(meta, "torque_limit") - torque_limit) > 1e-9
        or abs(_as_float(meta, "z_max") - z_max) > 1e-12
    ):
        problems.append("saved command limits do not match the current expert")
    train = np.asarray(meta["train_index"])
    val = np.asarray(meta["val_index"])
    if len(train) != N_TRAIN or len(val) != N_VAL or np.intersect1d(train, val).size or len(np.union1d(train, val)) != N_EPISODES:
        problems.append(f"split is {len(train)}/{len(val)} and does not partition the {N_EPISODES} episodes")
    if problems:
        raise SystemExit(f"{path} cannot be resumed:\n  " + "\n  ".join(problems))


def _report(spec, resumed):
    lo, hi = spec["box_lo"], spec["box_hi"]
    targets = spec["targets"]
    where = "Reloaded" if resumed else "Sampled"
    print(
        f"Shoulder height {spec['shoulder_z']:.4f} m (frame {SHOULDER_FRAME}), "
        f"reset base xy {np.round(spec['base_xy'], 4)}.",
        flush=True,
    )
    print(
        f"Box x [{lo[0]:.3f}, {hi[0]:.3f}]  y [{lo[1]:.3f}, {hi[1]:.3f}]  "
        f"z [{lo[2]:.3f}, {hi[2]:.3f}]  ({BOX_Z:.2f} m up from the shoulder).",
        flush=True,
    )
    print(
        f"{where} {len(targets)} targets (seed {int(spec['seed'])}). "
        f"x [{targets[:, 0].min():.3f}, {targets[:, 0].max():.3f}]  "
        f"y [{targets[:, 1].min():.3f}, {targets[:, 1].max():.3f}]  "
        f"z [{targets[:, 2].min():.3f}, {targets[:, 2].max():.3f}]. "
        f"Train {N_TRAIN}, val {N_VAL}.",
        flush=True,
    )
    print(
        f"DART noise std: wheels {spec['wheel_noise_std']} rad/s, "
        f"torques {spec['torque_noise_std']} N*m. "
        f"Clips: wheels ±{spec['wheel_limit']:.4f} rad/s, torques ±{spec['torque_limit']:.1f} N*m.",
        flush=True,
    )


def _write_meta(path, spec):
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez(handle, **spec)
    os.replace(tmp, path)


def ensure_dataset(ctx, data_dir, seed, wheel_std, torque_std):
    """Create the shared target list and split, or reload them so a resume cannot drift."""
    path = meta_path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    base_pos, shoulder = measure_reset(ctx)
    lo, hi = box_bounds(base_pos, shoulder[2])
    wheel_limit, torque_limit, z_max = _limits(ctx)
    if os.path.isfile(path):
        with np.load(path) as meta:
            _check_meta(meta, path, seed, ctx["time_step"], lo, hi, wheel_std, torque_std, wheel_limit, torque_limit, z_max)
            spec = {key: np.array(meta[key]) for key in meta.files}
        _report(spec, resumed=True)
        return spec

    rng = np.random.default_rng(seed)
    targets = sample_targets(N_EPISODES, lo, hi, rng)
    train_index, val_index = train_val_split(N_EPISODES, rng)
    spec = {
        "targets": targets,
        "train_index": train_index,
        "val_index": val_index,
        "seed": np.int64(seed),
        "dt": np.float64(ctx["time_step"]),
        "base_xy": np.array(base_pos[:2], dtype=np.float64),
        "shoulder_z": np.float64(shoulder[2]),
        "box_lo": lo,
        "box_hi": hi,
        "wheel_noise_std": np.float64(wheel_std),
        "torque_noise_std": np.float64(torque_std),
        "wheel_limit": np.float64(wheel_limit),
        "torque_limit": np.float64(torque_limit),
        "z_max": np.float64(z_max),
        "observation_names": np.array(OBSERVATION_NAMES),
        "label_names": np.array(LABEL_NAMES),
    }
    _write_meta(path, spec)
    _report(spec, resumed=False)
    return spec


def ensure_near_dataset(ctx, data_dir, seed, wheel_std, torque_std):
    """Append the close-target list, or reload it so a resume cannot redraw it.

    The original ``targets`` array and the 85/15 split stay as they are.
    ``near_index`` is the episode numbers of this batch, starting at 100.
    """
    path = meta_path(data_dir)
    if not os.path.isfile(path):
        raise SystemExit(f"No dataset at {path}. Collect the original {N_EPISODES} episodes first.")
    base_pos, shoulder = measure_reset(ctx)
    lo, hi = box_bounds(base_pos, shoulder[2])
    near_lo, near_hi = near_bounds(base_pos)
    wheel_limit, torque_limit, z_max = _limits(ctx)
    with np.load(path) as meta:
        _check_meta(meta, path, seed, ctx["time_step"], lo, hi, wheel_std, torque_std, wheel_limit, torque_limit, z_max)
        spec = {key: np.array(meta[key]) for key in meta.files}

    expected_index = np.arange(N_EPISODES, N_EPISODES + NEAR_EPISODES, dtype=np.int64)
    if "near_targets" in spec:
        targets = np.asarray(spec["near_targets"], dtype=float)
        index = np.asarray(spec["near_index"])
        if (
            targets.shape != (NEAR_EPISODES, 3)
            or np.any(targets < near_lo)
            or np.any(targets >= near_hi)
            or not np.array_equal(index, expected_index)
            or not (np.allclose(spec["near_lo"], near_lo) and np.allclose(spec["near_hi"], near_hi))
        ):
            raise SystemExit(
                f"{path} already has a near-target list that does not match "
                f"|x|,|y| < {NEAR_XY} m and z in [{NEAR_Z_LO}, {NEAR_Z_HI})."
            )
        resumed = True
    else:
        # A stream of its own, so this draw cannot depend on how the first 100 were sampled.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        targets = sample_targets(NEAR_EPISODES, near_lo, near_hi, rng)
        spec["near_targets"] = targets
        spec["near_index"] = expected_index
        spec["near_lo"] = near_lo
        spec["near_hi"] = near_hi
        _write_meta(path, spec)
        resumed = False

    where = "Reloaded" if resumed else "Sampled"
    print(
        f"{where} {NEAR_EPISODES} near targets as episodes "
        f"{int(expected_index[0])}..{int(expected_index[-1])} (seed {seed}).",
        flush=True,
    )
    print(
        f"Near box x [{near_lo[0]:.3f}, {near_hi[0]:.3f}]  "
        f"y [{near_lo[1]:.3f}, {near_hi[1]:.3f}]  "
        f"z [{near_lo[2]:.3f}, {near_hi[2]:.3f}].",
        flush=True,
    )
    print(
        f"x [{targets[:, 0].min():.3f}, {targets[:, 0].max():.3f}]  "
        f"y [{targets[:, 1].min():.3f}, {targets[:, 1].max():.3f}]  "
        f"z [{targets[:, 2].min():.3f}, {targets[:, 2].max():.3f}]. "
        f"All {NEAR_EPISODES} are extra training episodes; the original {N_TRAIN}/{N_VAL} split is unchanged.",
        flush=True,
    )
    return spec


def log_episode(ctx, target_ee_pos, noise_rng, wheel_std, torque_std, wheel_limit, torque_limit, z_max):
    """One Markov episode. The logged ``z`` is the input of the command on that row.

    The last logged row is the state on which the episode stopped. The integrator
    is updated from the expert's cartesian error and reach weight. Noise is
    applied only to the command the simulator receives.
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

    expert = task._markov_expert(ctx, target_ee_pos)
    if abs(wheel_speed_limit(expert.base_params) - wheel_limit) > 1e-9:
        raise RuntimeError("This target's wheel-speed limit does not match the dataset clip")
    if abs(float(expert.arm_params.max_torque) - torque_limit) > 1e-9 or abs(float(expert.arm_params.z_max) - z_max) > 1e-12:
        raise RuntimeError("This target's torque or integrator limit does not match the dataset")
    stop_criteria = StopCriteria(reach_tolerance=task.REACH_TOLERANCE, max_steps=task.max_steps)
    cmd_plant = MotorCommands()
    # z integrates the end-effector error outside the expert. It starts at rest.
    z = np.zeros(3)

    time_all = []
    observation_all = []
    wheel_cmd_all = []
    torque_arm_all = []
    wheel_applied_all = []
    torque_applied_all = []
    cartesian_error_all = []
    reach_weight_all = []
    ee_dist_all = []

    current_time = 0.0
    ee_pos = None
    stop_message = None
    user_quit = False
    checked_layout = False

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
        ee_dist = float(np.linalg.norm(target_ee_pos - ee_pos))

        # The command uses this z. The update below is the input of the next step.
        z_in = np.array(z, dtype=float, copy=True)
        target_rel = relative_target(target_ee_pos, base_pos, base_bearing)
        cmd = expert(q_pin, q_arm, qd_arm, target_rel, z_in)
        obs = np.asarray(observation_vector(target_rel, q_pin, q_arm, qd_arm, z_in), dtype=np.float64)
        if not checked_layout:
            assert_observation_layout(obs, target_rel, q_arm, qd_arm, z_in)
            checked_layout = True
        wheel_label = clean_wheel_pair(cmd.wheel_cmd)
        torque_label = np.array(cmd.torque_arm, dtype=np.float64, copy=True)
        wheel_applied, torque_applied = apply_dart(
            cmd.wheel_cmd, cmd.torque_arm, noise_rng, wheel_std, torque_std, wheel_limit, torque_limit,
        )
        z = integrator_update(z_in, cmd.cartesian_error, cmd.reach_weight, time_step, z_max)

        cmd_plant.SetControlCmd(
            np.concatenate((wheel_applied, torque_applied, np.zeros(6))),
            ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6,
        )
        sim.Step(cmd_plant, "torque")

        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                dyn_model.DisplayModel(sim.GetMotorAngles(index))

        keys = sim.GetPyBulletClient().getKeyboardEvents()
        q_key = ord("q")
        if q_key in keys and keys[q_key] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            user_quit = True
            stop_message = "Stopped by user"
        else:
            arm_engaged = cmd.reach_weight >= 0.999
            stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist, arm_engaged)

        time_all.append(current_time)
        observation_all.append(obs)
        wheel_cmd_all.append(wheel_label)
        torque_arm_all.append(torque_label)
        wheel_applied_all.append(wheel_applied)
        torque_applied_all.append(np.array(torque_applied, dtype=np.float64, copy=True))
        cartesian_error_all.append(np.array(cmd.cartesian_error, dtype=np.float64, copy=True))
        reach_weight_all.append(float(cmd.reach_weight))
        ee_dist_all.append(ee_dist)

        if stop_message:
            break

        if int(current_time / 5.0) != int((current_time - time_step) / 5.0) and current_time >= 5.0:
            print(
                f"    t={current_time:.0f}s  ee {ee_dist:.3f} m  reach_w {cmd.reach_weight:.2f}",
                flush=True,
            )
        current_time += time_step

    if not time_all:
        raise RuntimeError("Episode ended before any sample was logged")

    final_error = float(ee_dist_all[-1])
    final_ee_pos = np.asarray(ee_pos, dtype=np.float64).copy()
    stop_message = stop_message or ""
    return {
        "time": np.asarray(time_all, dtype=np.float64),
        "observation": np.asarray(observation_all, dtype=np.float64),
        "wheel_cmd": np.asarray(wheel_cmd_all, dtype=np.float64),
        "torque_arm": np.asarray(torque_arm_all, dtype=np.float64),
        "wheel_applied": np.asarray(wheel_applied_all, dtype=np.float64),
        "torque_applied": np.asarray(torque_applied_all, dtype=np.float64),
        "cartesian_error": np.asarray(cartesian_error_all, dtype=np.float64),
        "reach_weight": np.asarray(reach_weight_all, dtype=np.float64),
        "ee_dist": np.asarray(ee_dist_all, dtype=np.float64),
        "target": target_ee_pos,
        "dt": np.float64(time_step),
        "reached": np.bool_(stop_message.startswith("Target reached")),
        "final_error": np.float64(final_error),
        "final_ee_pos": final_ee_pos,
        "final_time": np.float64(current_time),
        "stop_message": stop_message,
        "user_quit": user_quit,
    }


def save_episode(path, episode, split):
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            time=episode["time"],
            observation=episode["observation"],
            wheel_cmd=episode["wheel_cmd"],
            torque_arm=episode["torque_arm"],
            wheel_applied=episode["wheel_applied"],
            torque_applied=episode["torque_applied"],
            cartesian_error=episode["cartesian_error"],
            reach_weight=episode["reach_weight"],
            ee_dist=episode["ee_dist"],
            target=episode["target"],
            dt=episode["dt"],
            reached=episode["reached"],
            final_error=episode["final_error"],
            final_ee_pos=episode["final_ee_pos"],
            final_time=episode["final_time"],
            stop_message=np.asarray(episode["stop_message"]),
            split=np.asarray(split),
            observation_names=np.array(OBSERVATION_NAMES),
            label_names=np.array(LABEL_NAMES),
        )
    os.replace(tmp, path)


def check():
    """Sampler, split, observation columns, and the zero-noise DART path. No simulator."""
    if N_TRAIN + N_VAL != N_EPISODES:
        raise AssertionError(f"split {N_TRAIN}+{N_VAL} does not add up to {N_EPISODES}")
    seen = np.zeros(len(OBSERVATION_NAMES), dtype=int)
    for column in (TARGET_REL_SLICE, ROLL_PITCH_SLICE, Q_SLICE, QD_SLICE, Z_SLICE):
        seen[column] += 1
    if not np.all(seen == 1):
        raise AssertionError("observation slices overlap or leave a gap")

    q_pin = np.zeros(7)
    q_pin[6] = 1.0
    target_rel = np.array([0.2, -0.1, 0.3])
    q_arm = np.linspace(-0.2, 0.4, ARM_DOF)
    qd_arm = np.linspace(0.0, 0.1, ARM_DOF)
    z = np.array([0.01, -0.02, 0.03])
    obs = np.asarray(observation_vector(target_rel, q_pin, q_arm, qd_arm, z), dtype=float)
    assert_observation_layout(obs, target_rel, q_arm, qd_arm, z)
    if np.max(np.abs(obs[ROLL_PITCH_SLICE])) > 1e-12:
        raise AssertionError("identity base pose should have zero roll and pitch")

    lo = np.array([-1.0, -1.0, 0.5])
    hi = np.array([1.0, 1.0, 1.2])

    def draw(seed):
        rng = np.random.default_rng(seed)
        targets = sample_targets(N_EPISODES, lo, hi, rng)
        train, val = train_val_split(N_EPISODES, rng)
        return targets, train, val

    targets, train, val = draw(SEED)
    again, train_again, val_again = draw(SEED)
    if not (np.allclose(targets, again) and np.array_equal(train, train_again) and np.array_equal(val, val_again)):
        raise AssertionError("sampler or split is not determined by the seed")
    if targets.shape != (N_EPISODES, 3) or np.any(targets < lo) or np.any(targets >= hi):
        raise AssertionError("a target left the box")
    near_lo = np.array([-NEAR_XY, -NEAR_XY, NEAR_Z_LO])
    near_hi = np.array([NEAR_XY, NEAR_XY, NEAR_Z_HI])
    near_rng = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    near = sample_targets(NEAR_EPISODES, near_lo, near_hi, near_rng)
    near_again = sample_targets(NEAR_EPISODES, near_lo, near_hi, np.random.default_rng(np.random.SeedSequence([SEED, 2])))
    if near.shape != (NEAR_EPISODES, 3) or not np.allclose(near, near_again):
        raise AssertionError("near sampler is not determined by the seed")
    if np.any(near < near_lo) or np.any(near >= near_hi):
        raise AssertionError("a near target left |x|,|y| < 0.5 or the height band")
    if len(train) != N_TRAIN or len(val) != N_VAL or np.intersect1d(train, val).size or len(np.union1d(train, val)) != N_EPISODES:
        raise AssertionError("train/val split does not partition the episodes")

    params = BaseControllerParams(standoff=0.65)
    half = params.wheel_base_width / 2.0
    expect = max(params.max_ang_vel * half, params.max_lin_vel + params.max_ang_vel_driving * half) / params.wheel_radius
    if abs(wheel_speed_limit(params) - expect) > 1e-12:
        raise AssertionError("wheel speed limit does not match the twist endpoints")

    wheel = np.array([0.4, -0.2, -0.2, 0.4])
    torque = np.linspace(-1.0, 1.0, ARM_DOF)
    applied_w, applied_t = apply_dart(wheel, torque, np.random.default_rng(0), 0.0, 0.0, expect, 30.0)
    if not (np.allclose(applied_w, wheel) and np.allclose(applied_t, torque)):
        raise AssertionError("zero DART noise changed the expert command")
    if not np.allclose(clean_wheel_pair(wheel), [0.4, -0.2]):
        raise AssertionError("wheel label is not [right, left]")
    noisy_w, noisy_t = apply_dart(wheel, torque, np.random.default_rng(1), 50.0, 50.0, expect, 30.0)
    if np.any(np.abs(noisy_w) > expect + 1e-12) or np.any(np.abs(noisy_t) > 30.0 + 1e-12):
        raise AssertionError("DART noise left the expert clips")
    if not (np.allclose(noisy_w[0], noisy_w[3]) and np.allclose(noisy_w[1], noisy_w[2])):
        raise AssertionError("noisy wheel command broke the [right, left, left, right] layout")
    print(
        f"markov dataset check ok ({len(OBSERVATION_NAMES)} inputs, {len(LABEL_NAMES)} labels, "
        f"{N_TRAIN}/{N_VAL} split)",
        flush=True,
    )


def _log_missing(ctx, args, spec, indices, targets, split_name, total):
    """Log ``indices`` that are not already on disk. ``targets`` lines up with ``indices``."""
    ran = 0
    for index, target in zip(indices, targets):
        path = episode_path(args.out, index)
        if os.path.isfile(path):
            print(f"[{index + 1}/{total}] already logged ({split_name})", flush=True)
            continue
        if args.limit and ran >= args.limit:
            break
        target = np.asarray(target, dtype=float)
        print(f"[{index + 1}/{total}] {split_name}  target {np.round(target, 3)}", flush=True)
        wall_start = time.perf_counter()
        noise_rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
        episode = log_episode(
            ctx, target, noise_rng, args.wheel_noise_std, args.torque_noise_std,
            float(spec["wheel_limit"]), float(spec["torque_limit"]), float(spec["z_max"]),
        )
        save_episode(path, episode, split_name)
        ran += 1
        print(
            f"    {'reached' if episode['reached'] else 'missed'}  "
            f"error {float(episode['final_error']):.4f} m at t={float(episode['final_time']):.2f}s "
            f"(wall {time.perf_counter() - wall_start:.1f}s, {len(episode['time'])} samples) "
            f"| {episode['stop_message']}",
            flush=True,
        )
        if episode["user_quit"]:
            break
    return ran


def main():
    args = parse_args()
    if args.check:
        check()
        return
    if args.wheel_noise_std < 0.0 or args.torque_noise_std < 0.0:
        raise SystemExit("Noise standard deviations must be >= 0")
    if args.near:
        ctx = task.setup_simulation()
        ran = 0
        try:
            spec = ensure_near_dataset(ctx, args.out, args.seed, args.wheel_noise_std, args.torque_noise_std)
            local = episode_indices(NEAR_EPISODES, args.shard)
            indices = [N_EPISODES + offset for offset in local]
            targets = [np.asarray(spec["near_targets"][offset], dtype=float) for offset in local]
            total = N_EPISODES + NEAR_EPISODES
            print(f"Logging {len(indices)} near episode slots into {args.out}", flush=True)
            ran = _log_missing(ctx, args, spec, indices, targets, "train", total)
        finally:
            ctx["client"].disconnect()
        print(f"Finished. New episodes this run: {ran}", flush=True)
        return
    indices = episode_indices(N_EPISODES, args.shard)
    ctx = task.setup_simulation()
    ran = 0
    try:
        spec = ensure_dataset(ctx, args.out, args.seed, args.wheel_noise_std, args.torque_noise_std)
        split_name = np.array(["train"] * N_EPISODES, dtype="U8")
        split_name[np.asarray(spec["val_index"], dtype=int)] = "val"
        print(f"Logging {len(indices)} episode slots into {args.out}", flush=True)
        for index in indices:
            path = episode_path(args.out, index)
            if os.path.isfile(path):
                print(f"[{index + 1}/{N_EPISODES}] already logged ({split_name[index]})", flush=True)
                continue
            if args.limit and ran >= args.limit:
                break
            target = np.asarray(spec["targets"][index], dtype=float)
            print(
                f"[{index + 1}/{N_EPISODES}] {split_name[index]}  target {np.round(target, 3)}",
                flush=True,
            )
            wall_start = time.perf_counter()
            noise_rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
            episode = log_episode(
                ctx, target, noise_rng, args.wheel_noise_std, args.torque_noise_std,
                float(spec["wheel_limit"]), float(spec["torque_limit"]), float(spec["z_max"]),
            )
            save_episode(path, episode, str(split_name[index]))
            ran += 1
            print(
                f"    {'reached' if episode['reached'] else 'missed'}  "
                f"error {float(episode['final_error']):.4f} m at t={float(episode['final_time']):.2f}s "
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
