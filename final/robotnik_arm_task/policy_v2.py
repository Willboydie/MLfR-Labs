"""Split base/arm networks, and the single network they are compared with.

The base law is a function of the heading-frame target and the standoff, so
its network is small and it does not see the arm. A shallow decision tree
classifies the expert's exact wheel-deadzone and, when it fires, the wheels
are set to zero. A tanh classifier did not learn that rectangle: at 200 Adam
steps it still left about 15% of the deadzone moving, because the expert's
yaw command falls from about 0.45 rad/s to zero across 0.05 rad. The tree
splits on the same two errors the base law uses. Mean squared error on its
own does not zero the wheels: the deadzone is a thin shell and an error of
0.2 rad/s barely moves a standardised loss. A command the base net already
puts under 0.02 rad/s is zeroed as well.

The arm network sees the cartesian error, the reach weight, and the gravity
torque, which deployment already computes in order to update the integrator.
Its label is the torque with gravity removed. Gravity is added back and the
sum is clipped to the expert's limit. The useful holding torque at 1 cm is
about 1 N·m; leaving gravity in the label makes that term a small residual
on top of a much larger, known vector.

The monolithic network maps the same arm features to wheels and torques
together, with gravity left in the torque label and with no deadzone gate.
Closed-loop comparison of the two is the architecture result. Neither fit uses
row-wise early stopping. The number reported beside the weights is an
episode holdout, and the number that matters is the frozen test.

``python policy_v2.py --check`` fits the base net and the gate on the
closed-form base law only. It does not start a simulator.
"""
import time

import numpy as np
import pinocchio as pin
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from features_v2 import (
    CONTINUOUS_JOINTS,
    arm_feature_names,
    arm_features_row,
    base_feature_matrix,
    base_feature_names,
    parked_labels,
    synthetic_base_samples,
)
from robotnik_markov_expert import heading_frame_configuration

BASE_HIDDEN = (32, 32)
PARK_DEPTH = 6
ARM_HIDDEN = (128, 128)
MONO_HIDDEN = (128, 128)
ACTIVATION = "tanh"
LR = 1e-3
ALPHA = 1e-4
BATCH_SIZE = 64
SEED = 0


def count_params(model):
    total = 0
    for coef in getattr(model, "coefs_", ()):
        total += int(np.size(coef))
    for intercept in getattr(model, "intercepts_", ()):
        total += int(np.size(intercept))
    return total


def _regressor(hidden, max_iter):
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=ACTIVATION,
        alpha=ALPHA,
        learning_rate_init=LR,
        batch_size=BATCH_SIZE,
        max_iter=max_iter,
        early_stopping=False,
        shuffle=True,
        random_state=SEED,
        solver="adam",
        tol=1e-4,
    )


def _fit_regressor(model, features, labels, sample_weight):
    scaler_x = StandardScaler().fit(features)
    scaler_y = StandardScaler().fit(labels)
    model.fit(scaler_x.transform(features), scaler_y.transform(labels), sample_weight=sample_weight)
    return model, scaler_x, scaler_y


def _predict_regressor(model, scaler_x, scaler_y, features):
    raw = scaler_y.inverse_transform(model.predict(scaler_x.transform(features)))
    return np.asarray(raw, dtype=np.float64)


def choose_park_threshold(labels, probability):
    """Highest deadzone recall whose precision is still at least 0.90.

    A false park stops the base on the way in. A missed park is the creep
    this gate exists to remove, so recall is what the search maximises once
    precision is acceptable. 0.5 is the fallback when no threshold qualifies.
    """
    labels = np.asarray(labels, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    best_strict = None
    best_precise = None
    for threshold in np.linspace(0.05, 0.95, 37):
        pred = probability >= threshold
        true_positive = int(np.sum(pred & labels))
        false_positive = int(np.sum(pred & ~labels))
        false_negative = int(np.sum(~pred & labels))
        precision = true_positive / (true_positive + false_positive + 1e-9)
        recall = true_positive / (true_positive + false_negative + 1e-9)
        if precision >= 0.90 and (best_strict is None or (recall, precision) > best_strict[0]):
            best_strict = ((recall, precision), float(threshold))
        if best_precise is None or (precision, recall) > best_precise[0]:
            best_precise = ((precision, recall), float(threshold))
    if best_strict is not None:
        return best_strict[1]
    return best_precise[1]


def fit_park(features, wheels, max_iter):
    """Deadzone classifier.

    The label is the expert's exact zero. The fade sits just outside that box
    and is a negative example, so the cut stays on the box rather than in the
    gap beyond it. ``max_iter`` is unused: this fit is a tree, not Adam.
    """
    del max_iter
    wheels = np.asarray(wheels, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    labels = parked_labels(wheels)
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    if len(positive) < 20 or len(negative) < 20:
        raise RuntimeError(f"Park fit needs both classes, got {len(positive)} parked and {len(negative)} driving")
    # Tune the threshold on a slice the classifier does not train on.
    hold = rng.permutation(len(features))
    cut = max(1, int(round(0.2 * len(features))))
    tune = hold[:cut]
    train = hold[cut:]
    # Every row is used, including the fade. The label changes on a rectangle
    # in (abs dist error, abs bearing), and a depth-6 tree can cut that
    # rectangle. A tanh net smoothed across the 0.05 rad shoulder instead.
    scaler = StandardScaler().fit(features[train])
    model = DecisionTreeClassifier(
        max_depth=PARK_DEPTH,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
    )
    model.fit(scaler.transform(features[train]), labels[train].astype(np.int64))
    probability = model.predict_proba(scaler.transform(features[tune]))[:, 1]
    threshold = choose_park_threshold(labels[tune], probability)
    pred = probability >= threshold
    true_positive = int(np.sum(pred & labels[tune]))
    false_positive = int(np.sum(pred & ~labels[tune]))
    false_negative = int(np.sum(~pred & labels[tune]))
    stats = {
        "threshold": float(threshold),
        "tune_recall": true_positive / (true_positive + false_negative + 1e-9),
        "tune_precision": true_positive / (true_positive + false_positive + 1e-9),
        "tune_n": int(len(tune)),
        "tune_park_fraction": float(np.mean(labels[tune])),
        "kind": "decision_tree",
        "max_depth": int(PARK_DEPTH),
        "n_nodes": int(model.tree_.node_count),
    }
    return model, scaler, threshold, stats


def gate_wheels(pair, probability, threshold):
    """Exact zero in the deadzone. A command already under 0.02 rad/s is zeroed too.

    The second clause is the creep the squared loss does not penalise: the base
    network is close, and the classifier is unsure, and the wheels would
    otherwise keep a small speed forever.
    """
    pair = np.asarray(pair, dtype=np.float64).reshape(-1)
    small = float(np.max(np.abs(pair))) < 0.02
    if float(probability) >= float(threshold) or (float(probability) >= 0.35 and small):
        return np.zeros(4, dtype=np.float64)
    return expand_wheels(pair)


def expand_wheels(pair):
    """``[right, left]`` to the plant order ``[right, left, left, right]``."""
    right = float(pair[0])
    left = float(pair[1])
    return np.array([right, left, left, right], dtype=np.float64)


def gravity_torque(expert, q_pin):
    """The gravity term the expert adds, in the heading frame, shape (7,)."""
    data = getattr(expert, "_grav_data", None)
    if data is None:
        data = expert.model.createData()
        expert._grav_data = data
    q_canon = heading_frame_configuration(q_pin)
    generalised = pin.computeGeneralizedGravity(expert.model, data, q_canon)
    return np.asarray(generalised[expert.arm_v], dtype=np.float64)


def feedback(expert, q_pin, target_rel):
    """Cartesian error, reach weight, and gravity. All three are the expert's own formulas."""
    error, reach_w = expert.cartesian_feedback(q_pin, target_rel)
    tau_g = gravity_torque(expert, q_pin)
    return np.asarray(error, dtype=np.float64), float(reach_w), tau_g


def command_from_policy(policy, q_pin, q_arm, qd_arm, target_rel, z, expert, feedback_tuple=None,
                        use_base=True, use_arm=True):
    """Wheel command (4,) and arm torque (7,) from a split or monolithic policy.

    ``use_base`` and ``use_arm`` select which half of a split policy to evaluate.
    The caller fills the other half from the expert. A monolithic policy always
    evaluates both heads; the unused half is discarded by the caller.
    """
    if feedback_tuple is None:
        error, reach_w, tau_g = feedback(expert, q_pin, target_rel)
    else:
        error, reach_w, tau_g = feedback_tuple
    roll_pitch = _roll_pitch(q_pin)
    continuous = np.asarray(policy["continuous_joints"], dtype=bool)
    wheels = np.zeros(4, dtype=np.float64)
    torque = np.zeros(7, dtype=np.float64)
    if policy["kind"] == "mono":
        features = arm_features_row(
            target_rel, roll_pitch, q_arm, qd_arm, z, error, reach_w, tau_g, continuous,
        ).reshape(1, -1)
        raw = _predict_regressor(policy["model"], policy["x_scaler"], policy["y_scaler"], features)[0]
        wheels = expand_wheels(raw[:2])
        torque = np.asarray(raw[2:], dtype=np.float64)
    elif policy["kind"] == "split":
        if use_base:
            base_row = base_feature_matrix(target_rel, expert.base_params.standoff).reshape(1, -1)
            pair = _predict_regressor(
                policy["base_model"], policy["base_x_scaler"], policy["base_y_scaler"], base_row,
            )[0]
            probability = float(policy["park_model"].predict_proba(policy["park_scaler"].transform(base_row))[0, 1])
            # Exact zero inside the learned deadzone. Clipping below must not turn it back on.
            wheels = gate_wheels(pair, probability, policy["park_threshold"])
        if use_arm:
            arm_row = arm_features_row(
                target_rel, roll_pitch, q_arm, qd_arm, z, error, reach_w, tau_g, continuous,
            ).reshape(1, -1)
            residual = _predict_regressor(
                policy["arm_model"], policy["arm_x_scaler"], policy["arm_y_scaler"], arm_row,
            )[0]
            torque = residual + tau_g
    else:
        raise ValueError(f"Unknown policy kind {policy.get('kind')!r}")
    wheel_limit = float(policy["wheel_limit"])
    torque_limit = float(policy["torque_limit"])
    wheels = np.clip(wheels, -wheel_limit, wheel_limit)
    if policy["kind"] == "split" and use_base and np.all(wheels == 0.0):
        wheels = np.zeros(4, dtype=np.float64)
    torque = np.clip(np.asarray(torque, dtype=np.float64), -torque_limit, torque_limit)
    return wheels, torque, np.asarray(error, dtype=np.float64), float(reach_w)


def _roll_pitch(q_pin):
    quat = pin.Quaternion(q_pin[6], q_pin[3], q_pin[4], q_pin[5])
    quat.normalize()
    roll, pitch, _ = pin.rpy.matrixToRpy(quat.toRotationMatrix())
    return np.array([roll, pitch], dtype=np.float64)


def _empty_policy(kind, continuous, limits):
    return {
        "kind": kind,
        "continuous_joints": np.asarray(continuous, dtype=bool).tolist(),
        "wheel_limit": float(limits["wheel_limit"]),
        "torque_limit": float(limits["torque_limit"]),
        "activation": ACTIVATION,
        "lr": LR,
        "alpha": ALPHA,
        "batch_size": BATCH_SIZE,
        "early_stopping": False,
    }


def fit_split(base_x, base_y, base_w, arm_x, arm_y, arm_w, continuous, limits, max_iter, warm=None):
    """Fit the base net, the deadzone gate, and the arm net.

    ``warm`` is a previous split policy. The two regressors continue from its
    weights and from its scalers, so a DAgger round does not restart them and
    does not refit the standardisation on a different label scale. The gate is
    small and is fit from scratch, because its threshold is retuned each time.
    """
    policy = _empty_policy("split", continuous, limits)
    if warm is None:
        base_model, base_x_scaler, base_y_scaler = _fit_regressor(
            _regressor(BASE_HIDDEN, max_iter), base_x, base_y, base_w,
        )
        arm_model, arm_x_scaler, arm_y_scaler = _fit_regressor(
            _regressor(ARM_HIDDEN, max_iter), arm_x, arm_y, arm_w,
        )
    else:
        base_x_scaler = warm["base_x_scaler"]
        base_y_scaler = warm["base_y_scaler"]
        arm_x_scaler = warm["arm_x_scaler"]
        arm_y_scaler = warm["arm_y_scaler"]
        base_model = warm["base_model"]
        arm_model = warm["arm_model"]
        base_model.set_params(max_iter=max_iter, warm_start=True, early_stopping=False)
        arm_model.set_params(max_iter=max_iter, warm_start=True, early_stopping=False)
        base_model.fit(base_x_scaler.transform(base_x), base_y_scaler.transform(base_y), sample_weight=base_w)
        arm_model.fit(arm_x_scaler.transform(arm_x), arm_y_scaler.transform(arm_y), sample_weight=arm_w)
    park_model, park_scaler, threshold, park_stats = fit_park(base_x, base_y, max_iter)
    policy.update({
        "base_model": base_model,
        "base_x_scaler": base_x_scaler,
        "base_y_scaler": base_y_scaler,
        "arm_model": arm_model,
        "arm_x_scaler": arm_x_scaler,
        "arm_y_scaler": arm_y_scaler,
        "park_model": park_model,
        "park_scaler": park_scaler,
        "park_threshold": float(threshold),
        "park_stats": park_stats,
        "base_hidden": list(BASE_HIDDEN),
        "arm_hidden": list(ARM_HIDDEN),
        "arm_residual": True,
        "n_params": int(count_params(base_model) + count_params(arm_model) + count_params(park_model)),
        "base_n_iter": int(base_model.n_iter_),
        "arm_n_iter": int(arm_model.n_iter_),
        "n_base": int(len(base_x)),
        "n_arm": int(len(arm_x)),
    })
    return policy


def fit_mono(features, labels, sample_weight, continuous, limits, max_iter):
    model, scaler_x, scaler_y = _fit_regressor(_regressor(MONO_HIDDEN, max_iter), features, labels, sample_weight)
    policy = _empty_policy("mono", continuous, limits)
    policy.update({
        "model": model,
        "x_scaler": scaler_x,
        "y_scaler": scaler_y,
        "hidden": list(MONO_HIDDEN),
        "arm_residual": False,
        "n_params": int(count_params(model)),
        "n_iter": int(model.n_iter_),
        "n_samples": int(len(features)),
    })
    return policy


def regression_rmse(policy, features, labels, head):
    """RMSE of one head on already-built features. Not a closed-loop number."""
    if head == "base":
        pred = _predict_regressor(policy["base_model"], policy["base_x_scaler"], policy["base_y_scaler"], features)
    elif head == "arm":
        pred = _predict_regressor(policy["arm_model"], policy["arm_x_scaler"], policy["arm_y_scaler"], features)
    elif head == "mono":
        pred = _predict_regressor(policy["model"], policy["x_scaler"], policy["y_scaler"], features)
    else:
        raise ValueError(head)
    return np.sqrt(np.mean((pred - labels) ** 2, axis=0))


def time_predict(policy, expert, q_pin, q_arm, qd_arm, target_rel, z, repeats=200):
    """Median milliseconds for one command, and for one expert call, after a short warmup."""
    for _ in range(5):
        command_from_policy(policy, q_pin, q_arm, qd_arm, target_rel, z, expert)
        expert(q_pin, q_arm, qd_arm, target_rel, z)
    started = time.perf_counter()
    for _ in range(repeats):
        command_from_policy(policy, q_pin, q_arm, qd_arm, target_rel, z, expert)
    policy_ms = (time.perf_counter() - started) * 1000.0 / repeats
    started = time.perf_counter()
    for _ in range(repeats):
        expert(q_pin, q_arm, qd_arm, target_rel, z)
    expert_ms = (time.perf_counter() - started) * 1000.0 / repeats
    return {"policy_ms": policy_ms, "expert_ms": expert_ms, "repeats": repeats, "n_params": int(policy["n_params"])}


def check():
    """The gate has to zero the deadzone and leave ordinary driving alone."""
    continuous = np.zeros(7, dtype=bool)
    continuous[list(CONTINUOUS_JOINTS)] = True
    if arm_feature_names(continuous)[0] != "target_rel_x":
        raise AssertionError("arm feature names drifted")
    if base_feature_names()[-1] != "abs_bearing_err":
        raise AssertionError("base feature names drifted")
    if len(base_feature_names()) != 8:
        raise AssertionError("base features are no longer 8-D")
    rng = np.random.default_rng(0)
    features, labels = synthetic_base_samples(1500, 600, 1500, base_z=0.08, rng=rng)
    weights = np.ones(len(features))
    weights[:1500] = 5.0
    model, x_scaler, y_scaler = _fit_regressor(_regressor(BASE_HIDDEN, 120), features, labels, weights)
    park_model, park_scaler, threshold, stats = fit_park(features, labels, 120)
    pred = _predict_regressor(model, x_scaler, y_scaler, features)
    probability = park_model.predict_proba(park_scaler.transform(features))[:, 1]
    small = np.max(np.abs(pred), axis=1) < 0.02
    fire = (probability >= threshold) | ((probability >= 0.35) & small)
    gated = pred.copy()
    gated[fire] = 0.0
    dead = np.max(np.abs(labels), axis=1) < 1e-8
    drive = np.max(np.abs(labels), axis=1) > 0.4
    dead_left = float(np.mean(np.max(np.abs(gated[dead]), axis=1) > 1e-6))
    drive_killed = float(np.mean(fire[drive]))
    dead_rmse = float(np.sqrt(np.mean(gated[dead] ** 2)))
    print(
        f"policy_v2 check: deadzone leak {dead_left:.3f}, drive false-park {drive_killed:.3f}, "
        f"gated deadzone rmse {dead_rmse:.4f} rad/s, threshold {threshold:.2f}, "
        f"park recall {stats['tune_recall']:.2f}",
        flush=True,
    )
    if dead_left > 0.05:
        raise SystemExit("The park gate left more than 5% of the deadzone moving")
    if drive_killed > 0.05:
        raise SystemExit("The park gate stopped more than 5% of ordinary driving")
    print("policy_v2 ok", flush=True)


if __name__ == "__main__":
    check()
