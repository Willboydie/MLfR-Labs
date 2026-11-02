"""Deployed MLP: the cross-validation winner, tanh with two layers of 128.

The network maps the Markov observation (heading-frame target, roll, pitch,
arm position and velocity, integrator z) to two wheel speeds and seven arm
torques. z is not inside the network. Playback updates it with the expert's
cartesian error and reach weight.

Train the file the GUI loads with::

    pixi run python -u ../final/robotnik_arm_task/cv_markov_mlp.py --train-winner
"""
import os

import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TASK_DIR, "markov_mlp_cv", "tanh-128x128.joblib")

# Must match the winning row of cv_markov_mlp.CONFIGS.
WINNER = {
    "name": "tanh-128x128",
    "hidden": (128, 128),
    "activation": "tanh",
    "lr": 1e-3,
    "alpha": 1e-4,
}
N_TRAIN_EPISODES = 80


def load_policy(path=None):
    """Load the trained network and the scalers that were fit with it."""
    import joblib

    path = MODEL_PATH if path is None else path
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No trained MLP at {path}. From RoboEnv, run: "
            "pixi run python -u ../final/robotnik_arm_task/cv_markov_mlp.py --train-winner"
        )
    policy = joblib.load(path)
    if policy.get("name") != WINNER["name"]:
        raise RuntimeError(f"{path} is {policy.get('name')!r}, not {WINNER['name']}.")
    return policy


def predict_command(policy, observation):
    """Wheel speeds ``[right, left, left, right]`` and seven arm torques.

    Both are clipped to the limits stored with the model, which are the expert's
    own command limits.
    """
    obs = np.asarray(observation, dtype=np.float64).reshape(1, -1)
    if obs.shape[1] != 22:
        raise ValueError(f"MLP observation must have 22 entries, got {obs.shape[1]}")
    scaled = policy["x_scaler"].transform(obs)
    raw = policy["y_scaler"].inverse_transform(policy["model"].predict(scaled))[0]
    wheel_limit = float(policy["wheel_limit"])
    torque_limit = float(policy["torque_limit"])
    right = float(np.clip(raw[0], -wheel_limit, wheel_limit))
    left = float(np.clip(raw[1], -wheel_limit, wheel_limit))
    torque = np.clip(np.asarray(raw[2:], dtype=np.float64), -torque_limit, torque_limit)
    return np.array([right, left, left, right], dtype=np.float64), torque
