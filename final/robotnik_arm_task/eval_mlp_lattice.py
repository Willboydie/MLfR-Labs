"""Headless playback of the dataset winners on a 27-point test lattice.

The lattice is the 3 x 3 x 3 grid of a 2 m x 2 m x 1 m box centred at
(0, 0, 0.5). The points sit on the boundary as well as in the middle:

* x and y in {-1, 0, 1}
* z in {0, 0.5, 1}

The Markov expert is run on the same points. A reach is the expert's own
stop: end-effector error under 10 mm.

From the RoboEnv checkout::

    pixi run python -u ../final/robotnik_arm_task/eval_mlp_lattice.py --check
    pixi run python -u ../final/robotnik_arm_task/eval_mlp_lattice.py
"""
import os
import sys

if "--check" not in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import argparse
import json

import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(TASK_DIR, "markov_mlp_lattice")
CENTER = np.array([0.0, 0.0, 0.5])
MODELS = {
    "all": os.path.join(TASK_DIR, "markov_mlp_cv_all", "winner.joblib"),
    "success": os.path.join(TASK_DIR, "markov_mlp_cv_success", "winner.joblib"),
}
DAGGER_MODEL = os.path.join(TASK_DIR, "markov_dagger", "winner.joblib")


def lattice_targets():
    """27 world-frame points. Their mean is the box centre."""
    xs = CENTER[0] + np.array([-1.0, 0.0, 1.0])
    ys = CENTER[1] + np.array([-1.0, 0.0, 1.0])
    zs = CENTER[2] + np.array([-0.5, 0.0, 0.5])
    targets = np.array([(x, y, z) for x in xs for y in ys for z in zs], dtype=float)
    if targets.shape != (27, 3):
        raise RuntimeError(f"Expected 27 lattice points, got {targets.shape[0]}.")
    if not np.allclose(targets.mean(axis=0), CENTER):
        raise RuntimeError(f"Lattice centre is {targets.mean(axis=0)}, not {CENTER}.")
    low = CENTER - np.array([1.0, 1.0, 0.5])
    high = CENTER + np.array([1.0, 1.0, 0.5])
    if np.any(targets < low - 1e-9) or np.any(targets > high + 1e-9):
        raise RuntimeError("A lattice point sits outside the 2 x 2 x 1 m box.")
    return targets


def parse_args():
    parser = argparse.ArgumentParser(description="Play the winning MLPs on the 27-point lattice.")
    parser.add_argument("--check", action="store_true", help="Print the lattice and exit.")
    return parser.parse_args()


def result_path():
    return os.path.join(OUT_DIR, "results.jsonl")


def load_done(path):
    done = {}
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            done[(record["controller"], int(record["index"]))] = record
    return done


def append_result(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_policy(path):
    import joblib

    from markov_mlp import predict_command

    if not os.path.isfile(path):
        raise SystemExit(f"No winner at {path}. Run cv_dataset_mlp.py first.")
    policy = joblib.load(path)
    predict_command(policy, np.zeros(22))
    return policy


def summarise(records):
    rows = []
    for controller in ("markov", "all", "success", "dagger"):
        items = [record for record in records if record["controller"] == controller and "final_error" in record]
        if not items:
            continue
        error = np.array([item["final_error"] for item in items], dtype=float)
        reached = np.array([item["reached"] for item in items], dtype=bool)
        by_z = []
        for z_value in (0.0, 0.5, 1.0):
            chosen = [item for item in items if abs(item["target"][2] - z_value) < 1e-9]
            z_error = np.array([item["final_error"] for item in chosen], dtype=float)
            z_reached = np.array([item["reached"] for item in chosen], dtype=bool)
            by_z.append({
                "z": z_value,
                "n": int(len(chosen)),
                "n_reached": int(z_reached.sum()),
                "median_error": float(np.median(z_error)),
            })
        rows.append({
            "controller": controller,
            "name": items[0].get("name", "markov"),
            "n": int(len(items)),
            "n_reached": int(reached.sum()),
            "median_error": float(np.median(error)),
            "mean_error": float(error.mean()),
            "max_error": float(error.max()),
            "by_z": by_z,
        })
    return rows


def print_summary(rows):
    print(f"\n{'controller':<10}  {'reached':>10}  {'median m':>10}  {'mean m':>10}  {'max m':>10}", flush=True)
    for row in rows:
        print(
            f"{row['controller']:<10}  {row['n_reached']:>4}/{row['n']:<5}  "
            f"{row['median_error']:10.4f}  {row['mean_error']:10.4f}  {row['max_error']:10.4f}",
            flush=True,
        )
        for band in row["by_z"]:
            print(
                f"    z={band['z']:.1f}  {band['n_reached']}/{band['n']}  "
                f"median {band['median_error']:.4f} m",
                flush=True,
            )


def main():
    args = parse_args()
    targets = lattice_targets()
    if args.check:
        print(f"{len(targets)} targets centred at {CENTER.tolist()}.", flush=True)
        print(np.array2string(targets, precision=1, floatmode="fixed"), flush=True)
        return

    import robotnik_arm_task as task

    policies = {name: load_policy(path) for name, path in MODELS.items()}
    controllers = ["markov", "all", "success"]
    if os.path.isfile(DAGGER_MODEL):
        policies["dagger"] = load_policy(DAGGER_MODEL)
        controllers.append("dagger")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = result_path()
    done = load_done(path)
    ctx = task.setup_simulation()
    try:
        for index, target in enumerate(targets):
            for controller in controllers:
                if (controller, index) in done:
                    previous = done[(controller, index)]
                    same_policy = (
                        controller != "dagger"
                        or previous.get("iteration") == policies["dagger"].get("iteration")
                    )
                    if same_policy and "final_error" in previous:
                        continue
                if controller == "markov":
                    ctx.pop("mlp_policy", None)
                    run_as = "markov"
                    model_name = "markov"
                else:
                    ctx["mlp_policy"] = policies[controller]
                    run_as = "mlp"
                    model_name = policies[controller]["name"]
                print(
                    f"[{controller} {index + 1}/27] {model_name}  target {np.round(target, 1)}",
                    flush=True,
                )
                try:
                    result = task.run_episode(
                        ctx, target, record=False, verbose=False, realtime=False, controller=run_as,
                    )
                except Exception as exc:
                    append_result(path, {
                        "controller": controller,
                        "name": model_name,
                        "index": index,
                        "target": target.tolist(),
                        "error": str(exc),
                    })
                    print(f"    failed: {exc}", flush=True)
                    continue
                message = result["stop_message"] or ""
                record = {
                    "controller": controller,
                    "name": model_name,
                    "index": index,
                    "target": target.tolist(),
                    "final_error": float(result["final_error"]),
                    "final_ee_pos": np.asarray(result["final_ee_pos"], dtype=float).tolist(),
                    "time": float(result["time"]),
                    "stop_message": message,
                    "reached": bool(message.startswith("Target reached")),
                    "iteration": None if controller == "markov" else policies[controller].get("iteration"),
                }
                append_result(path, record)
                done[(controller, index)] = record
                print(
                    f"    {'reached' if record['reached'] else 'missed'}  "
                    f"error {record['final_error']:.4f} m at t={record['time']:.2f}s  | {message}",
                    flush=True,
                )
    finally:
        ctx["client"].disconnect()

    records = list(load_done(path).values())
    rows = summarise(records)
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "center": CENTER.tolist(),
            "box_m": [2.0, 2.0, 1.0],
            "targets": targets.tolist(),
            "controllers": rows,
        }, handle, indent=2)
    print_summary(rows)
    expected = len(targets) * len(controllers)
    finished = sum(row["n"] for row in rows)
    if finished != expected:
        raise SystemExit(f"Lattice playback finished {finished} of {expected} runs.")
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
