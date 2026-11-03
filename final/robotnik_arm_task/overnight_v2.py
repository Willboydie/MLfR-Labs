"""Resume-safe pipeline: collect, train, frozen test, swap, DAgger, robustness.

From the RoboEnv checkout::

    pixi run python -u ../final/robotnik_arm_task/overnight_v2.py
    pixi run python -u ../final/robotnik_arm_task/overnight_v2.py --smoke

``--smoke`` logs two episodes, trains for a few epochs, and scores one grid
point, to check that the pieces join. The full run is the one to leave
overnight. Every stage skips work that is already on disk. The progress
report is rewritten after each stage.
"""
import argparse
import os
import subprocess
import sys

from progress_report import REPORT_PATH, ROOT, select_winner, write_report


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(description="Run the frozen-test rebuild.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--root", default="")
    return parser.parse_args()


def _run(script, *args):
    command = [PY, "-u", os.path.join(TASK_DIR, script), *args]
    print("\n=== " + " ".join(command) + "\n", flush=True)
    subprocess.check_call(command)


def _report(root):
    try:
        if os.path.isfile(os.path.join(root, "results.jsonl")) or os.path.isfile(os.path.join(root, "data", "index.jsonl")):
            select_winner(root)
        path = write_report(root, REPORT_PATH if root == ROOT else os.path.join(TASK_DIR, "progress_report_smoke.md"))
        print(f"Wrote {path}", flush=True)
    except Exception as exc:
        print(f"Report update failed: {exc}", flush=True)


def main():
    args = parse_args()
    root = args.root or (os.path.join(TASK_DIR, "markov_v2_smoke") if args.smoke else ROOT)
    data = os.path.join(root, "data")
    frozen = os.path.join(root, "frozen_targets.npz")
    episodes = "2" if args.smoke else "80"
    epochs = "4" if args.smoke else "80"
    dagger_epochs = "4" if args.smoke else "30"
    dagger_iters = "1" if args.smoke else "3"
    dagger_rollouts = "1" if args.smoke else "8"
    limit = ["--limit", "1"] if args.smoke else []
    legacy = [
        ("legacy_dagger", os.path.join(TASK_DIR, "markov_dagger", "winner.joblib")),
        ("legacy_all", os.path.join(TASK_DIR, "markov_mlp_cv_all", "winner.joblib")),
        ("legacy_success", os.path.join(TASK_DIR, "markov_mlp_cv_success", "winner.joblib")),
    ]
    try:
        _run("features_v2.py")
        _run("policy_v2.py")
        _run("collect_v2.py", "--out", data, "--frozen", frozen, "--episodes", episodes)
        _report(root)
        _run("train_v2.py", "--data", data, "--out", root, "--max-iter", epochs, "--tag", "split_iter_00", "--set-winner")
        _report(root)
        _run("eval_frozen.py", "--frozen", frozen, "--out", root, "--controller", "markov", "--name", "markov", *limit)
        _run("eval_frozen.py", "--frozen", frozen, "--out", root, "--controller", "phased", "--name", "phased", "--sets", "grid", *limit)
        split = os.path.join(root, "split_iter_00.joblib")
        _run("eval_frozen.py", "--frozen", frozen, "--out", root, "--controller", "policy", "--name", "split_iter_00", "--policy", split, *limit)
        _run("eval_frozen.py", "--frozen", frozen, "--out", root, "--controller", "swap-expert-base", "--name", "expert_base", "--policy", split, *limit)
        _run("eval_frozen.py", "--frozen", frozen, "--out", root, "--controller", "swap-learned-base", "--name", "learned_base", "--policy", split, *limit)
        mono = os.path.join(root, "mono.joblib")
        if os.path.isfile(mono):
            _run("eval_frozen.py", "--frozen", frozen, "--out", root, "--controller", "policy", "--name", "mono", "--policy", mono, *limit)
        _report(root)
        if not args.smoke:
            env = os.environ.copy()
            env["V2_DAGGER_ITERS"] = dagger_iters
            env["V2_DAGGER_ROLLOUTS"] = dagger_rollouts
            command = [PY, "-u", os.path.join(TASK_DIR, "dagger_v2.py"), "--data", data, "--out", root, "--iters", dagger_iters, "--rollouts", dagger_rollouts, "--epochs", dagger_epochs]
            print("\n=== " + " ".join(command) + "\n", flush=True)
            subprocess.check_call(command, env=env)
            for iteration in range(1, int(dagger_iters) + 1):
                path = os.path.join(root, f"split_iter_{iteration:02d}.joblib")
                if os.path.isfile(path):
                    _run(
                        "eval_frozen.py", "--frozen", frozen, "--out", root, "--controller", "policy",
                        "--name", f"split_iter_{iteration:02d}", "--policy", path,
                    )
            _report(root)
            for name, path in legacy:
                if os.path.isfile(path):
                    _run("eval_frozen.py", "--frozen", frozen, "--out", root, "--controller", "policy", "--name", name, "--policy", path)
            _report(root)
        winner = os.path.join(root, "winner.joblib")
        if os.path.isfile(winner) and not args.smoke:
            _run(
                "eval_frozen.py", "--frozen", frozen, "--out", root, "--controller", "policy",
                "--name", "deployed", "--policy", winner, "--robustness", "--time",
            )
        _report(root)
    finally:
        _report(root)
    print(f"Finished. Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
