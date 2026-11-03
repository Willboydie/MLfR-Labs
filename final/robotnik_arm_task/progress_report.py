"""Lecturer-facing progress report for the frozen-test rebuild.

The file is rewritten from whatever results are on disk, so a run that is
still going still leaves an up-to-date account. Means omit unstable episodes.
A later DAgger iteration is deployed only when it beats the others on the
frozen set.
"""
import glob
import json
import os
import shutil

import numpy as np

from eval_frozen import load_records, result_path, summarise


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(TASK_DIR, "markov_v2")
REPORT_PATH = os.path.join(TASK_DIR, "progress_report.md")
PLOT_DIR = os.path.join(TASK_DIR, "plots", "v2")


def _read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _fmt(value, digits=3, suffix=""):
    if value is None:
        return "—"
    return f"{value:.{digits}f}{suffix}"


def _rate(row, key):
    if row is None:
        return "—"
    return f"{row['within'][key]}/{row['n']}"


def select_winner(root):
    """Deploy the full-set policy with the best 25 mm count, then 10 mm, then stability, then median error."""
    records = list(load_records(result_path(root)).values())
    rows = summarise(records)
    best = None
    considered = []
    for row in rows:
        name = row["controller"]
        eligible = name == "mono" or name.startswith("split_iter")
        path = os.path.join(root, f"{name}.joblib")
        full = row["n_grid"] >= 24 and row["n_heldout"] >= 20 and os.path.isfile(path)
        if not eligible or not full:
            continue
        median = 1e9 if row["median_error"] is None else float(row["median_error"])
        key = (row["within"]["25"], row["within"]["10"], -row["n_unstable"], -median)
        considered.append({"controller": name, "within_25": row["within"]["25"], "within_10": row["within"]["10"],
                           "n_unstable": row["n_unstable"], "median_error": row["median_error"]})
        if best is None or key > best[0]:
            best = (key, path, name)
    if best is None:
        return None
    winner = os.path.join(root, "winner.joblib")
    shutil.copyfile(best[1], winner)
    payload = {
        "controller": best[2],
        "source": os.path.basename(best[1]),
        "rule": "On the full frozen set: most targets within 25 mm, then within 10 mm, then fewest unstable runs, then lowest median error among the stable runs. Unstable runs count as misses and are left out of the median.",
        "considered": considered,
    }
    with open(os.path.join(root, "winner_selection.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _bars(rows):
    os.makedirs(PLOT_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = [row for row in rows if row["n"] >= 8 and not row["controller"].startswith("robust") and "noise" not in row["controller"]]
    if not keep:
        return []
    labels = [row["controller"].replace("_", "\n") for row in keep]
    x = np.arange(len(keep))
    width = 0.18
    figure, axis = plt.subplots(figsize=(max(8, len(keep) * 1.3), 4.8))
    for offset, (mm, color) in enumerate((("10", "#1b4f72"), ("25", "#148f77"), ("50", "#b9770e"), ("100", "#922b21"))):
        height = [row["within"][mm] / row["n"] for row in keep]
        axis.bar(x + (offset - 1.5) * width, height, width, label=f"{mm} mm", color=color)
    axis.set_ylabel("Fraction of targets")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_title("Frozen-set success. Unstable runs count as misses.")
    axis.legend(ncol=4, fontsize=8)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    path = os.path.join(PLOT_DIR, "success_rates.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return ["plots/v2/success_rates.png"]


def _dataset_section(root):
    meta_path = os.path.join(root, "data", "dataset_meta.npz")
    index = _read_jsonl(os.path.join(root, "data", "index.jsonl"))
    if not os.path.isfile(meta_path):
        return "Collection has not created a dataset yet.\n"
    with np.load(meta_path) as meta:
        targets = np.asarray(meta["targets"], dtype=float)
        n = int(meta["n_episodes"])
        wheel_std = float(meta["wheel_noise_std"])
        torque_std = float(meta["torque_noise_std"])
        continuous = np.asarray(meta["continuous_joints"], dtype=bool)
        n_val = int(len(meta["val_index"]))
    logged = len(index)
    reached = sum(bool(row["reached"]) for row in index)
    spun = sum(bool(row["spun"]) for row in index)
    randomized = sum(bool(row["randomized"]) for row in index)
    kept = sum(int(row.get("n_kept", 0)) for row in index)
    raw = sum(int(row.get("n_logged", 0)) for row in index)
    beyond = int(np.sum((targets[:, 2] < 0.25) | (targets[:, 2] > 1.25)))
    lines = [
        f"The training draw is {n} targets, of which {logged} have been logged. "
        f"Heights run from {targets[:, 2].min():.2f} m to {targets[:, 2].max():.2f} m, "
        f"and {beyond} of the {n} ({beyond / n:.0%}) sit slightly outside 0.25–1.25 m. "
        f"None of them is a frozen test point.",
        "",
        f"Of the logged episodes, {reached} reached under the expert's own 10 mm dwell, "
        f"{spun} were cut when joint 1 or 3 spun, and {randomized} started from a perturbed base and arm "
        f"({logged - randomized} started at the default pose). "
        f"The spin cut removed {raw - kept} of {raw} logged rows. "
        f"DART noise on the plant command is {wheel_std} rad/s on the wheels and {torque_std} N·m on the torques. "
        f"The stored label is the clean expert command.",
        "",
        f"Unlimited joints, from the simulator limits, are {np.flatnonzero(continuous).tolist()} "
        "(0-based: joints 1, 3, 5 and 7 when this is 0, 2, 4, 6). Those angles enter the network as sin and cos. "
        f"An episode holdout of {n_val} expert episodes is used only for the offline RMSE. "
        "The deployed weights are fit on every expert episode.",
    ]
    return "\n".join(lines) + "\n"


def _table(rows, names):
    header = "| Controller | Reached | 10 mm | 25 mm | 50 mm | 100 mm | Unstable | Median error | First 10 mm | Stop time |"
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = [header, rule]
    found = False
    by_name = {row["controller"]: row for row in rows}
    for name in names:
        row = by_name.get(name)
        if row is None:
            continue
        found = True
        body.append(
            f"| {name} | {row['n_reached']}/{row['n']} | {_rate(row, '10')} | {_rate(row, '25')} | "
            f"{_rate(row, '50')} | {_rate(row, '100')} | {row['n_unstable']}/{row['n']} | "
            f"{_fmt(row['median_error'])} m | {_fmt(row.get('median_time_to_10'), 1)} s | {_fmt(row['median_time_reached'], 1)} s |"
        )
    if not found:
        return "No frozen-set rows for this table yet.\n"
    return "\n".join(body) + "\n"


def write_report(root=ROOT, dest=REPORT_PATH):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    records = list(load_records(result_path(root)).values())
    rows = summarise(records)
    by_name = {row["controller"]: row for row in rows}
    selection = _read_json(os.path.join(root, "winner_selection.json"))
    timing = _read_json(os.path.join(root, "timing.json"))
    dagger = _read_json(os.path.join(root, "dagger", "iterations.json")) or []
    summaries = []
    for path in sorted(glob.glob(os.path.join(root, "*_summary.json"))):
        summaries.append(_read_json(path))
    split_summary = next((item for item in summaries if item and item.get("tag") == "split_iter_00"), None)
    figures = []
    try:
        figures = _bars(rows)
    except Exception as exc:
        figures = []
        figure_error = str(exc)
    else:
        figure_error = ""

    markov = by_name.get("markov")
    phased = by_name.get("phased")
    lines = []
    lines.append("# Progress report: split base/arm policy on a frozen test")
    lines.append("")
    lines.append("This run answers the two closed-loop failures (the base never parks, and the arm torque error swamps the 1 N·m holding signal) and the evaluation problems that made the earlier numbers hard to compare. The test list is fixed before any new training target is drawn. Every model below that has finished is scored on that same list. An episode that leaves the workspace or winds joint 1 or 3 is marked unstable and is not entered into the mean or the median.")
    lines.append("")
    lines.append("## What changed")
    lines.append("")
    lines.append("The base command is a function of the heading-frame target and the standoff. It is fit by a small network, separate from the arm. Exact wheel zeros are rare in a raw trajectory (the yaw command falls to zero across 0.05 rad), so mean squared error barely sees them. The base fit therefore mixes the logged rows with samples of the expert's own base law, including the deadzone. A depth-6 decision tree, fit on those same labels, gates the wheels to exactly zero inside that box. A 32-unit tanh classifier did not: after 200 Adam steps it still left about 15% of the exact-zero region moving. The threshold is chosen for deadzone recall at a precision of at least 0.90, so the gate does not stop the base on the way in. A base-network command that is already under 0.02 rad/s is zeroed as well.")
    lines.append("")
    lines.append("The arm network's label is the torque with gravity removed. Gravity is added back at deployment and the sum is clipped to the expert's limit. The inputs add the cartesian error, the reach weight, and the gravity torque. Continuous joints 1, 3, 5 and 7 are sin/cos, not raw radians. Training rows in which joint 1 or 3 spins, and the quarter second before that, are dropped. Targets are drawn from about 0.25–1.25 m, with roughly one in eight slightly outside that band. About three quarters of episodes start from a perturbed base pose and arm configuration. DART noise is applied to the command the simulator receives. The label stays clean.")
    lines.append("")
    lines.append("The comparison network is one MLP on the same cleaned rows and the same arm features, predicting wheels and full torques together, with no deadzone gate. DAgger warm-starts the split networks, mixes in expert blocks early (0.75, then 0.35, then 0, in half-second blocks rather than at 1 kHz), and is accepted only when the frozen set improves. Sklearn early stopping is off. The offline number is an episode holdout, not a random slice of rows from the training episodes.")
    lines.append("")
    lines.append("## Frozen test")
    lines.append("")
    frozen_path = os.path.join(root, "frozen_targets.npz")
    if os.path.isfile(frozen_path):
        with np.load(frozen_path) as frozen:
            grid = np.asarray(frozen["grid"])
            held = np.asarray(frozen["heldout"])
        lines.append(
            f"The test is the original 24-point assessment grid (x, y in {{-1.2, 0, 1.2}}, z in {{0.25, 0.70, 1.15}}, footprint removed) plus {len(held)} held-out targets drawn once with seed 17. "
            f"Held-out heights run from {held[:, 2].min():.2f} m to {held[:, 2].max():.2f} m, "
            f"{int(np.sum((held[:, 2] < 0.25) | (held[:, 2] > 1.25)))} of them slightly outside the training band. "
            "Training rejects any draw within 8 cm of a frozen point. The phased expert is reported on the 24-point grid, which is where it was previously scored 24/24. Learned policies are reported on all 44."
        )
    else:
        lines.append("The frozen target file has not been written yet.")
    lines.append("")
    lines.append("A reach in the first column is the controller's own stop: for the Markov expert and the learned policies, the end effector must hold 10 mm for 0.5 s with the arm engaged, after 8 s. The millimetre columns are the final true error, and they are false when the run is unstable. Median error uses the stable runs only. First 10 mm is the first time the true error entered 10 mm, including runs that later left it. Stop time is the time of a successful stop, which includes the 0.5 s dwell. Unstable runs are not averaged into the error.")
    lines.append("")
    order = ["phased", "markov", "mono", "split_iter_00", "split_iter_01", "split_iter_02", "split_iter_03",
             "legacy_dagger", "legacy_all", "legacy_success"]
    lines.append(_table(rows, order))
    if figures:
        lines.append(f"![Success rates]({figures[0]})")
        lines.append("")
    elif figure_error:
        lines.append(f"The success-rate figure was not written ({figure_error}).")
        lines.append("")
    incomplete = [f"{row['controller']} ({row['n']} runs)" for row in rows if row["n"] < 44 and row["controller"] in order]
    if incomplete:
        lines.append("Still short of a full frozen set: " + ", ".join(incomplete) + ". Those rows are partial and are not used to pick the deployed weights.")
        lines.append("")
    lines.append("## Hand-coded quantities, and why the Markov expert replaces the phased one")
    lines.append("")
    lines.append("At deployment the network does not invent the integrator. The following are the expert's own formulas, the same ones used while logging:")
    lines.append("")
    lines.append("- The integrator state z, updated as z ← clip(w · (z + e dt), ±z_max). w = 0 clears it.")
    lines.append("- The cartesian error e and the reach weight w, from forward kinematics and the parking geometry. The arm network receives both. w is also what tells the stop rule that the arm is engaged.")
    lines.append("- The standoff distance, interpolated from the target height. The base network receives it, along with the standoff error and the bearing of the target. Those two scalars are the inputs of the expert's base law.")
    lines.append("- Gravity torque. The arm network predicts the torque with this term removed, and the term is added back before the clip.")
    lines.append("")
    lines.append("This is a deliberate replacement of the phased expert, not a hidden copy of its internal phase variable. The Markov expert has no phases. On the original grid it is the right object to imitate only if it matches the phased expert there.")
    lines.append("")
    if phased is not None or markov is not None:
        lines.append(_table(rows, ["phased", "markov"]))
        if phased is not None and markov is not None and phased["n_grid"] >= 24 and markov["n_grid"] >= 24:
            lines.append(
                f"On the runs scored so far, the phased expert reached {phased['n_reached']}/{phased['n']} "
                f"and the Markov expert reached {markov['n_reached']}/{markov['n']}. "
                "The phased row is the 24-point grid. The Markov row includes the held-out targets when those have been run."
            )
            lines.append("")
    else:
        lines.append("The grid comparison has not been run yet.")
        lines.append("")
    if markov is not None:
        lines.append(
            f"When the Markov expert has parked, the median base speed over the last 0.5 s of its stable runs is "
            f"{_fmt(markov['median_base_speed_cm_s'], 2)} cm/s and the median absolute wheel command is "
            f"{_fmt(markov['median_wheel_abs_tail'], 3)} rad/s. "
            "The failure this addresses was a learner that still commanded about 0.45 rad/s in the expert's exact-zero region and drifted at about 1.7 cm/s, which a 10 mm / 0.5 s dwell cannot survive."
        )
        lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(_dataset_section(root))
    lines.append("## Models")
    lines.append("")
    if split_summary is None:
        lines.append("The split policy has not been trained yet.")
    else:
        hold = split_summary.get("holdout") or {}
        park = split_summary.get("park_stats") or {}
        lines.append(
            f"The split policy has {split_summary.get('n_params_split', '—')} MLP parameters "
            f"({split_summary.get('base_inputs', 8)} base inputs, a 32×32 tanh base net; "
            f"{split_summary.get('arm_inputs', 37)} arm inputs, a 128×128 tanh arm net). "
            f"The deadzone gate is a depth-{park.get('max_depth', 6)} decision tree "
            f"({park.get('n_nodes', '—')} nodes), not another MLP. "
            "A 32-unit tanh classifier, trained for 200 steps on the same labels, still left about 15% of the exact-zero region commanding a non-zero wheel speed. The tree is fit on those same labels, including the fade, so the cut can sit on the expert's box. "
            f"The monolithic comparison has {split_summary.get('n_params_mono', '—')} parameters. "
            f"The earlier single network was about 20.6k parameters on 22 raw inputs. "
            f"Base training added {split_summary.get('n_synthetic_base', '—')} samples of the closed-form base law. "
            f"Adam ran {split_summary.get('base_n_iter', '—')} epochs on the base and {split_summary.get('arm_n_iter', '—')} on the arm, "
            f"with early stopping off, in {split_summary.get('train_seconds', 0):.0f} s."
        )
        lines.append("")
        if park:
            lines.append(
                f"The deadzone threshold on the tune slice is {park.get('threshold', 0):.2f}, "
                f"with recall {park.get('tune_recall', 0):.2f} and precision {park.get('tune_precision', 0):.2f}. "
                "That slice is 20% of the base rows, including the synthetic deadzone, and it is not the frozen test."
            )
            lines.append("")
        if hold:
            lines.append(
                f"Episode-holdout, not used for the deployed weights: gated wheel RMSE {_fmt(hold.get('gated_wheel_rmse'))} rad/s "
                f"(deadzone {_fmt(hold.get('gated_deadzone_rmse'))} rad/s, moving {_fmt(hold.get('gated_moving_rmse'))} rad/s), "
                f"arm residual RMSE {_fmt(hold.get('arm_residual_rmse'))} N·m, "
                f"on {hold.get('n', 0)} rows. "
                f"Deadzone recall on those rows is {_fmt(hold.get('park_recall_on_rows'), 2)} and the false-park rate on moving rows is {_fmt(hold.get('park_false_on_moving'), 2)}. "
                "This is not an R² from a random 10% of rows inside the training episodes. Closed-loop numbers above are the ones to quote."
            )
            lines.append("")
    lines.append("## Which half fails: the 2×2")
    lines.append("")
    lines.append("Expert base with learned arm, and learned base with expert arm, against expert/expert (Markov) and learned/learned (the behaviour-cloning split, before DAgger). This is the diagnostic for which half to keep working on.")
    lines.append("")
    lines.append(_table(rows, ["markov", "expert_base", "learned_base", "split_iter_00"]))
    lines.append("## DAgger")
    lines.append("")
    lines.append("Rollout medians below are the 8 targets drawn for that iteration. They are a new draw every iteration, so they are not a learning curve. The learning curve is the frozen-set table above, one row per saved iteration, on the same 44 targets.")
    lines.append("")
    if not dagger:
        lines.append("DAgger has not recorded an iteration yet.")
    else:
        lines.append("| Iteration | Expert-block probability | Rollouts reached | Unstable | Rollout median (stable) | New rows |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in dagger:
            lines.append(
                f"| {row.get('iteration', '—')} | {float(row.get('beta', 0)):.2f} | "
                f"{row.get('n_reached', '—')}/{row.get('n_rollouts', '—')} | "
                f"{row.get('n_unstable', '—')} | {_fmt(row.get('rollout_median_error'))} m | {row.get('n_rows', '—')} |"
            )
        lines.append("")
        lines.append("Iteration 4 of the previous run got worse (0.52 m to 0.74 m) on a fresh draw of 10 targets and was still the one a reader would have assumed was best. Here a later iteration is deployed only if the frozen-set rule says so.")
    lines.append("")
    if selection and selection.get("source") and selection.get("rule"):
        lines.append(
            f"The deployed file is `{selection['source']}` (`{selection['controller']}`). "
            f"Rule: {selection['rule']}"
        )
    elif selection:
        tag = selection.get("tag") or selection.get("controller") or "the behaviour-cloning split"
        reason = selection.get("reason") or "It has not yet been compared with later iterations on the full frozen set."
        lines.append(
            f"The file currently pointed at by `winner.joblib` is `{tag}`. {reason} "
            "A later iteration replaces it only after the full 44-point comparison."
        )
    else:
        lines.append("No full-set learned policy has been selected yet. Until then the demo falls back to the behaviour-cloning split if that file exists, and otherwise to the earlier 22-input network.")
    lines.append("")
    lines.append("## Robustness, size, and step time")
    lines.append("")
    lines.append("Robustness is the eight grid points at z = 0.70 m. Noise is added to the measurements the controller sees (joint angle 0.02 rad, joint velocity 0.05 rad/s, base position 1 cm, yaw 0.02 rad), not to the scored pose. The push is 50 N on the base for 0.5 s at t = 2 s. The payload adds 0.5 kg to the end-effector link.")
    lines.append("")
    robust_names = [name for name in by_name if name.endswith("-noise") or name.endswith("-push") or name.endswith("-payload")]
    if robust_names:
        lines.append(_table(rows, robust_names))
    else:
        lines.append("The robustness runs have not been recorded yet.")
        lines.append("")
    if timing:
        lines.append(
            f"One command takes {_fmt(timing.get('policy_ms'), 3)} ms for the deployed policy and {_fmt(timing.get('expert_ms'), 3)} ms for one expert call "
            f"({timing.get('repeats', 0)} calls after warmup, single sample). "
            f"The policy figure includes the hand-coded error, reach weight, and gravity. The expert figure includes inverse kinematics. "
            f"Parameter count of the timed policy: {timing.get('n_params', '—')}."
        )
    else:
        lines.append("Step time has not been measured yet.")
    lines.append("")
    lines.append("## What is not claimed")
    lines.append("")
    lines.append("The base network sees samples drawn from the expert's base law, not only from logged trajectories. That is disclosed because the law does not depend on the arm, and because the exact-zero set is otherwise about 1% of the labels. The arm is learned from the logged and DAgger rows only.")
    lines.append("")
    lines.append("z, e, w, the standoff, and gravity are not network outputs. A report that quotes these policies as mapping raw joint angles to torques with no side information would be wrong. They map the Markov observation, with sin/cos on the continuous joints, plus those quantities, to the commands.")
    lines.append("")
    lines.append("The frozen test always starts from the default pose. Perturbed initial conditions are in the training distribution, and in the DAgger rollouts, so the policy is asked to generalise to them during data collection. They are not part of the 44-point score.")
    lines.append("")
    lines.append("Offline RMSE is not the success rate. The previous 0.98–0.99 R² was sklearn holding out random rows from episodes that were also in the training fold. That number is not reported here.")
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("From the RoboEnv checkout, in the pixi environment:")
    lines.append("")
    lines.append("```")
    lines.append("pixi run python -u ../final/robotnik_arm_task/overnight_v2.py")
    lines.append("```")
    lines.append("")
    lines.append("The script resumes. Collection skips episodes already on disk, training skips a weights file that exists, and the frozen test skips a target it has already scored. `progress_report.md` is rewritten as stages finish. The demo (`robotnik_arm_task.py`, controller MLP) loads `markov_v2/winner.joblib` when that file is a split or monolithic policy, and otherwise the earlier `markov_mlp_cv/tanh-128x128.joblib`.")
    lines.append("")
    text = "\n".join(lines)
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, dest)
    return dest


if __name__ == "__main__":
    select_winner(ROOT)
    path = write_report()
    print(path, flush=True)
