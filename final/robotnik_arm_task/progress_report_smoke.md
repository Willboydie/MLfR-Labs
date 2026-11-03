# Progress report: split base/arm policy on a frozen test

This run answers the two closed-loop failures (the base never parks, and the arm torque error swamps the 1 N·m holding signal) and the evaluation problems that made the earlier numbers hard to compare. The test list is fixed before any new training target is drawn. Every model below that has finished is scored on that same list. An episode that leaves the workspace or winds joint 1 or 3 is marked unstable and is not entered into the mean or the median.

## What changed

The base command is a function of the heading-frame target and the standoff. It is fit by a small network, separate from the arm. Exact wheel zeros are rare in a raw trajectory (the yaw command falls to zero across 0.05 rad), so mean squared error barely sees them. The base fit therefore mixes the logged rows with samples of the expert's own base law, including the deadzone, and a second small network gates the wheels to exactly zero when it classifies that deadzone. The threshold is chosen for deadzone recall at a precision of at least 0.90, so the gate does not stop the base on the way in.

The arm network's label is the torque with gravity removed. Gravity is added back at deployment and the sum is clipped to the expert's limit. The inputs add the cartesian error, the reach weight, and the gravity torque. Continuous joints 1, 3, 5 and 7 are sin/cos, not raw radians. Training rows in which joint 1 or 3 spins, and the quarter second before that, are dropped. Targets are drawn from about 0.25–1.25 m, with roughly one in eight slightly outside that band. About three quarters of episodes start from a perturbed base pose and arm configuration. DART noise is applied to the command the simulator receives. The label stays clean.

The comparison network is one MLP on the same cleaned rows and the same arm features, predicting wheels and full torques together, with no deadzone gate. DAgger warm-starts the split networks, mixes in expert blocks early (0.75, then 0.35, then 0, in half-second blocks rather than at 1 kHz), and is accepted only when the frozen set improves. Sklearn early stopping is off. The offline number is an episode holdout, not a random slice of rows from the training episodes.

## Frozen test

The test is the original 24-point assessment grid (x, y in {-1.2, 0, 1.2}, z in {0.25, 0.70, 1.15}, footprint removed) plus 20 held-out targets drawn once with seed 17. Held-out heights run from 0.11 m to 1.15 m, 4 of them slightly outside the training band. Training rejects any draw within 8 cm of a frozen point. The phased expert is reported on the 24-point grid, which is where it was previously scored 24/24. Learned policies are reported on all 44.

A reach in the first column is the controller's own stop: for the Markov expert and the learned policies, the end effector must hold 10 mm for 0.5 s with the arm engaged, after 8 s. The millimetre columns are the final true error, and they are false when the run is unstable. Median error and median time use the stable runs only. Time to reach, in the last column, is the time of the successful stop.

| Controller | Reached | 10 mm | 25 mm | 50 mm | 100 mm | Unstable | Median error | Median time to reach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phased | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0.010 m | 25.1 s |
| markov | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0.004 m | 21.9 s |
| mono | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | — m | — s |
| split_iter_00 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | — m | — s |

Still short of a full frozen set: markov (1 runs), mono (1 runs), phased (1 runs), split_iter_00 (1 runs). Those rows are partial and are not used to pick the deployed weights.

## Hand-coded quantities, and why the Markov expert replaces the phased one

At deployment the network does not invent the integrator. The following are the expert's own formulas, the same ones used while logging:

- The integrator state z, updated as z ← clip(w · (z + e dt), ±z_max). w = 0 clears it.
- The cartesian error e and the reach weight w, from forward kinematics and the parking geometry. The arm network receives both. w is also what tells the stop rule that the arm is engaged.
- The standoff distance, interpolated from the target height. The base network receives it, along with the standoff error and the bearing of the target. Those two scalars are the inputs of the expert's base law.
- Gravity torque. The arm network predicts the torque with this term removed, and the term is added back before the clip.

This is a deliberate replacement of the phased expert, not a hidden copy of its internal phase variable. The Markov expert has no phases. On the original grid it is the right object to imitate only if it matches the phased expert there.

| Controller | Reached | 10 mm | 25 mm | 50 mm | 100 mm | Unstable | Median error | Median time to reach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phased | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0.010 m | 25.1 s |
| markov | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0.004 m | 21.9 s |

When the Markov expert has parked, the median base speed over the last 0.5 s of its stable runs is 0.52 cm/s and the median absolute wheel command is 0.466 rad/s. The failure this addresses was a learner that still commanded about 0.45 rad/s in the expert's exact-zero region and drifted at about 1.7 cm/s, which a 10 mm / 0.5 s dwell cannot survive.

## Data

The training draw is 2 targets, of which 2 have been logged. Heights run from 0.66 m to 1.20 m, and 0 of the 2 (0%) sit slightly outside 0.25–1.25 m. None of them is a frozen test point.

Of the logged episodes, 2 reached under the expert's own 10 mm dwell, 0 were cut when joint 1 or 3 spun, and 2 started from a perturbed base and arm (0 started at the default pose). The spin cut removed 0 of 50829 logged rows. DART noise on the plant command is 0.15 rad/s on the wheels and 1.0 N·m on the torques. The stored label is the clean expert command.

Unlimited joints, from the simulator limits, are [0, 2, 4, 6] (0-based: joints 1, 3, 5 and 7 when this is 0, 2, 4, 6). Those angles enter the network as sin and cos. An episode holdout of 0 expert episodes is used only for the offline RMSE. The deployed weights are fit on every expert episode.

## Models

The split policy has 23689 MLP parameters (8 base inputs, a 32×32 tanh base net; 37 arm inputs, a 128×128 tanh arm net). The deadzone gate is a depth-6 decision tree (11 nodes), not another MLP. A 32-unit tanh classifier, trained for 200 steps on the same labels, still left about 15% of the exact-zero region commanding a non-zero wheel speed. The tree is fit on those same labels, including the fade, so the cut can sit on the expert's box. The monolithic comparison has 22537 parameters. The earlier single network was about 20.6k parameters on 22 raw inputs. Base training added 8500 samples of the closed-form base law. Adam ran 4 epochs on the base and 4 on the arm, with early stopping off, in 5 s.

The deadzone threshold on the tune slice is 0.05, with recall 1.00 and precision 1.00. That slice is 20% of the base rows, including the synthetic deadzone, and it is not the frozen test.

## Which half fails: the 2×2

Expert base with learned arm, and learned base with expert arm, against expert/expert (Markov) and learned/learned (the behaviour-cloning split, before DAgger). This is the diagnostic for which half to keep working on.

| Controller | Reached | 10 mm | 25 mm | 50 mm | 100 mm | Unstable | Median error | Median time to reach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| markov | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0.004 m | 21.9 s |
| expert_base | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | — m | — s |
| learned_base | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 | 0.063 m | — s |
| split_iter_00 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | — m | — s |

## DAgger

Rollout medians below are the 8 targets drawn for that iteration. They are a new draw every iteration, so they are not a learning curve. The learning curve is the frozen-set table above, one row per saved iteration, on the same 44 targets.

DAgger has not recorded an iteration yet.

The deployed file is `split_iter_00.joblib` (`split_iter_00`). Rule: Provisional, before any frozen-set comparison.

## Robustness, size, and step time

Robustness is the eight grid points at z = 0.70 m. Noise is added to the measurements the controller sees (joint angle 0.02 rad, joint velocity 0.05 rad/s, base position 1 cm, yaw 0.02 rad), not to the scored pose. The push is 50 N on the base for 0.5 s at t = 2 s. The payload adds 0.5 kg to the end-effector link.

The robustness runs have not been recorded yet.

Step time has not been measured yet.

## What is not claimed

The base network sees samples drawn from the expert's base law, not only from logged trajectories. That is disclosed because the law does not depend on the arm, and because the exact-zero set is otherwise about 1% of the labels. The arm is learned from the logged and DAgger rows only.

z, e, w, the standoff, and gravity are not network outputs. A report that quotes these policies as mapping raw joint angles to torques with no side information would be wrong. They map the Markov observation, with sin/cos on the continuous joints, plus those quantities, to the commands.

The frozen test always starts from the default pose. Perturbed initial conditions are in the training distribution, and in the DAgger rollouts, so the policy is asked to generalise to them during data collection. They are not part of the 44-point score.

Offline RMSE is not the success rate. The previous 0.98–0.99 R² was sklearn holding out random rows from episodes that were also in the training fold. That number is not reported here.

## How to reproduce

From the RoboEnv checkout, in the pixi environment:

```
pixi run python -u ../final/robotnik_arm_task/overnight_v2.py
```

The script resumes. Collection skips episodes already on disk, training skips a weights file that exists, and the frozen test skips a target it has already scored. `progress_report.md` is rewritten as stages finish. The demo (`robotnik_arm_task.py`, controller MLP) loads `markov_v2/winner.joblib` when that file is a split or monolithic policy, and otherwise the earlier `markov_mlp_cv/tanh-128x128.joblib`.
