# Markov MLP datasets

Two datasets were cut from the 150 saved Markov episodes. **All** keeps every trajectory (89 reaches, 61 misses). **Success** keeps the 89 episodes that finished inside 10 mm.

Each set was scored with the same 5-fold episode holdout as the first search: samples every 25 ms, training episodes shuffled after the split, score equal to mean standardised MSE. Training size was again a factor (25, 55, and every episode in the training fold: 120 on All, 71 or 72 on Success). The original six networks stayed in the table. Around tanh-128×128, three settings the first search had left fixed were each moved on their own: learning rate (1e-4, 1e-2), L2 (1e-5, 1e-3), and batch size (64, 1024).

## Cross-validation

On **All**, the best row is tanh-128×128 with batch 64, trained on the full fold. Standardised MSE is 0.0375 ± 0.0230, wheel RMSE 0.235 rad/s, torque RMSE 1.27 N·m. The original tanh network at batch 256 is 0.0388 ± 0.0230. L2 at 1e-5 and 1e-3 land on that same number. Learning rate 1e-4 and 1e-2, and batch 1024, are worse (0.049 to 0.051). The fold spread is about 0.023, so batch 64 is not a real separation from the previous winner. What does separate is the amount of data: the best 55-episode row is 0.064, and 25-episode rows are several times worse. Tanh still beats ReLU once misses are included.

On **Success**, the best row is ReLU-128×128 with L2 1e-2, again using the full fold. Standardised MSE is 0.158 ± 0.097, wheel RMSE 0.378 rad/s, torque RMSE 0.68 N·m. Plain ReLU-128×128 is 0.160. The tanh variants, including the learning-rate and batch moves, sit at 0.171 to 0.195. The gap between those ReLU networks is much smaller than the fold spread. Torque error is lower than on All because the successful episodes contain less of the large driving commands; wheel error is higher. The two standardised scores are not on the same scale, because each fold standardises with its own training labels.

The deployed weights use the winning settings on every episode of that dataset (150 and 89), since the lattice was not part of either set.

## DAgger

The All network was then trained with DAgger. Iteration 0 is that tanh-128×128, batch-64 policy. Each of the next three iterations rolls the current network out on 10 new targets, seven in the original 2 m box and three in the near square. The lattice is not among them. The plant follows the network. At the states it visits, the label is the Markov command, and the integrator is updated the same way as at playback. Rows are kept every 25 ms. The network is retrained from scratch on the expert episodes plus every DAgger row so far.

On those rollouts the network never reached. Median error fell from 1.17 m to 1.00 m to 0.52 m. Mean absolute disagreement with the expert fell from 0.93 rad/s and 12.3 N·m to 0.57 rad/s and 5.9 N·m. The third iteration added 16,010 rows, 182,949 in total.

## Lattice

The test is the 3×3×3 grid of a 2×2×1 m box centred at (0, 0, 0.5): x and y in {−1, 0, 1}, z in {0, 0.5, 1}. The same 27 points were run headless for the Markov expert, both cross-validation winners, and the DAgger network. A reach is the expert's own 10 mm stop.

| Controller | Reached | Median error | Mean error | Max error |
| --- | ---: | ---: | ---: | ---: |
| Markov expert | 18/27 | 0.008 m | 0.014 m | 0.031 m |
| DAgger, tanh batch 64 | 0/27 | 0.57 m | 0.56 m | 1.18 m |
| All, tanh batch 64 | 0/27 | 0.88 m | 0.80 m | 1.47 m |
| Success, ReLU L2 1e-2 | 0/27 | 60 m | 214 m | 1710 m |

The expert reached every target at z = 0.5 and z = 1.0. All nine targets on the floor (z = 0) stopped near 30 mm and were marked not reachable. The All network never reached. It stayed in the workspace and ran out the 40 s cap about 0.7–1.5 m from the target. The Success network also never reached, and it did not stay in the workspace: closed-loop error grew to tens or hundreds of metres.

DAgger cuts the All network's median lattice error from 0.88 m to 0.57 m, and the mean from 0.80 m to 0.56 m. The gain is largest above the floor: median error at z = 1.0 m falls from 0.93 m to 0.32 m, and at z = 0.5 m from 0.72 m to 0.48 m. The closest miss is 93 mm. It still reaches none of the 27 points, so three iterations do not close the gap to the expert. It does stay in the workspace, unlike the Success network.

Tables and the fold logs are in `markov_mlp_cv_all/`, `markov_mlp_cv_success/`, `markov_dagger/`, and `markov_mlp_lattice/`.
