# The Learning Robot Story

Robots run on equations: physics tells you what torque moves an arm. But the equations
are never quite right. Masses are uncertain, friction is messy, the best controller
settings are unknown, and some behaviours are too complicated to write down at all.
*Machine Learning for Robotics* is about filling those gaps with data. This tutorial
tells that story in six chapters, each one learning something harder than the last.

```mermaid
flowchart LR
    A["1 · Workbench<br/>The robot in<br/>the computer"] --> B["2 · Fit the physics<br/>Learn the robot's<br/>hidden numbers"]
    B --> C["3 · Beyond the formula<br/>Learn what physics<br/>leaves out"]
    C --> D["4 · Learn to tune<br/>Find the best settings<br/>in few tries"]
    D --> E["5 · Learn to act<br/>Copy an expert<br/>controller"]
    E --> F["6 · Where next<br/>Trial and error,<br/>sim-to-real"]
```

| Chapter | The question it answers | Labs |
|---|---|---|
| [1 · Workbench](1-workbench.md) | How do we simulate and control a robot from Python? | Week 1 |
| [2 · Fit the physics](2-fit-the-physics.md) | Can data reveal the robot's masses and inertias? | Weeks 1–2 |
| [3 · Beyond the formula](3-beyond-the-formula.md) | What if the physics formula itself is incomplete? | Week 2 |
| [4 · Learn to tune](4-learn-to-tune.md) | How do we find good controller gains when every test is expensive? | Week 3 |
| [5 · Learn to act](5-learn-to-act.md) | Can a network learn to drive and reach like an expert? | Final task |
| [6 · Where next](6-where-next.md) | Learning by trial and error, and crossing from simulation to reality | — |

## One idea ties it together

Every model you build sits somewhere on a line from "all physics" to "all data":

```mermaid
flowchart LR
    W["⬜ White box<br/>physics equations, few unknowns<br/><i>little data · extrapolates well</i>"] --> G["◧ Grey box<br/>physics + a learned correction<br/><i>the best of both, often</i>"] --> B["⬛ Black box<br/>a network learns everything<br/><i>lots of data · extrapolates badly</i>"]
```

Chapter 2 lives at the white end, Chapter 3 moves to grey, and Chapters 5–6 go black.
Knowing *where* to sit for a given problem is the real skill.

## How each lesson works

```mermaid
flowchart LR
    S["📖 The story<br/>intuition + pictures"] --> L["🔎 Look up<br/>search terms + best sources"] --> P["🛠 Prove it<br/>a build with a pass mark"]
```

Lessons give you the *shape* of an idea, not the full detail. **Look up** lists what to
search for and where to read. **Prove it** is a project that only counts as passed when a
number says so.

## Pace

```mermaid
timeline
    title Fitting the block module
    Week 1 : Chapter 1 : Chapter 2 starts
    Week 2 : Chapter 2 finishes : Chapter 3
    Week 3 : Chapter 4
    Final task : Chapter 5
    Afterwards : Chapter 6 and all the Prove-it projects
```

## Where the labs fit

```mermaid
flowchart LR
    W1a["week1 · induction_simulator"] --> C1["Ch 1"]
    W1b["week1 · cartesian_kinematic_controller"] --> C1
    W1c["week1 · dynamic_regression"] --> C2["Ch 2"]
    W2["week2 · dynamic_regression_nonlinear"] --> C2
    W2 --> C3["Ch 3"]
    W3["week3 · BO_for_pid_tuning"] --> C4["Ch 4"]
    F["final · robotnik_arm_task"] --> C5["Ch 5"]
```

## Ground rules

* **Integrity.** The labs and the final task are assessed. This tutorial explains the
  ideas behind every TODO but never fills them in, and Chapter 5 deliberately doesn't
  prescribe a model for the final task. Do the "Prove it" projects in your own
  workspace, *after* the related lab, and check course policy before publishing.
* **Tools.** Everything runs in the course's
  [RoboEnv](https://github.com/VModugno/RoboEnv) environment (pixi), using
  pybullet for physics and Pinocchio for the robot model.
* **Method.** Plot the data before fitting it → fit something simple → test on data the
  model never saw → only then reach for something fancier.

## The bookshelf

* **Bishop, *Pattern Recognition and Machine Learning*** (free from Microsoft
  Research): regression, kernels, probabilistic thinking.
* **James et al., *An Introduction to Statistical Learning*** (free at
  [statlearning.com](https://www.statlearning.com/)): the friendliest route to
  regression, trees and model checking.
* **Rasmussen & Williams, *Gaussian Processes for Machine Learning*** (free at
  [gaussianprocess.org/gpml](https://gaussianprocess.org/gpml/)).
* **Lynch & Park, *Modern Robotics*** (free, with videos), for the robot physics:
  [hades.mech.northwestern.edu](https://hades.mech.northwestern.edu/index.php/Modern_Robotics).

Start here → [Chapter 1 · Workbench](1-workbench.md)
