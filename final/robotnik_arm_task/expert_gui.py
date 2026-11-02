"""Pick a Cartesian target and watch the expert reach it in PyBullet.

The robot begins at the default pose from the task config each time you press GO.
Run it with the RoboEnv environment, from the repository root:

    cd RoboEnv
    pixi run python ../final/robotnik_arm_task/expert_gui.py
"""
import os
import sys
import tkinter as tk
from tkinter import ttk

import matplotlib
import numpy as np
import pybullet as p

matplotlib.use("Agg")

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import robotnik_arm_task as task

# This window is the viewer, so open the PyBullet GUI even if a headless env var is set.
task.headless = False

# World-frame limits (m) for the sliders. Typed values outside this are still accepted.
AXIS_LIMITS = {"X": (-2.0, 2.0), "Y": (-2.0, 2.0), "Z": (0.05, 1.50)}


class ExpertApp:
    """Target sliders plus a GO button. Playback happens in the PyBullet window."""

    def __init__(self):
        self.ctx = None
        self._running = False
        self._stop = False
        self._away_from_start = False
        self._updating_target = False

        self.root = tk.Tk()
        self.root.title("Expert playback")
        self._controller = tk.StringVar(self.root, value="Original")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Expert playback", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            frame,
            text=(
                "The robot starts in its default pose.\n"
                "Set the red target, choose a controller, then press GO.\n"
                "Drag in the PyBullet window to orbit the camera."
            ),
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 10))

        self._target_vars = {}
        self._scales = []
        self._spins = []
        for row, axis in enumerate("XYZ", start=2):
            lo, hi = AXIS_LIMITS[axis]
            var = tk.DoubleVar(value=float(task.TARGET_EE_POS["XYZ".index(axis)]))
            self._target_vars[axis] = var
            ttk.Label(frame, text=f"{axis} (m)").grid(row=row, column=0, sticky="w", pady=3)
            scale = ttk.Scale(frame, from_=lo, to=hi, variable=var, length=220)
            scale.grid(row=row, column=1, sticky="ew", padx=8)
            spin = ttk.Spinbox(
                frame, from_=lo, to=hi, increment=0.01, textvariable=var, width=6, format="%.2f"
            )
            spin.grid(row=row, column=2, sticky="e")
            self._scales.append(scale)
            self._spins.append(spin)

        ttk.Label(frame, text="Controller").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.controller_box = ttk.Combobox(
            frame,
            textvariable=self._controller,
            values=("Original", "Markov", "MLP"),
            state="disabled",
            width=18,
        )
        self.controller_box.grid(row=5, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.controller_box.bind("<<ComboboxSelected>>", lambda _event: self._on_expert_changed())

        self.go_button = ttk.Button(frame, text="GO", command=self._on_go, state="disabled")
        self.go_button.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(14, 8), ipady=4)

        self.status = tk.StringVar(value="Opening the PyBullet window…")
        ttk.Label(frame, textvariable=self.status, wraplength=340, justify="left").grid(
            row=7, column=0, columnspan=3, sticky="w"
        )

        self.root.bind("<Return>", lambda _event: self._on_go())
        self.root.after(100, self._open_sim)

    def _read_target(self):
        try:
            return np.array([self._target_vars[axis].get() for axis in "XYZ"], dtype=float)
        except (tk.TclError, ValueError):
            return None

    def _expert_name(self):
        return self._controller.get()

    def _controller_name(self):
        return {"Original": "phased", "Markov": "markov", "MLP": "mlp"}[self._controller.get()]

    def _set_inputs_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for widget in (*self._scales, *self._spins, self.go_button):
            widget.configure(state=state)
        self.controller_box.configure(state="readonly" if enabled else "disabled")

    def _idle_status(self):
        target = self._read_target()
        if target is None:
            return "Enter a numeric target in metres."
        return (
            f"{self._expert_name()}. Default start pose. "
            f"Target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) m. Press GO."
        )

    def _on_expert_changed(self):
        if self._running or self.ctx is None:
            return
        if self._away_from_start:
            self.status.set(
                f"{self._expert_name()} selected. "
                "Press GO to run again from the default start."
            )
            return
        self.status.set(self._idle_status())

    def _sync_marker(self):
        target = self._read_target()
        if target is not None and self.ctx is not None:
            task._place_target_marker(self.ctx, target)

    def _reset_robot(self):
        sim = self.ctx["sim"]
        sim.ResetPose()
        self.ctx["client"].resetBaseVelocity(self.ctx["rid"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        sim.ReceiveObservation()

    def _frame_camera(self):
        client = self.ctx["client"]
        client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        client.resetDebugVisualizerCamera(
            cameraDistance=3.2,
            cameraYaw=42,
            cameraPitch=-32,
            cameraTargetPosition=[0.0, 0.2, 0.4],
        )

    def _on_target_changed(self, *_args):
        if self._running or self._updating_target or self.ctx is None:
            return
        self._updating_target = True
        try:
            for var in self._target_vars.values():
                try:
                    value = float(var.get())
                except (tk.TclError, ValueError):
                    return
                snapped = round(value, 2)
                if abs(snapped - value) > 1e-8:
                    var.set(snapped)
            if self._away_from_start:
                self._reset_robot()
                self._away_from_start = False
            self._sync_marker()
            self.status.set(self._idle_status())
        finally:
            self._updating_target = False

    def _open_sim(self):
        try:
            self.ctx = task.setup_simulation()
        except Exception as exc:
            self.status.set(f"Could not open PyBullet: {exc}")
            return
        self._frame_camera()
        self._sync_marker()
        for var in self._target_vars.values():
            var.trace_add("write", self._on_target_changed)
        self._set_inputs_enabled(True)
        self.status.set(self._idle_status())
        self.root.after(100, self._pump_viewer)

    def _pump_viewer(self):
        """Keep the PyBullet window responsive while waiting for GO."""
        if self._stop or self.ctx is None:
            return
        if not self._running:
            try:
                self.ctx["client"].getKeyboardEvents()
            except Exception:
                self.status.set("The PyBullet window was closed.")
                self.ctx = None
                self._set_inputs_enabled(False)
                return
        if not self._stop:
            self.root.after(100, self._pump_viewer)

    def _on_step(self, current_time, ee_dist_error):
        if self._stop:
            return False
        self.status.set(
            f"{self._expert_name()}   t={current_time:.1f}s   EE error {ee_dist_error:.3f} m"
        )
        try:
            self.root.update()
        except tk.TclError:
            return False
        return not self._stop

    def _on_go(self):
        if self._running or self.ctx is None:
            return
        target = self._read_target()
        if target is None:
            self.status.set("Enter a numeric target in metres.")
            return

        self._running = True
        expert_name = self._expert_name()
        self._set_inputs_enabled(False)
        self.status.set(f"Running {expert_name} from the default start…")
        self.root.update_idletasks()
        try:
            result = task.run_episode(
                self.ctx,
                target,
                record=False,
                verbose=True,
                realtime=True,
                on_step=self._on_step,
                controller=self._controller_name(),
            )
        except Exception as exc:
            result = None
            self.status.set(f"Playback failed: {exc}")
        self._running = False

        if self._stop:
            self._shutdown()
            return

        self._set_inputs_enabled(True)
        if result is None:
            return
        self._away_from_start = True
        message = result["stop_message"] or "Finished"
        self.status.set(
            f"{expert_name}: {message} Final error {result['final_error']:.3f} m. "
            "Move the target or press GO to run again from the default start."
        )

    def _on_close(self):
        self._stop = True
        if not self._running:
            self._shutdown()

    def _shutdown(self):
        ctx = self.ctx
        self.ctx = None
        if ctx is not None:
            try:
                ctx["client"].disconnect()
            except Exception:
                pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    ExpertApp().root.mainloop()


if __name__ == "__main__":
    main()
