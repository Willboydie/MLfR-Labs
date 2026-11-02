import os
os.environ["ROBOENV_HEADLESS"] = "1"

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

def run_one(job):
    target, controller = job
    import robotnik_arm_task as task

    ctx = task.setup_simulation()
    task.max_steps = int(round(task.COMPARE_SIM_SECONDS / float(ctx["time_step"])))
    try:
        result = task.run_episode(
            ctx, np.asarray(target, dtype=float),
            record=False, verbose=False, realtime=False, controller=controller,
        )
    finally:
        ctx["client"].disconnect()
    return controller, target, result["final_error"], result["time"], result["stop_message"]

if __name__ == "__main__":
    import robotnik_arm_task as task

    jobs = [(target.tolist(), name) for target in task.COMPARE_TARGETS for name in ("phased", "markov")]
    workers = 4  # physical cores. Try 6 or 8 only if RAM stays free.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for controller, target, error, sim_t, message in pool.map(run_one, jobs):
            print(f"{controller:6} {np.round(target, 2)}  error {error:.4f} m at t={sim_t:.2f}s | {message}", flush=True)