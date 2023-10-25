"""
    get the execution time per function object call ('agentSolverSettings') and plot it wrt time step avg. over multiple
    runs
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join

from scipy import signal
from scipy.ndimage import gaussian_filter1d
from torch.fft import fftfreq, fft


def get_execution_time(load_dir: str) -> list:
    with open(glob(join(load_dir, f"log.*Foam"))[0], "r") as file:
        data = file.readlines()

    # line we are looking for looks like this: '[agentSolverSettings]: execution time of function object was 43228ms'
    key = "[agentSolverSettings]: execution time of function object was"
    exec_times = [float(line.split(" ")[-1].strip("ms\n")) for line in data if line.startswith(key)]

    return exec_times


def fft(residual, n):
    pass


if __name__ == "__main__":
    # main path to all the cases and save path
    load_path = join("..", "run", "drl", "combined_smoother_interpolateCorrection_nFinestSweeps",
                     "results_cylinder2D")
    save_path = join(load_path, "plots")

    # names of top-level directory containing the simulations run with different settings
    case = "trained_policy_b16_PPO_every_2nd_dt_validation_every_dt_local"

    execution_times = []
    for log in glob(join(load_path, case, "*")):
        tmp = get_execution_time(log)

        # in case this simulation was done after implementing the timer, then we don't have an empty list
        if tmp:
            execution_times.append(tmp)

    # compute mean, std. deviation and sum of execution times (exec times are measured in microseconds)
    execution_times = pt.tensor(execution_times)
    print(f"Found {execution_times.size()[0]} cases. Execution times (mean per dt / 1 sigma per dt / avg. sum total /"
          f" 1 sigma sum total):\t{round(execution_times.mean().item() * 1e-6, 6)} s / "
          f"{round(execution_times.std().item() * 1e-6, 6)} s / "
          f"{round(execution_times.sum(dim=1).mean().item() * 1e-6, 3)} s /"
          f" {round(execution_times.sum(dim=1).std().item() * 1e-6, 3)} s")

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # plot the mean and std. deviation of execution times wrt dt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(execution_times.size()[1]), execution_times.mean(dim=0) * 1e-6, color="black",
            label=r"$\mu(t_{exec})$, $\sum t_{exec} =$ $"+str(round(execution_times.sum(dim=1).mean().item() * 1e-6, 2))
                  + "$ $s$")
    ax.set_xlabel(r"$\Delta t$ $no.$ $\#$")
    ax.set_ylabel("$t$   $[s]$")
    fig.tight_layout()
    ax.legend(loc="upper left", framealpha=1.0, ncol=1)
    plt.savefig(join(save_path, "execution_times_function_object_vs_dt.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")

    # do Fourier analysis (const. dt only)
    dt = 5.0e-5
    data = [signal.welch(execution_times[e, :] * 1e-6, 1/dt, nperseg=int(execution_times.size()[1] * 0.5),
                         nfft=execution_times.size()[1]) for e in range(execution_times.size()[0])]

    fig, ax = plt.subplots(figsize=(6, 3))

    # data[0] = frequency, data[1] = amplitude
    [ax.plot(data[0], data[1], label=f"$case$ ${i}$") for i, data in enumerate(data)]
    ax.set_xlabel(r"$f$   $[Hz]$")
    ax.set_ylabel(r"$PSD$")
    ax.set_xlim(0, 50)
    fig.tight_layout()
    ax.legend(loc="upper left", framealpha=1.0, ncol=1)
    plt.savefig(join(save_path, "PSD_execution_times_FO.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
