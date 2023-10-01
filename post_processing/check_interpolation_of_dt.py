"""
    compare the interpolated t_CPU at the dt of the trajectories with the original t_CPU at the dt of the base case in
    order to check if the interpolated t_CPU (used for rewards in DRL routine) are sufficiently accurate
"""
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from pandas import read_csv
from os import makedirs, path


def load_trajectories(load_dir: str) -> pt.Tensor:
    full_path = glob(join(load_dir, "postProcessing", "time", "*", "timeInfo.dat"))[0]
    _t_base = read_csv(full_path, sep="\t", comment="#", header=None, names=["t", "t_per_dt"], usecols=[0, 3])

    # convert to tensor in order to do computations later easier
    return pt.tensor(_t_base.values)


if __name__ == "__main__":
    base_path = join("..", "run", "drl", "base_weirOverflow")
    example_traj_path = join("..", "run", "drl", "combined_smoother_and_interpolateCorrection", "results_weirOverflow",
                             "random_policy", "run_1")

    save_path = join("..", "run", "drl", "combined_smoother_and_interpolateCorrection", "results_weirOverflow")

    # load the dt & CPU times of the base case
    t_base = load_trajectories(base_path)

    # load an example trajectory of a policy and make sure it has the same size ass the base case
    t_traj = load_trajectories(example_traj_path)[:t_base.size()[0], :]

    # interpolate the t_CPU of the base case at the dt of the trajectory -> interp(dt_traj, dt_base, t_CPU_base)
    t_cpu_base_interpolated = pt.from_numpy(np.interp(t_traj[:, 0], t_base[:, 0], t_base[:, 1]))

    # scaling factor for weirOverflow case
    factor = 1 / 0.4251

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # plot the results
    fig, ax = plt.subplots(figsize=(7, 4))

    # plot the original base case
    ax.plot(t_base[:, 0] / factor, t_base[:, 1], marker="o", fillstyle="none", color="black", label="$original$",
            ls=":")

    # plot the interpolated t_CPU of the base case wrt dt of trajectory
    ax.scatter(t_traj[:, 0] / factor, t_cpu_base_interpolated, marker="x", color="red", label="$interpolated$",
               zorder=10)

    ax.set_xlim(26, 26.1)
    ax.set_ylim(0.25, 0.65)
    ax.set_xlabel(r"$t \, / \, T$", fontsize=13)
    ax.set_ylabel(r"$t_{CPU} \, / \, \Delta t$   $[s]$", fontsize=13)
    fig.tight_layout()
    ax.legend(loc="upper left", framealpha=1.0, ncol=1)
    plt.savefig(join(save_path, "plots", "example_interpolation_t_cpu.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
