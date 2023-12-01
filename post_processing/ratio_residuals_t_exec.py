"""
    Compute the percentage of time steps, which yield the minimum execution time per time step in the specified
    interval with repsect to the solver settings. In case of variable time steps, the time steps are compared only up
    the last time step which is available for all cases.
"""
import torch as pt
import matplotlib.pyplot as plt

from typing import Union
from os.path import join
from os import path, makedirs

from post_processing.get_residuals_from_log import get_GAMG_residuals


def get_min_max_vs_t_exec(data: list, interval: int = 300, key: str = "exec_time",
                          sf: Union[int, float] = 1) -> pt.Tensor and list:
    """
    Compute the percentage of time steps, which yield the minimum execution time per time step in the specified
    interval with repsect to the solver settings. In case of variable time steps, the time steps are compared only up
    the last time step which is available for all cases.

    :param data: list with the filtered log data for each solver setting
    :param interval: how many dt should be in the interval for which the percentage is computed
    :param key: key of the dict for which parameter the comparison should be executed
    :param sf: scaling factor used to non-dimensionalize the numerical time
    :return: tensor with the percentage of time steps yielding the min t_exec and the non-dimensionalized time steps
    """
    # assuming all cases have the same amount of dt (required anyway for comparison)
    num_time = data[0]["time"]

    # remove everything but the target quantity (by default the min. exec time per time step)
    data = [d[key] for d in data]

    # determine min length
    global_min_len = min([len(j) for j in data])

    # loop over all dt and compute the amount of settings yielding the fastest execution time
    count, t = [], []
    for k in range(0, global_min_len, interval):
        # loop over all cases and determine which case yield the min. exec time per dt for the current interval
        tmp = [data[j][k:k+interval] for j in range(len(data))]
        # account for dt != const.
        min_len = min([len(j) for j in tmp])
        _, min_idx = pt.tensor([j[:min_len] for j in tmp]).min(dim=0)
        bins = pt.bincount(min_idx) / interval

        # in chase there are no occurrences for one or more cases in the current interval, add them as zero
        if len(bins) < len(data):
            all_bins = pt.zeros(len(data))
            all_bins[:len(bins)] = bins
            bins = all_bins

        count.append(bins)

        # avg. time for each interval, later used as x-label
        t.append(num_time[k+int(interval/2)] / sf)

    return pt.stack(count, dim=0), t


if __name__ == "__main__":
    # path to the simulation results and save path for plots
    load_path = join("..", "run", "parameter_study", "influence_solver_settings", "smoother")
    save_path = join(load_path, "plots", "surfaceMountedCube")

    # the names of the directories of the simulations
    cases = ["surfaceMountedCube_FDIC", "surfaceMountedCube_DICGaussSeidel", "surfaceMountedCube_GaussSeidel"]

    # legend entries for the plot
    legend = ["$FDIC$", "$DICGaussSeidel$", "$GaussSeidel$"]

    # save name for the plot
    save_name = "ratio_min_t_exec_smoother"

    # factor for making the numerical time dimensionless
    # mixerVesselAMI
    # factor = 1 / 1.6364

    # surfaceMountedCube
    factor = 1 / 0.15

    # cylinder2D
    # factor = 1 / 20

    # weirOverflow
    # factor = 1 / 0.4251

    # load the filtered log data for each case
    log_data = []
    for c in cases:
        if c.startswith("surfaceMountedCube"):
            load_path_tmp = join(load_path, c, "fullCase", "log_data_filtered.pt")
        else:
            load_path_tmp = join(load_path, c, "log_data_filtered.pt")

        try:
            # check if log file was already processed
            log_data.append(pt.load(load_path_tmp))
        except FileNotFoundError:
            # adjust load path
            load_path_tmp = join(load_path.split("log_data_filtered")[0])

            # else filter log file wrt GAMG & save the data from the log file
            pt.save(get_GAMG_residuals(join(load_path_tmp, c)), join(load_path_tmp, c, "log_data_filtered.pt"))
            log_data.append(pt.load(join(load_path_tmp, c, "log_data_filtered.pt")))

    # compute the relative amount of time steps yielding the minimum execution time with respect to the loaded cases
    ratio_min_exec_times, avg_time_step = get_min_max_vs_t_exec(log_data, sf=factor)

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})

    fig, ax = plt.subplots(figsize=(6, 3))
    for i in range(len(cases)):
        ax.plot(avg_time_step, ratio_min_exec_times[:, i], label=legend[i])
    ax.set_xlabel(r"$t \, / \, T$", fontsize=12)
    ax.set_ylabel(r"$N_{t, min}$   $[\%]$", fontsize=12)
    # ax.vlines(5.2, 0.2, 0.8, ls="-.", lw=2, color="red")
    # ax.vlines(3.6, 0.2, 0.8, ls="-.", lw=2, color="red")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.legend(loc="upper center", framealpha=1.0, ncol=3)
    plt.savefig(join(save_path, f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
