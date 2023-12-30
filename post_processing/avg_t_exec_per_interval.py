"""
    compute the avg. value of a quantity, e.g. execution time per time step, in a defined interval with respect
    to different solver settings
"""
import torch as pt
import matplotlib.pyplot as plt

from os.path import join
from os import path, makedirs
from typing import Union, Tuple, List

from post_processing.get_residuals_from_log import get_GAMG_residuals


def get_avg_min_t_exec_vs_settings(data: list, interval: int = 10, key: str = "exec_time",
                                   sf: Union[int, float] = 1) -> Tuple[List, List]:
    """
    compute the avg. value of a quantity, e.g. execution time per time step, in a defined interval with respect
    to different solver settings

    :param data: list with the filtered log data for each solver setting
    :param interval: how many dt should be in the interval for which the percentage is computed
    :param key: key of the dict for which parameter the comparison should be executed
    :param sf: scaling factor used to non-dimensionalize the numerical time
    :return: lists containing the avg. execution time for each interval and the corresponding non-dimensionalized dt's
    """
    # assuming all cases have the same amount of dt (required anyway for comparison)
    num_time = [d["time"] for d in data]

    # remove everything but the target quantity (by default the min. exec time per time step)
    data = [d[key] for d in data]

    # loop over all dt and compute the amount of settings yielding the fastest execution time
    t_exec, t_cfd = [], []
    for idx, d in enumerate(data):
        # loop over all cases and determine which case yield the min. exec time per dt for the current interval
        tmp = pt.tensor([[pt.tensor(d[j:j+interval]).mean(), pt.tensor(num_time[idx][j:j+interval]).mean()] for j in
                        range(0, len(d), interval)])
        t_cfd.append(tmp[:, 1] / sf)
        t_exec.append(tmp[:, 0] / sf)

    # estimate the hypothetical exec time optimal settings would yield (based on avg. exec time in each interval), note:
    # the accuracy of this estimation increases with decreasing interval size
    if key == "exec_time":
        # we assume that the optimal yield the min. number of time steps as well
        min_dt = min([len(j) for j in t_exec])

        # take the solver setting yielding the min. exec time in each interval and print results
        tmp = pt.stack([case[:min_dt] for case in t_exec]).min(dim=0)[0].sum()

        # compare that to the other settings
        print("possible savings (for optimal settings):")
        for j, case in enumerate(t_exec):
            print(f"\tcompared to case {j}: {round((tmp / case.sum()).item(), 4)}")

    return t_exec, t_cfd


if __name__ == "__main__":
    # path to the simulation results and save path for plots
    load_path = join("..", "run", "parameter_study", "influence_solver_settings", "smoother")
    save_path = join(load_path, "plots", "mixerVesselAMI", "plots_latex")

    # the names of the directories of the simulations
    cases = ["mixerVesselAMI_FDIC", "mixerVesselAMI_DICGaussSeidel", "mixerVesselAMI_GaussSeidel"]

    # legend entries for the plot
    legend = ["$no$", "$yes$"]

    # save name for the plot
    save_name = "avg_min_t_exec"

    # factor for making the numerical time dimensionless
    # mixerVesselAMI
    factor = 1 / 1.6364

    # surfaceMountedCube
    # factor = 1 / 0.15

    # cylinder2D
    # factor = 1 / 20

    # weirOverflow
    # factor = 1 / 0.4251

    # load the filtered log data for each case
    log_data = []
    for c in cases:
        load_path_tmp = join(load_path, c, "log_data_filtered.pt")

        try:
            # check if log file was already processed
            log_data.append(pt.load(load_path_tmp))
        except FileNotFoundError:
            # adjust load path
            load_path_tmp = join(load_path_tmp.split("log_data_filtered")[0])

            # else filter log file wrt GAMG & save the data from the log file
            pt.save(get_GAMG_residuals(load_path_tmp), join(load_path_tmp, "log_data_filtered.pt"))
            log_data.append(pt.load(join(load_path_tmp, "log_data_filtered.pt")))

    # compute the avg. execution time within a specified time interval with respect to the loaded cases
    avg_min_exec_times, avg_time_step = get_avg_min_t_exec_vs_settings(log_data, sf=factor)

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})

    # plot the results
    fig, ax = plt.subplots(figsize=(6, 3))
    for i in range(len(cases)):
        ax.plot(avg_time_step[i], avg_min_exec_times[i], label=legend[i])
    ax.set_xlabel(r"$t \, / \, T$", fontsize=12)
    ax.set_ylabel(r"$t_{exec}\, / \, T$", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.legend(loc="upper center", framealpha=1.0, ncol=3)
    plt.savefig(join(save_path, f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
