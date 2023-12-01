"""
    this script used the filtered solver log file in order to go through all the parameters and compares for each time
    step which of the cases yields the min. / max. value of each parameter and plots the results.

    Note: in order to compare different simulations to each other, it is necessary that all simulations have exactly the
          same time steps. Otherwise, it is not possible to check for each time steps which of the settings yields the
          min. / max. value (without interpolating the time steps)
"""
import torch as pt
import matplotlib.pyplot as plt

from typing import Union
from os.path import join
from os import path, makedirs

from post_processing.get_residuals_from_log import get_GAMG_residuals


def get_min_max_vs_parameter(data: list) -> dict:
    """
    go through all the filtered log data and check for each time step which of the loaded solver settings / log data
    yields the min. / max. value, e.g. min. t_exec, convergence rate, ...

    It is assumed that all cases have the same dt and time steps, otherwise a comparison is not possible (without
    interpolating all time steps to one scale). In case this prerequisite is not given, this function exits with an
    error message.

    :param data: list with the filtered log data for each solver setting
    :return: a dictionary containing the mi. / max. values wrt the solver settings for each time step
    """
    data_out, tmp = {}, 0

    # loop over all keys, for each key determine which solver setting yields the min. / max. value
    for key in data[0].keys():
        # we only want to compare scalar values wrt time step such as exec time
        if type(data[0][key][0]) == float and key != "time":
            # take the current parameter for all solver settings and put them into a list, this only works if
            # dt = const., because we want to check for each dt which config. was faster / slower etc.
            try:
                tmp = pt.tensor([data[j][key] for j in range(len(data))])

            # if the dt was not the same for different settings, then exit because we can't compare the results directly
            except ValueError:
                print("The time step is expected to be constant for all simulations.")
                exit()

            # then determine the min. / max. values for each time step
            min_vals = pt.argmin(tmp, dim=0)
            max_vals = pt.argmax(tmp, dim=0)

            # re-sort the values wrt min / max values of solver settings, assuming all cases have same time steps
            t_min = [[] for _ in range(len(data))]
            t_max = [[] for _ in range(len(data))]
            val_min = [[] for _ in range(len(data))]
            val_max = [[] for _ in range(len(data))]

            for j, t in enumerate(data[0]["time"]):
                # sort the time steps
                t_min[min_vals[j]].append(t)
                t_max[max_vals[j]].append(t)

                # sort the values corresponding to the time step
                val_min[min_vals[j]].append(data[min_vals[j]][key][j])
                val_max[max_vals[j]].append(data[max_vals[j]][key][j])

            # assign re-sorted time step & values to new dict
            data_out[f"{key}_min"] = val_min
            data_out[f"{key}_max"] = val_max
            data_out[f"t_{key}_max"] = t_max
            data_out[f"t_{key}_min"] = t_min
        else:
            continue
    return data_out


def plot_results(x, y, save_dir: str, ylabel: str = "", xlabel: str = r"$t \, / \, T$", save_name: str = "",
                 legend_list: list = None, scaling_factor: Union[int, float] = 1, y_log: bool = True) -> None:
    """
    plot the solver settings which yielded the min. / max. value of some property regarding the GAMG solver wrt to time
    step

    :param x: numerical time steps
    :param y: corresponding min. / max. residual data wrt solver setting
    :param save_dir: directory where the plot should be saved to
    :param ylabel: label for the y-axis
    :param xlabel: label for the x-axis
    :param save_name: save name of the plot
    :param legend_list: list for legend entries, if wanted
    :param scaling_factor: factor for scaling the x-axis (non-dimensionalizing the time step)
    :param y_log: flag if the y-axis should be logarithmic or not
    :return: None
    """
    # scale the num. time with period length of dominant frequency in the flow field
    x = [[s / scaling_factor for s in simulation] for simulation in x]
    y = [[s / scaling_factor for s in simulation] for simulation in y] if ylabel.startswith("t") else y

    fig, ax = plt.subplots(figsize=(6, 3))
    for j in range(len(x)):
        if legend:
            ax.scatter(x[j], y[j], label=legend_list[j], marker=".")
        else:
            ax.scatter(x[j], y[j])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if y_log:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=3)
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # path to the simulation results and save path for plots
    load_path = join("..", "run", "parameter_study", "influence_solver_settings", "interpolateCorrection")
    save_path = join(load_path, "plots", "weirOverflow", "plots_latex")

    # the names of the directories of the simulations
    cases = ["weirOverflow_no", "weirOverflow_yes"]

    # legend entries for the plot
    legend = ["$no$", "$yes$"]

    # which parameters / properties of the residuals should be compared, if None, then all parameters will be compared
    params = ["exec_time_min"]

    # factor for making the time dimensionless; here the period of the dominant frequency in the flow field is used
    # for the surfaceMountedCube case
    # factor = 1 / 0.15

    # for cylinder2D case
    factor = 1 / 20

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
            load_path_tmp = join(load_path_tmp.split("log_data_filtered")[0])

            # else filter log file wrt GAMG & save the data from the log file
            pt.save(get_GAMG_residuals(load_path_tmp), join(load_path_tmp, "log_data_filtered.pt"))
            log_data.append(pt.load(join(load_path_tmp, "log_data_filtered.pt")))

    # for each time step, determine which configuration yielded min. / max. values, e.g. t_exec
    min_max_values = get_min_max_vs_parameter(log_data)

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})

    # plot the results
    if params is None:
        params = [k for k in min_max_values.keys() if not k.startswith("t_") and not k.startswith("init_residual_")]

    # labels for the plots
    """
    ylabels = [r"$min(t^*_{exec}) \, / \, T$", r"$max(t^*_{exec}) \, / \, T$",
               r"$min(\sum{N_{GAMG}})$", r"$max(\sum{N_{GAMG}})$",
               r"$min(N_{GAMG, \, max})$", r"$max(N_{GAMG, \, max})$",
               "$min(\\boldsymbol{R}_{0, max})$", "$max(\\boldsymbol{R}_{0, max})$",
               "$min(|\Delta \\boldsymbol{R}_{max}|)$", "$max(|\Delta \\boldsymbol{R}_{max}|)$",
               "$min(|\Delta \\boldsymbol{R}_{min}|)$", "$max(|\Delta \\boldsymbol{R}_{min}|)$",
               "$min(|\Delta \\boldsymbol{R}_{median}|)$", "$max(|\Delta \\boldsymbol{R}_{median}|)$"]
    """
    ylabels = [r"$min(t^*_{exec}) \, / \, T$"]

    # loop over the keys and plot the min- & max. values wrt solver setting
    for i, k in enumerate(params):
        if "residual" in k or "convergence_rate" in k:
            log_y = True
        else:
            log_y = False

        plot_results(min_max_values[f"t_{k}"], min_max_values[k], save_path, ylabels[i], y_log=log_y,
                     save_name=f"comparison_{k}_GS", legend_list=legend, scaling_factor=factor)
