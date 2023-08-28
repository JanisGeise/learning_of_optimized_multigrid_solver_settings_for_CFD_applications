"""
    plot the required execution time of the simulations vs. the setup, e.g. number of subdomains, solver settings, ...
"""
import torch as pt
import matplotlib.pyplot as plt

from typing import List
from os.path import join
from os import path, makedirs

from post_processing.get_residuals_from_log import get_execution_time_from_log, get_n_cells_from_log


def get_number_of_subdomains(load_dir: str) -> int:
    """
    get the number of subdomains for a simulation from the 'decomposeParDict'

    :param load_dir: path to the directory where the results of the simulation is located
    :return: the number of subdomains
    """
    with open(join(load_dir, "system", "decomposeParDict"), "r") as f:
        lines = f.readlines()

    n = [int(i.split(" ")[-1].strip(";\n")) for i in lines if i.startswith("numberOfSubdomains")][0]
    return n


def load_total_exec_times(load_dir: str, simulations: List[list], default: list = None) -> dict:
    """
    get the execution time, number of subdomains and number of cells (of the grid) for the specified simulations

    :param load_dir: path to the main directory containing all simulations which are supposed to be loaded
    :param simulations: list of list with all simulations, each list will later be one line in the plot
    :param default: the indices of the default setting, these cases will be used as reference for scaling the exec. time
    :return: dict containing the scaled exec. times, n_subdomains and n_cells for each simulation
    """
    subdomains, t_exec, n_cells = [], [], []
    for idx, simulation in enumerate(simulations):
        tmp_domains, tmp_cells, tmp_t = [], [], []
        for s in simulation:
            if s.startswith("surfaceMountedCube"):
                tmp_t.append(get_execution_time_from_log(join(load_dir, s, "fullCase")))
                tmp_domains.append(get_number_of_subdomains(join(load_dir, s, "fullCase")))
                tmp_cells.append(get_n_cells_from_log(join(load_dir, s, "fullCase")))
            else:
                tmp_t.append(get_execution_time_from_log(join(load_dir, s)))
                tmp_domains.append(get_number_of_subdomains(join(load_dir, s)))
                tmp_cells.append(get_n_cells_from_log(join(load_dir, s)))

        # scale the execution times wrt default setup if a default case is given
        if default:
            if type(default) == list:
                # scale wrt the cases
                t_exec.append(pt.tensor(tmp_t) / tmp_t[default[idx]])
            else:
                # scale all simulations with the same t_exec (comparison different decomp. methods)
                t_exec.append(pt.tensor(tmp_t) / default)
        else:
            t_exec.append(tmp_t)
        subdomains.append(tmp_domains)
        n_cells.append(tmp_cells)

    return {"t_exec": t_exec, "n_domains": subdomains, "n_cells": n_cells}


def plot_exec_times_vs_setting(data: dict, save_dir: str = "", save_name: str = "comparison_t_exec",
                               legend_list: list = None, y_label: str = r"$t^*_{exec}$",
                               x_label: str = "$N_{domains}$") -> None:
    """
    plot the (scaled) execution times wrt a specified target parameter, such as solver settings for each simulation

    :param data: the dict containing the loaded execution times and corresponding x-values
    :param save_dir: directory to which the plot should be saved to
    :param save_name: name of the plot
    :param legend_list: legend entries for the plot if wanted
    :param y_label: label for the y-axis
    :param x_label: label for the x-axis
    :return: None
    """
    fig, ax = plt.subplots(figsize=(6, 3))

    # plot horizontal line as reference if scaled
    # ax.hlines(1, 20, 80, color="black", ls="-.", alpha=0.5)
    # ax.set_xlim(20, 80)

    for i in range(len(data["t_exec"])):
        if legend:
            ax.plot(data[key][i], data["t_exec"][i], label=legend_list[i], marker="x")
        else:
            ax.plot(data[key][i], data["t_exec"][i], marker="x")

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    fig.tight_layout()

    if legend_list:
        plt.legend(loc="upper right", framealpha=1.0, ncol=1)

    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # path to the simulation results and save path for plots
    load_path = join(r"..", "run", "parameter_study", "influence_n_subdomains")
    save_path = join(load_path, "plots", "surfaceMountedCube")

    # the names of the directories of the simulations, each list will be plotted later as one line
    cases = [
                ["surfaceMountedCube_20subdomains_simple", "surfaceMountedCube_36subdomains_simple",
                 "surfaceMountedCube_40subdomains_simple", "surfaceMountedCube_60subdomains_simple",
                 "surfaceMountedCube_80subdomains_simple"],
                ["surfaceMountedCube_20subdomains_hierarchical", "surfaceMountedCube_36subdomains_hierarchical",
                 "surfaceMountedCube_40subdomains_hierarchical", "surfaceMountedCube_60subdomains_hierarchical",
                 "surfaceMountedCube_80subdomains_hierarchical"],
                ["surfaceMountedCube_20subdomains_scotch", "surfaceMountedCube_36subdomains_scotch",
                 "surfaceMountedCube_40subdomains_scotch", "surfaceMountedCube_60subdomains_scotch",
                 "surfaceMountedCube_80subdomains_scotch"],
            ]

    # default setting for each simulation, used for scaling all execution times in order to compare the cases relative
    # to each other, if 'None', then the execution times will be plotted in [s], not scaled
    # default_idx = [2, 2, 2]

    # in case all the simulations should be scaled with one global execution times (not wrt cases)
    default_idx = 136737

    # in case we want to scale the execution times, we need to provide a default value for each parameter setting (list)
    if default_idx is not None and type(default_idx) == list:
        assert len(default_idx) == len(cases), "The index of the default setting need to be specified for each list" \
                                               " within the 'cases' list!"

    # save name of the plot
    name = "t_exec_vs_decomposition_method"

    # legend entries for the plot
    legend = ["$simple$", "$hierarchical$", "$scotch$"]

    # xlabel for the plot (will be ignored if N_cells or N_domains should be plotted)
    xlabel = "$nCellsInCoarsestLevel$"

    # which parameter should be on the x-axis, if key != 'n_cells' or 'n_domains' then the x_label_list will be used
    key = "n_domains"

    # in case we don't have numeric values such as n_subdomains or n_cells for the x-axis use the x_label_list as
    # x-tick-values, otherwise this parameter will be ignored
    # x_label_list = ["$GaussSeidel$", "$DICGaussSeidel$", "$FDIC$"]
    x_label_list = [10, 100, 1000]
    # x_label_list = ["$no$", "$yes$"]

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load the data for all cases
    times = load_total_exec_times(load_path, cases, default_idx)

    # add the x_label as key in case we don't want to use numeric values for the x-axis
    if key != "n_cells" and key != "n_domains":
        times[key] = [x_label_list] * len(cases)
    elif key == "n_cells":
        xlabel = "$N_{cells}$"
    elif key == "n_domains":
        xlabel = "$N_{domains}$"

    # plot the results
    plot_exec_times_vs_setting(times, save_path, save_name=name, legend_list=legend, x_label=xlabel)
