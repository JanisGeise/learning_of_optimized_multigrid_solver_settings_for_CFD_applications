"""
    plots the amount of time steps of the complete simulation in case dt is based on the Courant number (dt != const.)
"""
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs

from post_processing.plot_execution_times_vs_setup import get_number_of_subdomains
from post_processing.plot_grid_convergence_study_weirOverflow import get_n_cells_from_log


def get_number_of_time_steps(load_path: str) -> int:
    """
    get the total amount of required time steps of the simulation from the solver's log file

    :param load_path: path to the top-level directory of the simulation containing the log file from the flow solver
    :return: amount of time steps
    """
    with open(glob(join(load_path, f"log.*Foam"))[0], "r") as f:
        logfile = f.readlines()

    # get the amount of time steps from the solver log file
    counter = 0
    for i in range(len(logfile)):
        if logfile[i].startswith("Time = "):
            counter += 1
        else:
            continue
    return counter


if __name__ == "__main__":
    # path to the top-level directory containing all simulations
    main_load_path = join(r"..", "run", "parameter_study", "influence_solver_settings", "smoother")

    # the names of the directories of the simulations
    cases = ["mixerVesselAMI_FDIC", "mixerVesselAMI_DICGaussSeidel", "mixerVesselAMI_GaussSeidel"]

    # index of the default settings
    default_idx = 1

    # name of the top-level directory where the plots should be saved
    save_path = join(main_load_path, "plots", "mixerVesselAMI", "plots_latex")

    # make directory for plots
    if not path.exists(join(save_path)):
        makedirs(join(save_path))

    # load the amount of time steps for each case
    n_dt = [get_number_of_time_steps(join(main_load_path, c)) for c in cases]
    default_n_dt = n_dt[default_idx]

    # labels & ticks for the x-axis (here N_cells for x-axis)
    # xticks = [get_n_cells_from_log(join(main_load_path, c)) for c in cases]
    # xlabel = r"$N_{cells}$"

    # alternative x-ticks
    # xticks = [get_number_of_subdomains(join(main_load_path, c)) for c in cases]
    # xlabel = r"$N_{domains}$"
    xticks = ["$FDIC$", "$DICGaussSeidel$", "$GaussSeidel$"]
    xlabel = r""

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # plot results
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(xticks, [n / default_n_dt for n in n_dt], color="black", marker="x")
    # ax.hlines(1, min(xticks), max(xticks), color="black", ls="-.", alpha=0.5)
    ax.set_ylabel(r"$N_{\Delta t} \, / \, N_{\Delta t, default}$")
    ax.set_xlabel(xlabel)
    # ax.set_xlim(min(xticks), max(xticks))
    fig.tight_layout()
    plt.savefig(join(save_path, f"n_dt.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
