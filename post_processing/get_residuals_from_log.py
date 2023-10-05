"""
    load and filter the solver's log file wrt the residuals of GAMG when solving the pressure equation. Compute the
    statistical properties of the residuals such as convergence rate wrt the time steps of the simulation and plot them.

    Further some helper functions for getting the number of cells and the execution time from the log files.
"""
import torch as pt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from typing import Union
from os import path, makedirs


def get_execution_time_from_log(load_dir: str) -> float:
    """
    get the total execution time of the simulation from the solver's log file

    :param load_dir: path to the top-level directory of the simulation containing the log file from the flow solver
    :return: execution time of the simulation in [s]
    """
    with open(glob(join(load_dir, f"log.*Foam"))[0], "r") as f:
        logfile = f.readlines()

    # the final execution time is located somewhere at the end of the logfile, but the exact location depends on what
    # else is additionally written out, so just take the last exec time of final time step
    return [float(i.split(" ")[2]) for i in logfile if i.startswith("ExecutionTime = ")][-1]


def get_n_cells_from_log(load_dir: str) -> int:
    """
    get the amount of cells for the simulation from the 'log.checkMesh' file (if present). If not present, the
    'log.snappyHexMesh' file will be used, otherwise (if only 'blockMesh' used), then the number of cells located in the
    'log.blockMesh' file will be taken

    :param load_dir: path to the top-level directory of the simulation containing the log files
    :return: amount of cells of the mesh
    """
    # if log file from 'checkMesh' is available, then use 'checkMesh' log file
    if glob(join(load_dir, f"log.checkMesh")):
        with open(glob(join(load_dir, f"log.checkMesh"))[0], "r") as f:
            logfile = f.readlines()

        # number of cells are located under 'Mesh stats' at the beginning of the log file
        n_cells = [int(line.split(" ")[-1].strip("\n")) for line in logfile if line.startswith("    cells: ")][0]

    # in case there is no log file from 'checkMesh' available, then check is snappyHexMesh was used
    elif glob(join(load_dir, f"log.snappyHexMesh")):
        with open(glob(join(load_dir, f"log.snappyHexMesh"))[0], "r") as f:
            logfile = f.readlines()

        # number of cells are located at the end of the log file
        n_cells = [int(line.split(":")[2].split(" ")[0]) for line in logfile if line.startswith("Snapped mesh :")][0]

    else:
        # else use the log file from 'blockMesh'
        with open(glob(join(load_dir, f"log.blockMesh"))[0], "r") as f:
            logfile = f.readlines()

        # number of cells are located under 'Mesh Information' at the end of the log file
        n_cells = [int(line.split(" ")[-1].strip("\n")) for line in logfile if line.startswith("  nCells: ")][0]

    return n_cells


def get_GAMG_residuals(case_path: str) -> dict:
    """
    filter the solver log file for the residuals of the pressure equation solved with GAMG. Then the statistical
    properties of the residuals such as convergence rate are computed wrt the time steps.

    :param case_path: path to the case directory where the log file is located
    :return: the statistical properties of the residuals wrt time steps
    """
    # log for solver is always 'log.*Foam', but the name may differ from the actual solver used, e.g. PIMPLE is used in
    # 'overInterPhaseChangeDyMFoam'
    with open(glob(join(case_path, f"log.*Foam"))[0], "r") as f:
        logfile = f.readlines()

    # check if simulation converged -> in that case just return empty list because we can't really use the results
    if not logfile[-1].startswith("Finalising parallel run"):
        print("Simulation diverged, no data available!")
        return {}
    else:
        residual_data = {"initial_residual": [], "final_residual": [], "n_gamg_iter": [], "n_solver_iter": [],
                         "time": [], "exec_time": []}
        init_residual, final_residual, n_gamg_iter, next_time_step = [], [], [], False

        for i, line in enumerate(logfile):
            # new num. time step starts
            if line.startswith("Time = "):
                # go through all the lines of current time step, starting with the next line
                residual_data["time"].append(float(line.strip("\n").split(" ")[-1]))
                l = i + 1
                while not next_time_step:
                    # ignore all quantities derived from p, e.g. 'p_corr' or 'pFinal'
                    if logfile[l].startswith("GAMG:  Solving for p_rgh") or logfile[l].startswith("GAMG:  Solving for p,"):
                        tmp = logfile[l].split(",")
                        init_residual.append(float(tmp[1].split(" ")[-1]))
                        final_residual.append(float(tmp[2].split(" ")[-1]))
                        n_gamg_iter.append(float(tmp[-1].split(" ")[-1]))

                    # end of num. time step
                    elif logfile[l].startswith("ExecutionTime = "):
                        # required solver iterations are printed 1 line above the execution time, we can't convert them
                        # to tensors because each list container different amount of data
                        residual_data["n_solver_iter"].append(int(logfile[l-1].split(" ")[-2]))
                        residual_data["initial_residual"].append(init_residual)
                        residual_data["final_residual"].append(final_residual)
                        residual_data["n_gamg_iter"].append(n_gamg_iter)

                        # get execution time for each time step
                        if not residual_data["exec_time"]:
                            residual_data["exec_time"].append(float(logfile[l].split(" ")[2]))
                        else:
                            # if this is not the 1st time step, we need to subtract the exec time until the previous
                            # time step -> take the cumulative exec time until the last time step and subtract that
                            # from the current exec time
                            dt = round(float(logfile[l].split(" ")[2]) - sum(residual_data["exec_time"]), 2)
                            residual_data["exec_time"].append(dt)

                        # clear the tmp lists
                        init_residual, final_residual, n_gamg_iter = [], [], []
                        break
                    l += 1

        # compute statistical properties of the initial residuals and N_GAMG_iter
        residual_data["sum_gamg_iter"] = pt.tensor([sum(i) for i in residual_data["n_gamg_iter"]])
        residual_data["max_gamg_iter"] = pt.tensor([max(i) for i in residual_data["n_gamg_iter"]])
        residual_data["max_init_residual"] = pt.tensor([max(i) for i in residual_data["initial_residual"]])
        residual_data["init_residual"] = pt.tensor([i[0] for i in residual_data["initial_residual"]])

        # compute the max. & median convergence rate -> median, because at beginning high, but at the end = 0
        convergence_rate = [[abs(t[i+1] - t[i]) for i in range(len(t)-1)] for t in residual_data["initial_residual"]]
        residual_data["max_convergence_rate"] = pt.tensor([max(i) for i in convergence_rate])
        residual_data["min_convergence_rate"] = pt.tensor([min(i) for i in convergence_rate])
        residual_data["avg_convergence_rate"] = pt.tensor([pt.mean(pt.tensor(i)).item() for i in convergence_rate])
        residual_data["median_convergence_rate"] = pt.tensor([pt.median(pt.tensor(i)).item() for i in convergence_rate])

        # convert the N PIMPLE iter to tensor
        residual_data["n_solver_iter"] = pt.tensor(residual_data["n_solver_iter"])

        return residual_data


def map_keys_to_labels(key: str) -> str:
    """
    map the key of the dict to a label, which will be used as subtitle for plotting

    :param key: the key to map
    :return: the label associated with the key
    """

    if key == "n_solver_iter":
        label = r"$N_{iter, \, solver} \, / \, \Delta t$"
    elif key == "exec_time":
        label = "$t_{exec}$"
    elif key == "sum_gamg_iter":
        label = r"$\sum{N_{GAMG}} \, / \, \Delta t$"
    elif key == "max_gamg_iter":
        label = r"$N_{GAMG, \, max} \, / \, \Delta t$"
    elif key == "max_init_residual":
        label = "$max(\\boldsymbol{R}_0)$"
    elif key == "init_residual":
        label = "$\\boldsymbol{R}_0$"
    elif key == "min_convergence_rate":
        label = "$|\Delta \\boldsymbol{R}_{min}|$"
    elif key == "median_convergence_rate":
        label = "$|\Delta \\boldsymbol{R}_{median}|$"
    elif key == "avg_convergence_rate":
        label = "$|\Delta \\boldsymbol{R}_{avg}|$"
    elif key == "max_convergence_rate":
        label = "$|\Delta \\boldsymbol{R}_{max}|$"
    elif key == "time":
        label = "$t_{num}$"
    elif key == "ratio_gamg_iter":
        label = r"$\frac{\left(\sum{N_{GAMG}}\right) - N_{GAMG,\, max}}{\left(\sum{N_{GAMG}}\right)+N_{GAMG, \, max}}$"
    elif key == "ratio_solver_iter":
        label = r"$\frac{N_{PIMPLE}}{N_{PIMPLE, max}}$"
    elif key == "init_residual_scaled":
        label = "$sigmoid(\\boldsymbol{R}_0)^*$"
    elif key == "median_convergence_rate_scaled":
        label = "$sigmoid(|\Delta \\boldsymbol{R}_{median}|)^*$"
    elif key == "min_convergence_rate_scaled":
        label = "$sigmoid(|\Delta \\boldsymbol{R}_{min}|)^*$"
    elif key == "max_convergence_rate_scaled":
        label = "$sigmoid(|\Delta \\boldsymbol{R}_{max}|)^*$"
    else:
        label = "$parameter$"
    return label


def plot_correlations(data: pd.DataFrame, save_dir: str, save_name: str = "correlations") -> None:
    """
    plot the linear correlations of the statistical properties of the residuals as heatmap

    :param save_dir: path where the plot should be saved to
    :param data: the correlations
    :param save_name: name of the plot
    :return: None
    """
    # convert the key names of the data frame to labels for the heatmap
    labels = [map_keys_to_labels(label) for label in data.keys()]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(data, annot=True, cmap="Greens", square=True, cbar=True, linewidths=0.30, vmin=-1, vmax=1,
                linecolor="white", ax=ax, cbar_kws={"shrink": 0.75}, fmt=".2f", xticklabels=labels, yticklabels=labels)
    fig.tight_layout()
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_residuals(x: list, y: list, save_dir: str, save_name: str = "", ylabel: str = "", xlabel: str = r"$t \, / \, T$",
                   title: str = "", scaling_factor: Union[int, float] = 1, scale_dt: bool = False, log_path: str = None,
                   legend_entries: list = None, log_y: bool = False) -> None:
    """
    plot the results of the filtered solver log-file wrt the statistical properties of the GAMG residuals

    :param x: x-values for each case
    :param y: corresponding y-values
    :param save_dir: directory to which the plot should be saved to
    :param save_name: save name of the plot
    :param ylabel: label for the y-axis
    :param xlabel: label for the x-axis
    :param title: title of the plot if wanted
    :param scaling_factor: factor for making the time dimensionless if wanted
    :param scale_dt: flag if execution time per time step should be scaled wrt total execution time of the simulation
    :param log_path: path to the solver's log file in case 'scale_dt = True'
    :param legend_entries: list containing the legend entries
    :param log_y: flag if the y-axis should be logarithmic
    :return: None
    """

    # scale the num. time with period length of dominant frequency in the flow field
    x = [[i / scaling_factor for i in case] for case in x]

    # scale the execution time per time step with the total execution time if required
    if scale_dt:
        t_tot = get_execution_time_from_log(log_path)
        y = [[i / t_tot for i in case] for case in y]

    fig, ax = plt.subplots(figsize=(6, 3))
    for i in range(len(x)):
        if not legend_entries:
            ax.scatter(x[i], y[i], marker=".")
        else:
            ax.scatter(x[i], y[i], marker=".", label=legend_entries[i])

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if log_y:
        ax.set_yscale("log")

    ax.set_title(title)
    fig.tight_layout()

    if legend_entries:
        plt.legend(loc="best", framealpha=1.0, ncol=3)

    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # path to the simulation results and save path for plots
    load_path = join("..", "run", "parameter_study", "influence_grid", "mixerVesselAMI_fineGrid")
    save_path = join("..", "run", "parameter_study", "influence_grid", "plots", "mixerVesselAMI")

    # append to save name in case multiple cases are plotted we can save all plots in the same directory for easier
    # comparison without accidentally overwriting the plots
    append_to_save_name = r"fineGrid"

    # factor for making the time dimensionless; here the period of the dominant frequency in the flow field is used
    # for weirOverflow case
    # factor = 1 / 0.4251

    # for the surfaceMountedCube case
    # factor = 1 / 0.15

    # for the mixerVesselAMI case
    factor = 1 / 1.6364

    try:
        # check if log file was already processed
        residuals = pt.load(join(load_path, "log_data_filtered.pt"))
    except FileNotFoundError:
        # else filter log file wrt GAMG
        residuals = get_GAMG_residuals(load_path)

        # save the data from the log file in case we want to do sth. with them later
        pt.save(residuals, join(load_path, "log_data_filtered.pt"))

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})

    # plot correlations
    plot_correlations(pd.DataFrame.from_dict(residuals).corr(), save_path, f"correlations_{append_to_save_name}")

    # plot execution time per time step
    plot_residuals([residuals["time"]], [residuals["exec_time"]], save_path, ylabel=r"$t_{exec} \, / \, t_{total}$",
                   save_name=f"t_exec_vs_dt_{append_to_save_name}", scale_dt=True, log_path=load_path,
                   scaling_factor=factor, log_y=True)

    # plot N solver iterations vs. time step
    plot_residuals([residuals["time"]], [residuals["n_solver_iter"]], save_path, scaling_factor=factor,
                   ylabel=r"$N_{iter, \, solver}$", save_name=f"pimple_iter_vs_dt_{append_to_save_name}")
    
    # plot sum N GAMG iterations vs. time step
    plot_residuals([residuals["time"]], [residuals["sum_gamg_iter"]], save_path, scaling_factor=factor, log_y=True,
                   ylabel=r"$\sum{N_{GAMG}} \, / \, \Delta t$", save_name=f"sum_gamg_iter_vs_dt_{append_to_save_name}")

    # plot max. N GAMG iterations vs. time step
    plot_residuals([residuals["time"]], [residuals["max_gamg_iter"]], save_path, scaling_factor=factor,
                   ylabel=r"$N_{GAMG, \, max}$", save_name=f"max_gamg_iter_vs_dt_{append_to_save_name}")

    # plot the max. initial residual vs. dt
    plot_residuals([residuals["time"]], [residuals["max_init_residual"]], save_path, log_y=True, scaling_factor=factor,
                   ylabel="$max(\\boldsymbol{R}_0)$", save_name=f"max_init_residual_vs_dt_{append_to_save_name}")

    # plot max / median / min convergence rate vs. dt
    plot_residuals(3*[residuals["time"]], [residuals["max_convergence_rate"], residuals["median_convergence_rate"],
                                           residuals["min_convergence_rate"]],
                   save_path, f"convergence_rate_vs_dt_{append_to_save_name}", ylabel="$|\Delta \\boldsymbol{R}|$",
                   scaling_factor=factor, legend_entries=["$max.$", "$median$", "$min.$"], log_y=True)
