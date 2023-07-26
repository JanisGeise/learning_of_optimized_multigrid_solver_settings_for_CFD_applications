"""
    post-process the grid convergence study of the 'weirOverflow' case, in particular:
        - plot the execution time of the simulation wrt amount of cells of the mesh
        - plot the probe locations as overlay on top of the flow field at different time steps
        - plot the flow field of different mesh refinement levels at specified time steps for qualitative comparison
        - plot the probes of the pressure written out during the simulation for different mesh refinement levels
        - plot the avg. pressure encountered at each probe location throughout the simulation
"""
import regex as re
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from typing import Union
from os import path, makedirs
from matplotlib.patches import Polygon
from flowtorch.data import FOAMDataloader


def load_probes(load_path: str, n_probes, filename: str = "p", skip_n_points: int = 0) -> pd.DataFrame:
    """
    load the data of the probes written out during the simulation

    :param load_path: path to the top-level directory of the simulation
    :param n_probes: amount of probes placed in the flow field
    :param filename: name of the field written out in the probes directory, e.g. 'p', 'p_rgh' or 'U'
    :param skip_n_points: offset, in case we don't want to read in the 1st N time steps of the values
    :return: dataframe containing the values for each probe
    """
    # skip header, header = n_probes + 2 lines containing probe no. and time header
    if filename.startswith("p"):
        names = ["time"] + [f"{filename}_probe_{i}" for i in range(n_probes)]
        p = pd.read_table(join(load_path, "postProcessing", "probes", "0", filename), sep=r"\s+",
                          skiprows=(n_probes+2)+skip_n_points, header=None, names=names)
    else:
        names = ["time"]
        for i in range(n_probes):
            names += [f"{k}_probe_{i}" for k in ["ux", "uy", "uz"]]

        p = pd.read_table(join(load_path, "postProcessing", "probes", "0", filename), sep=r"\s+",
                          skiprows=(n_probes+2)+skip_n_points, header=None, names=names)

        # replace all parenthesis, because (ux u_y uz) is separated since all columns are separated with white space
        # as well
        for k in names:
            if k.startswith("ux"):
                p[k] = p[k].str.replace("(", "", regex=True).astype(float)
            elif k.startswith("uz"):
                p[k] = p[k].str.replace(")", "", regex=True).astype(float)
            else:
                continue

    return p


def get_probe_locations(path_controlDict: str) -> list:
    """
    get the number of probes defined in the control dict of the base case, initialize a dummy policy with N_probes

    :param: path_controlDict: path to the probe dict
    :return: N_probes defined in the control dict of the base case
    """
    key = "probeLocations"
    with open(path_controlDict, "r") as f:
        lines = f.readlines()

    # get the start and end line containing all probe locations from the 'probes' function object in the 'controlDict'
    start = [(re.findall(key, l), idx) for idx, l in enumerate(lines) if re.findall(key, l)][0][1]
    end = [(re.findall(r"\);", l), idx) for idx, l in enumerate(lines) if re.findall(r"\);", l) and idx > start][0][1]

    # strip everything but the probe locations, start + 2 because "probeLocations" and "(\n" are still in the list;
    # then convert to tuple containing the (x, y, z)- coordinates of each probe
    coord = [eval(line.strip("\n").strip(" ").replace(" ", ",")) for line in lines[start+2:end-1]]

    return coord


def get_execution_time_from_log(load_path: str) -> float:
    """
    get the total execution time of the simulation from the solver's log file

    :param load_path: path to the top-level directory of the simulation containing the log file from the flow solver
    :return: execution time of the simulation in [s]
    """
    with open(glob(join(load_path, f"log.*Foam"))[0], "r") as f:
        logfile = f.readlines()

    # the final execution time is located somewhere at the end of the logfile, but the exact location depends on what
    # else is additionally written out, so just take the last exec time of final time step
    return [float(i.split(" ")[2]) for i in logfile if i.startswith("ExecutionTime = ")][-1]


def get_n_cells_from_log(load_path: str) -> int:
    """
    get the amount of cells for the simulation from the 'log.checkMesh' file (if present). If not present, the
    'log.snappyHexMesh' file will be used, otherwise (if only 'blockMesh' used), then the number of cells located in the
    'log.blockMesh' file will be taken

    :param load_path: path to the top-level directory of the simulation containing the log files
    :return: amount of cells of the mesh
    """
    # if log file from 'checkMesh' is available, then use 'checkMesh' log file
    if glob(join(load_path, f"log.checkMesh")):
        with open(glob(join(load_path, f"log.checkMesh"))[0], "r") as f:
            logfile = f.readlines()

        # number of cells are located under 'Mesh stats' at the beginning of the log file
        n_cells = [int(line.split(" ")[-1].strip("\n")) for line in logfile if line.startswith("    cells: ")][0]

    # in case there is no log file from 'checkMesh' available, then check is snappyHexMesh was used
    elif glob(join(load_path, f"log.snappyHexMesh")):
        with open(glob(join(load_path, f"log.snappyHexMesh"))[0], "r") as f:
            logfile = f.readlines()

        # number of cells are located at the end of the log file
        n_cells = [int(line.split(":")[2].split(" ")[0]) for line in logfile if line.startswith("Snapped mesh :")][0]

    else:
        # else use the log file from 'blockMesh'
        with open(glob(join(load_path, f"log.blockMesh"))[0], "r") as f:
            logfile = f.readlines()

        # number of cells are located under 'Mesh Information' at the end of the log file
        n_cells = [int(line.split(" ")[-1].strip("\n")) for line in logfile if line.startswith("  nCells: ")][0]

    return n_cells


def plot_execution_times_vs_n_cells(load_path: str, simulations: list, save_dir: str) -> None:
    """
    plot the execution time of the simulations wrt number of cells of the mesh

    :param load_path: path to the top-level directory containing e.g. the grid convergence study
    :param simulations: the names of the directories of the simulation with e.g. different grid refinement levels
    :param save_dir: name of the top-level directory where the plots should be saved
    :return: None
    """
    # make directory for plots
    if not path.exists(join(save_path, "plots")):
        makedirs(join(save_path, "plots"))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # get the amount of cells and corresponding execution time
    n_cells = [get_n_cells_from_log(join(load_path, s)) for s in simulations]
    t_exec = [get_execution_time_from_log(join(load_path, s)) for s in simulations]

    # plot the execution time in [min] wrt amount of cells
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(n_cells, [i / 60 for i in t_exec], marker="x", color="black")
    ax.set_ylabel(r"$t_{exec} \quad [min]$")
    ax.set_xlabel("$N_{cells}$")
    fig.tight_layout()
    plt.savefig(join(save_dir, "plots", f"t_exec_vs_N_cells.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_fields(load_path: str, save_dir: str, simulation: Union[list, str] = "weirOverflow_default_grid",
                times: list = ["25", "85"], param: str = "alpha.water", plot_probe_loc: bool = False,
                annotation: str = None) -> None:
    """

    :param load_path: path to the top-level directory containing e.g. the grid convergence study
    :param save_dir: name of the top-level directory where the plots should be saved
    :param simulation: the names of the directories / directory of the simulation(s)
    :param times: time steps for which the flow field should be plotted
    :param param: which field should be plotted, e.g. 'p_rgh', 'alpha_water', ... NOTE: only scalar fields possible
    :param plot_probe_loc: flag if the probe locations should be plotted as overlay on top of the flow field
    :param annotation: annotate the subplots, either 'time' for the time step or 'grid' for the grid refinement level
    :return:
    """
    # grid levels
    grid = ["$coarse$", "$default$", "$fine$"]

    # make directory for plots
    if not path.exists(join(save_path, "plots")):
        makedirs(join(save_path, "plots"))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    if plot_probe_loc:
        # load the probe locations if specified and convert to 2D-coord. since the 'weirOverflow' is a 2D-simulation
        probes = get_probe_locations(join(load_path, simulation, "system", "controlDict"))
        probes = [i[:2] for i in probes]

    # create loader and load the fields
    if type(simulation) == str:
        loader = FOAMDataloader(join(load_path, simulation))
        field = [loader.load_snapshot(param, i) for i in times]
    else:
        loader = [FOAMDataloader(join(load_path, c)) for c in cases]
        field = [l.load_snapshot(param, times[i]) for i, l in enumerate(loader)]

    # check if field is a scalar field
    assert len(field[0].size()) == 1, f"field {param} is not a scalar field. Only scalar fields, e.g. 'p_rgh' can be " \
                                      f"plotted"

    # plot the probe locations into the flow fields at the specified times
    fig, ax = plt.subplots(nrows=len(times), figsize=(6, len(times)+1), sharex="col", sharey="col")
    for i in range(len(times)):
        # plot the flow field
        if type(simulation) == str:
            ax[i].tricontourf(loader.vertices[:, 0], loader.vertices[:, 1], field[i], cmap="jet", levels=50,
                              extend="both")
        else:
            ax[i].tricontourf(loader[i].vertices[:, 0], loader[i].vertices[:, 1], field[i], cmap="jet", levels=50,
                              extend="both")

        # add a patch for the weir
        weir = Polygon([[0, 0], [0, 30], [15, 30], [30, 0]], facecolor="white", edgecolor="none")
        ax[i].add_patch(weir)

        # add annotations if specified
        if annotation == "time":
            ax[i].annotate(f"$t = {times[i]} s$", (75, 40), annotation_clip=False, bbox=dict(facecolor="white",
                                                                                             edgecolor="none"))
        elif annotation == "grid":
            ax[i].annotate(f"${grid[i]}$", (75, 35), annotation_clip=False, bbox=dict(facecolor="white",
                                                                                      edgecolor="none"))

        if plot_probe_loc:
            # plot probe locations if specified
            if i == 0:
                ax[i].scatter([i[0] for i in probes], [i[1] for i in probes], marker="s", color="red",
                              edgecolor="white", label="$probes$")
            else:
                ax[i].scatter([i[0] for i in probes], [i[1] for i in probes], marker="s", color="red",
                              edgecolor="white")
        ax[i].set_ylim(0, 50)
        ax[i].set_xlim(-18, 90)
        ax[i].set_ylabel("$y$")

    ax[-1].set_xlabel("$x$")
    fig.tight_layout()
    if plot_probe_loc:
        fig.legend(loc="lower right", framealpha=1.0, ncol=1)
        fig.subplots_adjust(bottom=0.20)
    if plot_probe_loc:
        plt.savefig(join(save_dir, "plots", f"probe_locations_{param}.png"), dpi=340)
    else:
        plt.savefig(join(save_dir, "plots", f"comparison_{param}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_avg_pressure(load_path: str, simulations: list, save_dir: str, n_probes: int = 10,
                      field_name: str = "p") -> None:
    """
    plot the avg. pressure encountered at the probe locations throughout the simulation

    :param load_path: path to the top-level directory containing e.g. the grid convergence study
    :param simulations: the names of the directories of the simulation with e.g. different grid refinement levels
    :param save_dir: name of the top-level directory where the plots should be saved
    :param n_probes: amount of probes placed in the flow field
    :param field_name: name of the probed field, e.g. 'p' or 'p_rgh'
    :return: None
    """
    # make directory for plots
    if not path.exists(join(save_path, "plots")):
        makedirs(join(save_path, "plots"))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load the pressure values of all probes written out during the simulation
    data_p = [load_probes(join(load_path, s), filename=field_name, n_probes=n_probes) for s in simulations]

    # get the amount of cells for each simulation
    n_cells = [get_n_cells_from_log(join(load_path, s)) for s in simulations]

    # plot the mean p wrt number of mesh cells
    fig, ax = plt.subplots(nrows=n_probes, figsize=(6, 7), sharex="col")
    for i in range(n_probes):
        # compute & plot mean p of all cases for current probe
        ax[i].plot(n_cells, [p[f"{field_name}_probe_{i}"].mean() for p in data_p], label=f"$probe$ ${i}$", marker="x",
                   color="black")
        ax[i].set_ylabel("$\\bar{p}$" + f"$_{i}$", labelpad=20, rotation="horizontal")
    fig.supxlabel("$N_{cells}$")
    fig.supylabel("$\\bar{p}_i$ $\\qquad [Pa]$")
    fig.tight_layout()
    fig.subplots_adjust(top=0.99)
    plt.savefig(join(save_dir, "plots", f"{field_name}_avg_vs_N_cells.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_probes(save_dir: str, data: list, n_probes: int = 10, title: str = "", param: str = "p",
                legend_list: list = None, share_y: bool = True) -> None:
    """
    plot the values of the probes wrt time

    :param save_dir: name of the top-level directory where the plots should be saved
    :param data: the probe data loaded using the 'load_probes' function of this script
    :param n_probes: number of probes placed in the flow field
    :param title: title of the plot (if wanted)
    :param param: parameter, either 'p', 'p_rgh'; or, if U was loaded: 'ux', 'uy', 'uz'
    :param legend_list: legend entries for the plot (if wanted)
    :param share_y: flag if all probes should have the same scaling for the y-axis
    :return: None
    """
    # make directory for plots
    if not path.exists(join(save_path, "plots")):
        makedirs(join(save_path, "plots"))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    if share_y:
        fig, ax = plt.subplots(nrows=n_probes, ncols=1, figsize=(8, 8), sharex="all", sharey="all")
    else:
        fig, ax = plt.subplots(nrows=n_probes, ncols=1, figsize=(8, 8), sharex="all")

    for j in range(len(data)):
        for i in range(n_probes):
            ax[i].plot(data[j]["time"], data[j][f"{param}_probe_{i}"])
            ax[i].set_ylabel(f"$probe$ ${i + 1}$", rotation="horizontal", labelpad=35)
    ax[-1].set_xlabel(r"$t \qquad [s]$")
    fig.suptitle(title)
    fig.tight_layout()
    if legend_list:
        fig.subplots_adjust(bottom=0.12)
        fig.legend(legend_list, loc="lower center", framealpha=1.0, ncol=3)
    plt.savefig(join(save_dir, "plots", f"probes_vs_time_{param}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # path to the top-level directory containing the grid convergence study
    main_load_path = join(r"..", "run", "grid_convergence_study_weirOverflow")

    # the names of the directories of the simulation with different grid refinement levels
    cases = ["weirOverflow_coarse_grid", "weirOverflow_default_grid", "weirOverflow_fine_grid"]

    # name of the top-level directory where the plots should be saved, the plots will be in a subdirectory named 'plots'
    save_path = main_load_path

    # plot the amount of cells and corresponding execution time
    plot_execution_times_vs_n_cells(main_load_path, cases, save_path)

    # plot the probe locations as overlay of the alpha.water field at different time steps
    plot_fields(main_load_path, save_path, times=["25", "85"], plot_probe_loc=True, annotation="time")

    # qualitative comparison of a specific field for different refinement levels at a specified time step
    plot_fields(main_load_path, save_path, cases, times=len(cases)*["85"], annotation="grid")

    # plot the probes for pressure wrt time
    plot_probes(save_path, [load_probes(join(main_load_path, c), n_probes=10) for c in cases], title=r"$p$ $vs.$ $t$",
                share_y=False, legend_list=["$coarse$", "$default$", "$fine$"])

    plot_probes(save_path, [load_probes(join(main_load_path, c), filename="p_rgh", n_probes=10) for c in cases],
                param="p_rgh", title=r"$p_{rgh}$ $vs.$ $t$", share_y=False,
                legend_list=["$coarse$", "$default$", "$fine$"])

    # plot the pressure at all probe locations avg. over the complete time span of the simulation
    plot_avg_pressure(main_load_path, cases, save_path)
    plot_avg_pressure(main_load_path, cases, save_path, field_name="p_rgh")
