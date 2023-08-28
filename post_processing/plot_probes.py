"""
    load and plot the probed fields of a number of simulations
"""
from os import path, makedirs
from typing import Union

import regex as re
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join


def get_number_of_probes(load_dir: str) -> int:
    """
    get the number of probes defined in the control dict of the base case, initialize a dummy policy with N_probes

    :param: load_dir: path to the case directory where the simulation was executed
    :return: amount of probes defined in the control dict of the simualtion
    """
    key = "probeLocations"

    with open(join(load_dir, "system", "controlDict"), "r") as f:
        lines = f.readlines()

    # get dict containing all probe locations
    start = [(re.findall(key, l), idx) for idx, l in enumerate(lines) if re.findall(key, l)][0][1]
    end = [(re.findall(r"\);", l), idx) for idx, l in enumerate(lines) if re.findall(r"\);", l) and idx > start][0][1]

    # remove everything but the probe locations, start + 2 because "probeLocations" and "(\n" are in list
    lines = [line for line in lines[start+2:end] if len(line.strip("\n").split(" ")) > 1]

    return len(lines)


def load_probes(load_dir: str, n_probes, filename: str = "p", skip_n_points: int = 0) -> pd.DataFrame:
    """
    load the data of the probes written out during the simulation

    :param load_dir: path to the top-level directory of the simulation
    :param n_probes: amount of probes placed in the flow field
    :param filename: name of the field written out in the probes directory, e.g. 'p', 'p_rgh' or 'U'
    :param skip_n_points: offset, in case we don't want to read in the 1st N time steps of the values
    :return: dataframe containing the values for each probe
    """
    # skip header, header = n_probes + 2 lines containing probe no. and time header
    if filename.startswith("p"):
        names = ["time"] + [f"{filename}_probe_{i}" for i in range(n_probes)]
        p = pd.read_table(join(load_dir, "postProcessing", "probes", "0", filename), sep=r"\s+",
                          skiprows=(n_probes+2)+skip_n_points, header=None, names=names)
    else:
        names = ["time"]
        for i in range(n_probes):
            names += [f"{k}_probe_{i}" for k in ["ux", "uy", "uz"]]

        p = pd.read_table(join(load_dir, "postProcessing", "probes", "0", filename), sep=r"\s+",
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


def plot_probes(save_dir: str, data: list, n_probes: int = 10, title: str = "", param: str = "p",
                legend_list: list = None, share_y: bool = True, xlabel: str = r"$t \qquad [s]$",
                scaling_factor: Union[int, float] = 1) -> None:
    """
    plot the values of the probes wrt time

    :param save_dir: name of the top-level directory where the plots should be saved
    :param data: the probe data loaded using the 'load_probes' function of this script
    :param n_probes: number of probes placed in the flow field
    :param title: title of the plot (if wanted)
    :param param: parameter, either 'p', 'p_rgh'; or, if U was loaded: 'ux', 'uy', 'uz'
    :param legend_list: legend entries for the plot (if wanted)
    :param share_y: flag if all probes should have the same scaling for the y-axis
    :param xlabel: label for the x-axis
    :param scaling_factor: factor for making the time dimensionless if wanted
    :return: None
    """
    # make directory for plots
    if not path.exists(join(save_dir)):
        makedirs(join(save_dir))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    if share_y:
        fig, ax = plt.subplots(nrows=n_probes, ncols=1, figsize=(8, 8), sharex="all", sharey="all")
    else:
        fig, ax = plt.subplots(nrows=n_probes, ncols=1, figsize=(8, 8), sharex="all")

    for j in range(len(data)):
        for i in range(n_probes):
            ax[i].plot(data[j]["time"] / scaling_factor, data[j][f"{param}_probe_{i}"])
            ax[i].set_ylabel(f"$probe$ ${i + 1}$", rotation="horizontal", labelpad=35)
    ax[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout()
    if legend_list:
        fig.subplots_adjust(bottom=0.12)
        fig.legend(legend_list, loc="lower center", framealpha=1.0, ncol=3)
    plt.savefig(join(save_dir, f"probes_vs_time_{param}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # path to the top-level directory containing the results of all simulations which should be compared
    load_path = join(r"..", "run", "grid_convergence_study_weirOverflow")

    # the names of the directories containing the results of each simulation
    cases = ["weirOverflow_coarse_grid", "weirOverflow_default_grid", "weirOverflow_fine_grid"]

    # legend entries for the plot
    legend = ["$coarse$", "$default$", "$fine$"]

    # names of the fields defined in the control dict for which the probes should be plotted, e.g. 'U', 'p' or 'p_rgh'
    filenames = ["p_rgh", "U"]

    # name of the top-level directory where the plots should be saved
    save_path = join(load_path, "plots")

    # factor for making the time dimensionless; here the period of the dominant frequency in the flow field is used
    # for weirOverflow case
    factor = 1 / 0.4251

    # for the surfaceMountedCube case
    # factor = 1 / 0.15

    # for the mixerVesselAMI case
    # factor = 1 / 1.6364

    # plot the probes for all specified parameters wrt time
    for p in filenames:
        data = [load_probes(join(load_path, c), n_probes=get_number_of_probes(join(load_path, c)), filename=p)
                for c in cases]
        if p == "U":
            for i in ["ux", "uy", "uz"]:
                plot_probes(save_path, data, param=i, legend_list=legend, scaling_factor=factor, xlabel=r"$t \, / \, T$")
        else:
            plot_probes(save_path, data, param=p, legend_list=legend, scaling_factor=factor, xlabel=r"$t \, / \, T$")
