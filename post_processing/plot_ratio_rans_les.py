"""
    plot the ratio between RANS and LES for the 'surfaceMountedCube' case wrt time
"""
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from os import path, makedirs


if __name__ == "__main__":
    # path to the top-level directory containing all simulations
    load_path = join(r"..", "run", "parameter_study", "influence_n_subdomains")

    # the names of the directories of the simulations
    cases = ["surfaceMountedCube_20subdomains_hierarchical", "surfaceMountedCube_40subdomains_hierarchical",
             "surfaceMountedCube_80subdomains_hierarchical"]

    # name of the top-level directory where the plots should be saved
    save_path = join(load_path, "plots", "surfaceMountedCube", "hierarchical")

    # legend entries for the plot
    legend = ["$20$ $subdomains$, $1$ $node$", "$40$ $subdomains$, $2$ $nodes$", "$80$ $subdomains$, $4$ $nodes$"]

    # factor for making the time dimensionless; here the period of the dominant vortex shedding frequency (T = 1 / f)
    # is used. The dominant frequency of f = 0.15 Hz is taken from:
    # https://github.com/AndreWeiner/ofw2022_dmd_training/blob/main/dmd_flowtorch.ipynb
    factor = 1 / 0.15

    # make directory for plots
    if not path.exists(join(save_path)):
        makedirs(join(save_path))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load file containing the data about the DES field
    data = []
    for c in cases:
        data.append(pd.read_table(join(load_path, c, "fullCase", "postProcessing", "DESField", "0",
                                       "DESModelRegions.dat"), sep=r"\t", skiprows=2, header=None,
                                  names=["time", "LES", "RAS"], engine="python"))

    # plot the results
    fig, ax = plt.subplots(figsize=(6, 3))
    for c in range(len(cases)):
        ax.plot(data[c]["time"] / factor, data[c]["RAS"] / data[c]["LES"], label=legend[c])
    ax.set_ylabel(r"$RANS \,/\, LES$")
    ax.set_xlabel(r"$t \, / \, T$", fontsize=12)
    fig.tight_layout()
    plt.legend(loc="lower right", framealpha=1.0, ncol=1)
    plt.savefig(join(save_path, f"ratio_RANS_LES_vs_time.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
