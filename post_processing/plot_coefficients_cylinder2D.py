"""
    plot the drag coefficient of the cylinder with respect to the time step
"""
from os.path import join
from pandas import read_csv
from os import path, makedirs
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # path to the simulation results and save path for plots
    load_path = join("..", "run", "parameter_study", "influence_solver_settings", "interpolateCorrection")
    save_path = join(load_path, "plots", "cylinder2D", "plots_latex")

    # the names of the directories of the simulations
    cases = ["cylinder2D_no"]

    # legend entries for the plot
    legend = ["$no$", "$yes$"]

    # factor for making the time dimensionless; here the period of the dominant frequency in the flow field is used
    factor = 1 / 20

    data = [read_csv(join(load_path, c, "postProcessing", "forces", "0", "coefficient.dat"), skiprows=13, header=0,
                     sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"]) for c in cases]

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})

    # plot cd with respect to time
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(data[0]["t"] / factor, data[0]["cd"])
    ax.set_ylabel(r"$c_d$")
    ax.set_xlabel(r"$t \, / \, T$", fontsize=12)
    ax.vlines(5.2, 1.0, 4.8, ls="-.", lw=2, color="red")
    ax.vlines(3.6, 1.0, 4.8, ls="-.", lw=2, color="red")
    fig.tight_layout()
    # plt.legend(loc="lower right", framealpha=1.0, ncol=1)
    plt.savefig(join(save_path, f"cd_vs_t.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
