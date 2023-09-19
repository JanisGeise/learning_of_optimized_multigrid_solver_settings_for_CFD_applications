"""
    get the avg. and max. Courant numbers & interface Courant numbers (if present) from the solver's log file and plot
    them
"""
import matplotlib.pyplot as plt

from glob import glob
from torch import tensor
from os.path import join
from os import path, makedirs


def get_cfl_number(load_path: str) -> dict:
    """
    gets the avg. and max. Courant numbers from the solver's log file

    :param load_path: path to the top-level directory of the simulation containing the log file from the flow solver
    :return: dict containing the mean & max. Courant numbers, and if present the mean and max. CFL from the interface
    """
    with open(glob(join(load_path, f"log.*Foam"))[0], "r") as f:
        logfile = f.readlines()

    """
    solver log file looks something like this:
    
        Courant Number mean: 0.00156147 max: 0.860588
        Interface Courant Number mean: 0 max: 0
        deltaT = 0.000117371
        Time = 0.000117370892
    """
    start_line = False
    data = {"cfl_mean": [], "cfl_max": [], "cfl_interface_mean": [], "cfl_interface_max": [], "dt": [], "t": []}
    for line in logfile:
        # omit the initial Courant number (prior starting the time loop)
        if line.startswith("Starting time loop"):
            start_line = True
        if line.startswith("Courant Number mean") and start_line:
            data["cfl_mean"].append(float(line.split(" ")[3]))
            data["cfl_max"].append(float(line.split(" ")[-1].strip("\n")))

        elif line.startswith("Interface Courant Number mean") and start_line:
            data["cfl_interface_mean"].append(float(line.split(" ")[4]))
            data["cfl_interface_max"].append(float(line.split(" ")[-1].strip("\n")))

        elif line.startswith("deltaT") and start_line:
            data["dt"].append(float(line.split(" ")[-1].strip("\n")))

        elif line.startswith("Time") and start_line:
            data["t"].append(float(line.split(" ")[-1].strip("\n")))

        else:
            continue

    # convert time to tensor, so it can be non-dimensionalized easier
    data["t"] = tensor(data["t"])
    data["dt"] = tensor(data["dt"]) if data["dt"] else None

    return data


def get_finish_time(load_path: str) -> float:
    """
    get the finish time of the simulation from the solver's log file

    :param load_path: path to the top-level directory of the simulation containing the log file from the flow solver
    :return: finish time
    """
    with open(glob(join(load_path, f"log.*Foam"))[0], "r") as f:
        logfile = f.readlines()

    return float([line.split(" ")[-1].strip("\n") for line in logfile if line.startswith("Time = ")][-1])


if __name__ == "__main__":
    # path to the top-level directory containing all simulations
    main_load_path = join(r"..", "run", "drl", "smoother", "results_weirOverflow")

    # the names of the directories of the simulations
    cases = ["DICGaussSeidel_local/run_1/", "DICGaussSeidel_local_2nd/run_1/", "policy_default_smoother_only/run_1/"]

    # name of the top-level directory where the plots should be saved
    save_path = join(main_load_path, "plots", "comparsion_default_vs_policy_default_only")

    # legend entries for the plot
    legend = ["$DICGaussSeidel$", "$DICGaussSeidel$ $(2^{nd})$", "$DICGaussSeidel$ $(policy)$"]

    # make directory for plots
    if not path.exists(join(save_path)):
        makedirs(join(save_path))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # factor for making the time dimensionless; here the period of the dominant vortex shedding frequency (T = 1 / f)
    # is used. For the surfaceMountedCube, the frequency of f = 0.15 Hz is taken from:
    # https://github.com/AndreWeiner/ofw2022_dmd_training/blob/main/dmd_flowtorch.ipynb
    # factor = 1 / 0.15

    # for mixerVesselAMI
    # factor = 1 / 1.6364

    # for weirOverflow
    factor = 1 / 0.4251

    # for cylinder2D (approx. of vortex shedding frequency @ Re = 1000)
    # factor = 1 / 20

    results = [get_cfl_number(join(main_load_path, c)) for c in cases]

    # plot time steps vs. Courant number, if dt was determined based on Courant number
    if results[0]["dt"] is not None:
        # get the finish time of the simulations for non-dimensionalizing the y-axis
        t_end = [get_finish_time(join(main_load_path, c)) for c in cases]

        fig, ax = plt.subplots(figsize=(6, 3))
        for i in range(len(results)):
            ax.plot(results[i]["t"] / factor, results[i]["dt"] / t_end[i], label=legend[i])
        ax.set_xlabel(r"$t \, / \, T$")
        ax.set_ylabel(r"$\Delta t \, / \, t_{end}$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()
        plt.legend(loc="upper right", framealpha=1.0, ncol=1)
        plt.savefig(join(save_path, f"dt_vs_t.png"), dpi=340)
        plt.show(block=False)
        plt.pause(2)
        plt.close("all")

    # plot courant numbers vs. t
    # check if we have a CFL number for the interface (if an interface exists)
    if results[0]["cfl_interface_mean"]:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), sharex="all", sharey="row")
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    for i in range(len(results)):
        if results[0]["cfl_interface_mean"]:
            ax[0][0].plot(results[i]["t"] / factor, results[i]["cfl_mean"])
            ax[1][0].plot(results[i]["t"] / factor, results[i]["cfl_max"])
        else:
            ax[0].plot(results[i]["t"] / factor, results[i]["cfl_mean"])
            ax[1].plot(results[i]["t"] / factor, results[i]["cfl_max"])

        if results[i]["cfl_interface_mean"]:
            ax[0][1].plot(results[i]["t"] / factor, results[i]["cfl_interface_mean"])
            ax[1][1].plot(results[i]["t"] / factor, results[i]["cfl_interface_max"])

    if results[0]["cfl_interface_mean"]:
        ax[1][0].set_xlabel(r"$t \, / \, T$")
        ax[1][1].set_xlabel(r"$t \, / \, T$")
        ax[0][0].set_ylabel(r"$\mu (Co)$")
        ax[1][0].set_ylabel("$max(Co)$")
        ax[0][0].set_title("$courant$ $number$")
        ax[0][1].set_title("$interface$ $courant$ $number$")
        loc = "lower center"
    else:
        ax[0].set_xlabel(r"$t \, / \, T$")
        ax[1].set_xlabel(r"$t \, / \, T$")
        ax[0].set_ylabel(r"$\mu (Co)$")
        ax[1].set_ylabel("$max(Co)$")
        loc = "upper center"

    fig.tight_layout()
    if results[0]["cfl_interface_mean"]:
        fig.subplots_adjust(bottom=0.14)
    else:
        fig.subplots_adjust(top=0.82)
    fig.legend(legend[:len(results)], loc=loc, framealpha=1.0, ncol=3)
    plt.savefig(join(save_path, f"cfl_vs_t.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
