"""
    plot the total execution times of the simulations and the execution times wrt time step for different settings in
    comparison to the results using e.g. the final (trained) policy of the DRL training. It is assumed that each case /
    setting is run multiple times in order to account for variations within the execution times which are not caused by
    the settings but e.g. by the scheduling of the HPC cluster
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs
from pandas import read_csv, DataFrame


def load_cpu_times(case_path: str) -> DataFrame:
    times = read_csv(case_path, sep="\t", comment="#", header=None, names=["t", "t_exec", "t_cpu"], usecols=[0, 1, 3])
    return times


if __name__ == "__main__":
    # main path to all the cases and save path
    load_path = join("..", "run", "drl", "interpolateCorrection")
    save_path = join(load_path, "plots")

    # names of top-level directory containing the simulations run with different settings
    cases = ["no", "yes", "random_policy", "trained_policy_1st_try"]

    # xticks for the plots
    xticks = ["$no$", "$yes$", "$random$ $policy$", "$trained$ $policy$"]

    # which case contains the default setting -> used for scaling the execution times
    default_idx = 1

    # scaling factor for num. time, here: approx. period length of vortex shedding frequency @ Re = 1000
    factor = 1 / 20

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    results = {"time_step": [], "mean_t_exec": [], "std_t_exec": [], "mean_t_per_dt": [], "std_t_per_dt": []}
    for c in cases:
        tmp = []

        # for each case glob all runs so we can compute the avg. runtimes etc.
        for case in glob(join(load_path, c, "*")):
            tmp.append(load_cpu_times(join(case, "postProcessing", "time", "0", "timeInfo.dat")))

        # sort the results from each case into a dict, assuming we ran multiple cases for each config.
        for key in tmp[0].keys():
            if key == "t":
                # all time steps per settings are the same, so just take the 1st one
                results["time_step"].append(tmp[0][key])
            elif key == "t_exec":
                results["mean_t_exec"].append(pt.mean(pt.cat([pt.from_numpy(i[key].values) for i in tmp])).item())
                results["std_t_exec"].append(pt.std(pt.cat([pt.from_numpy(i[key].values) for i in tmp])).item())

            elif key == "t_cpu":
                results["mean_t_per_dt"].append(pt.mean(pt.stack([pt.from_numpy(i[key].values) for i in tmp]), dim=0))
                results["std_t_per_dt"].append(pt.std(pt.stack([pt.from_numpy(i[key].values) for i in tmp]), dim=0))
            else:
                continue

    # plot the avg. execution times and the corresponding std. deviation
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(zip(results["mean_t_exec"], results["std_t_exec"])):
        # scale all execution times with the execution time of the default settings
        ax.errorbar(xticks[i], r[0] / results["mean_t_exec"][default_idx],
                    yerr=r[1] / results["std_t_exec"][default_idx], barsabove=True, fmt="o", capsize=5)
    ax.set_ylabel(r"$t^*_{exec}$", fontsize=13)
    # ax.tick_params(axis="x", labelsize=13)
    # ax.set_title(r"$'interpolateCorrection'$")
    ax.grid(visible=True, which="major", linestyle="-", alpha=0.45, color="black", axis="y")
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(visible=True, which="minor", linestyle="--", alpha=0.35, color="black", axis="y")
    fig.tight_layout()
    plt.savefig(join(save_path, "mean_execution_times.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")

    # plot the execution time wrt time step
    fig, ax = plt.subplots(nrows=2, figsize=(6, 6), sharex="col")
    for i, r in enumerate(zip(results["mean_t_per_dt"], results["std_t_per_dt"])):
        # scale all execution times with the execution time of the default settings
        ax[0].scatter(results["time_step"][i] / factor, r[0] / results["mean_t_per_dt"][default_idx], marker=".")
        ax[1].scatter(results["time_step"][i] / factor, r[1] / results["std_t_per_dt"][default_idx], marker=".")

    ax[0].set_ylabel(r"$\mu{(t^*_{exec})}$", fontsize=13)
    ax[1].set_ylabel(r"$\sigma{(t^*_{exec})}$", fontsize=13)
    ax[1].set_xlabel(r"$t \, / \, T$", fontsize=13)
    # ax[0].set_title(r"$'interpolateCorrection'$")
    fig.tight_layout()
    fig.legend(xticks, loc="upper center", framealpha=1.0, ncol=4)
    fig.subplots_adjust(top=0.94)
    plt.savefig(join(save_path, "execution_times_vs_dt.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
