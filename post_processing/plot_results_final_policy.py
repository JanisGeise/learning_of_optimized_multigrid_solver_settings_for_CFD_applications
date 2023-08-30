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
    load_path = join("..", "run", "drl", "smoother", "results_weirOverflow")
    save_path = join(load_path, "plots")

    # names of top-level directory containing the simulations run with different settings
    cases = ["FDIC_local", "DIC_local", "DICGaussSeidel_local", "symGaussSeidel_local", "nonBlockingGaussSeidel_local",
             "GaussSeidel_local", "random_policy_local", "trained_policy_local"]

    # xticks for the plots
    xticks = ["$FDIC$", "$DIC$", "$DIC$\n$GaussSeidel$", "$sym$\n$GaussSeidel$", "$nonBlocking$\n$GaussSeidel$",
              "$GaussSeidel$", "$random$\n$policy$", "$trained$\n$policy$"]

    # which case contains the default setting -> used for scaling the execution times
    default_idx = 2

    # flag if the avg. execution time and corresponding std. deviation should be scaled wrt default setting
    scale = True

    # scaling factor for num. time, here: approx. period length of vortex shedding frequency @ Re = 1000
    # factor = 1 / 20

    # factor for weirOverflow case
    factor = 1 / 0.4251

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
                # in some cases, the amount of dt might differ, although the same setup, seed etc. was used
                # in that case we take the first N dt which are available for all cases because the dt plot is a
                # qualitative comparison anyway
                min_dt = min([i[key].size for i in tmp])

                results["mean_t_per_dt"].append(pt.mean(pt.stack([pt.from_numpy(i[key].values)[:min_dt] for i in tmp]),
                                                        dim=0))
                results["std_t_per_dt"].append(pt.std(pt.stack([pt.from_numpy(i[key].values)[:min_dt] for i in tmp]),
                                                      dim=0))
            else:
                continue

    # plot the avg. execution times and the corresponding std. deviation
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(zip(results["mean_t_exec"], results["std_t_exec"])):
        # scale all execution times with the execution time of the default settings
        if scale:
            ax.errorbar(xticks[i], r[0] / results["mean_t_exec"][default_idx],
                        yerr=r[1] / results["std_t_exec"][default_idx], barsabove=True, fmt="o", capsize=5)
            ax.set_ylabel(r"$t^*_{exec}$", fontsize=13)

        # no scaling
        else:
            ax.errorbar(xticks[i], r[0], yerr=r[1], barsabove=True, fmt="o", capsize=5)
            ax.set_ylabel(r"$t_{exec}$   $[s]$", fontsize=13)
    ax.grid(visible=True, which="major", linestyle="-", alpha=0.45, color="black", axis="y")
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(visible=True, which="minor", linestyle="--", alpha=0.35, color="black", axis="y")
    fig.tight_layout()
    if scale:
        plt.savefig(join(save_path, "mean_execution_times.png"), dpi=340)
    else:
        plt.savefig(join(save_path, "mean_execution_times_abs.png"), dpi=340)

    plt.show(block=False)
    plt.pause(2)
    plt.close("all")

    # make sure all cases have the same amount of time steps as the default case, if not then take the 1st N time steps
    # which are available for all cases (difference for 'weirOverflow' is ~10 dt and therefore not visible anyway)
    min_n_dt = min([len(i) for i in results["time_step"]])

    # plot the execution time wrt time step
    fig, ax = plt.subplots(nrows=2, figsize=(6, 6), sharex="col")
    for i, r in enumerate(zip(results["mean_t_per_dt"], results["std_t_per_dt"])):
        # scale all execution times with the execution time of the default settings
        if scale:
            ax[0].scatter(results["time_step"][i][:min_n_dt] / factor,
                          r[0][:min_n_dt] / results["mean_t_per_dt"][default_idx][:min_n_dt], marker=".")
            ax[1].scatter(results["time_step"][i][:min_n_dt] / factor,
                          r[1][:min_n_dt] / results["std_t_per_dt"][default_idx][:min_n_dt], marker=".")
            ax[0].set_ylabel(r"$\mu{(t^*_{exec})}$", fontsize=13)
            ax[1].set_ylabel(r"$\sigma{(t^*_{exec})}$", fontsize=13)

        # no scaling
        else:
            ax[0].scatter(results["time_step"][i][:min_n_dt] / factor, r[0][:min_n_dt], marker=".")
            ax[1].scatter(results["time_step"][i][:min_n_dt] / factor, r[1][:min_n_dt], marker=".")

            ax[0].set_ylabel(r"$\mu{(t_{exec})}$   $[s]$", fontsize=13)
            ax[1].set_ylabel(r"$\sigma{(t_{exec})}$   $[s]$", fontsize=13)

    ax[1].set_xlabel(r"$t \, / \, T$", fontsize=13)
    # ax[0].set_title(r"$'interpolateCorrection'$")
    fig.tight_layout()
    fig.legend(xticks, loc="upper center", framealpha=1.0, ncol=3)
    fig.subplots_adjust(top=0.83)
    if scale:
        plt.savefig(join(save_path, "execution_times_vs_dt.png"), dpi=340)
    else:
        plt.savefig(join(save_path, "execution_times_vs_dt_abs.png"), dpi=340)

    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
