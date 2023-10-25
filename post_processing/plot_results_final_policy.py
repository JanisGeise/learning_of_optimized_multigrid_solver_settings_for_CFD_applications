"""
    plot the total execution times of the simulations and the execution times wrt time step for different settings in
    comparison to the results using e.g. the final (trained) policy of the DRL training. It is assumed that each case /
    setting is run multiple times in order to account for variations within the execution times which are not caused by
    the settings but e.g. by the scheduling of the HPC cluster

    Note: the CPU times are taken in all cases (per time step and for total execution time of the simulation)
"""
import torch as pt
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from typing import Union
from os.path import join
from os import path, makedirs
from pandas import read_csv, DataFrame
from scipy.ndimage import gaussian_filter1d

from post_processing.get_residuals_from_log import get_GAMG_residuals, map_keys_to_labels


def load_cpu_times(case_path: str) -> DataFrame:
    """
    load the time steps, execution time and execution time per time step written out by the 'time' function object
    throughout the simulation

    :param case_path: path to the directory where the results of the simulation are located
    :return: the time steps, execution time and execution time per time step (CPU times only)
    """
    times = read_csv(case_path, sep="\t", comment="#", header=None, names=["t", "t_exec", "t_per_dt"], usecols=[0, 1, 3])
    return times


def load_residuals(case_path: str, new_features: bool = True) -> DataFrame:
    """
    load the properties of the residuals used as policy input wrt time step written out by the 'agentSolverSettings'
    function object throughout the simulation

    :param case_path: path to the directory where the results of the simulation are located
    :param new_features: flag if the cases with policy where run using the new input features (True) or old ones (False)
    :return: the properties of the residuals wrt time steps
    """
    # same names as in 'get_residuals_from_log.py', so we can use the 'map_keys_to_label' function later for plotting,
    # also these keys will be returned if we filter the log file when we didn't use a policy
    if new_features:
        names = ["time", "init_residual", "median_convergence_rate", "max_convergence_rate", "min_convergence_rate",
                 "ratio_gamg_iter", "ratio_solver_iter"]
        cols = [0, 2, 3, 4, 5, 6, 7]
    else:
        names = ["time", "init_residual", "median_convergence_rate", "max_convergence_rate", "min_convergence_rate",
                 "sum_gamg_iter", "max_gamg_iter", "n_solver_iter"]
        cols = [0, 2, 3, 4, 5, 6, 7, 8]
    res = read_csv(case_path, sep="\t", comment="#", header=None, names=names, usecols=cols)

    # in case we didn't use a policy (e.g. for the default simulations used for comparison) we need to compute the
    # properties of the residuals based on solver's log file since there is no 'agentSolverSettings' function object
    if res.empty:
        res = pd.DataFrame.from_dict(get_GAMG_residuals(case_path.split("postProcessing")[0]))

        # if new_features: compute the new features, we always have 50 as max. PIMPLE iterations
        if new_features:
            res["ratio_solver_iter"] = res["n_solver_iter"] / 50
            res["ratio_gamg_iter"] = (res["sum_gamg_iter"] - res["sum_gamg_iter"]) / \
                                     (res["sum_gamg_iter"] + res["sum_gamg_iter"])

        # drop the information we haven't available when using the policy
        res.drop([i for i in res.keys() if i not in names], inplace=True, axis=1)
    return res


def load_trajectory(load_dir: str) -> pt.Tensor or None:
    """
    load the trajectories written out by the 'agentSolverSettings' function object throughout the simulation

    :param load_dir: path to the directory where the results of the simulation are located
    :return: the probabilities of the policy output if a policy was used, else None is returned, each column corresponds
             to one output neuron of the policy network
    """
    try:
        tr = pd.read_table(load_dir, sep=",")

        # we don't need time step (loaded with CPU times) and action (action = category with the highest probability)
        tr.drop(columns=["t"], inplace=True)
        try:
            # in case we have 2 actions (new version of DRL training)
            tr.drop(columns=[" action0", " action1"], inplace=True)
        except KeyError:
            # otherwise there should only be one action
            tr.drop(columns=[" action"], inplace=True)

        # in case we have a policy where 'nCellsInCoarsestLevel' is present, remove the probs and convert the action to
        # a number (otherwise plot is not clear due to too many lines)
        if " action2" in tr:
            tr.drop(columns=[f" prob{i}" for i in range(7, 17)], inplace=True)

            # convert the action to the corresponding 'nFinestSweeps'
            classes = pt.arange(1, 11, 1)
            n_cells = classes[tr[" action2"]]
            tr["n_cells"] = n_cells
            tr.drop(columns=[" action2"], inplace=True)

        # convert to tensor, so we can avg. etc. easier later
        tr = pt.from_numpy(tr.values)

    except FileNotFoundError:
        tr = None

    return tr


def get_mean_and_std_exec_time(load_dir: str, simulations: list) -> dict:
    """
    load the execution times and execution times per time step from a series of simulations and compute the mean
    execution time and the corresponding std. deviation for each solver setting. It is assumed that each solver setting
    is run multiple times (for computing avg. / std. deviation) and each of the runs is located in a subdirectory of
    each case directory

    :param load_dir: path to the top-level directory containing all the simulations which should be processed
    :param simulations: names of the directories of the simulations, it is assumed that each simulation is in a sub-dir
    :return: dict containing the mean t_exec, t_per_dt & n_dt and the corresponding std. deviation
    """
    out_dict = {"t": [], "mean_t_exec": [], "std_t_exec": [], "mean_t_per_dt": [], "std_t_per_dt": [], "mean_n_dt": [],
                "std_n_dt": [], "mean_probs": []}

    for s in simulations:
        tmp, traj_tmp = [], []

        # for each case glob all runs, so we can compute the avg. runtimes etc.
        for i, case in enumerate(glob(join(load_dir, s, "*"))):
            tmp.append(load_cpu_times(join(case, "postProcessing", "time", "0", "timeInfo.dat")))
            traj_tmp.append(load_trajectory(join(case, "trajectory.txt")))

        # sort the results from each case into a dict, assuming we ran multiple cases for each config.
        for key in list(tmp[0].keys()) + ["n_dt", "probs"]:
            if key == "t":
                # all time steps per settings are the same, so just take the 1st one
                out_dict[key].append(tmp[0][key])
            elif key == "t_exec":
                # the total execution time (CPU) is the last value of t_exec (at last time step)
                final_t_exec = pt.cat([pt.from_numpy(i[key].values)[-1].unsqueeze(-1) for i in tmp])
                out_dict[f"mean_{key}"].append(pt.mean(final_t_exec))
                out_dict[f"std_{key}"].append(pt.std(final_t_exec))

            elif key == "t_per_dt":
                # in some cases, the amount of dt might differ, although the same setup, seed etc. was used
                # in that case we take the first N dt which are available for all cases because the dt plot is a
                # qualitative comparison anyway
                min_dt = min([i[key].size for i in tmp])

                out_dict[f"mean_{key}"].append(pt.mean(pt.stack([pt.from_numpy(i[key].values)[:min_dt] for i in tmp]),
                                                       dim=0))
                out_dict[f"std_{key}"].append(pt.std(pt.stack([pt.from_numpy(i[key].values)[:min_dt] for i in tmp]),
                                                     dim=0))

            # amount of time steps within the simulations, std. should be zero if same settings where used
            elif key == "n_dt":
                out_dict[f"mean_{key}"].append(pt.mean(pt.tensor([float(i["t"].size) for i in tmp])))
                out_dict[f"std_{key}"].append(pt.std(pt.tensor([float(i["t"].size) for i in tmp])))

            # compute the mean probability for each decision (e.g. for each smoother). Std. dev. should be zero if all
            # simulations per setting are initialized with same seed value and run with same policy, so just save mean
            elif key == "probs":
                # we only have a trajectory if we used a policy, for the smoother benchmarks skip (we append None, so
                # in that case check if the first list is empty)
                if traj_tmp[0] is None:
                    # appending None makes it easier to plot later
                    out_dict[f"mean_{key}"].append(None)
                else:
                    out_dict[f"mean_{key}"].append(pt.mean(pt.stack([i for i in traj_tmp], dim=-1), dim=-1))

            else:
                continue

    # if we have nCellsInCoarsestLevel, then put that in its own field
    tmp = []
    for i, o in enumerate(out_dict["mean_probs"]):
        if o is not None and o.size()[1] > 7:
            tmp.append(o[:, -1])
            out_dict["mean_probs"][i] = o[:, :7]
        else:
            tmp.append(None)
    out_dict["n_cells"] = tmp

    return out_dict


def plot_nFinestSweeps(n_cells: list, times: list, save_dir: str, sf: float = 1, legend: list = None) -> None:
    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # in case no legend is given, use empty strings
    legend = len(n_cells) * [""] if legend is None else legend

    fig, ax = plt.subplots(figsize=(7, 4))
    set_label = False
    for c in range(len(n_cells)):
        if n_cells[c] is not None:
            if not set_label:
                ax.scatter(times[c] / sf, n_cells[c], color=color[c], label=legend[c], marker=".")
                continue
            else:
                ax.scatter(times[c] / sf, n_cells[c], color=color[c], marker=".")
        else:
            continue
        set_label = True
    ax.set_xlabel(r"$t \, / \, T$", fontsize=13)
    ax.set_ylabel(r"$nFinestSweeps$", fontsize=13)
    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=2)
    fig.subplots_adjust(top=0.86)
    plt.savefig(join(save_dir, f"nFinestSweeps.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_avg_exec_times_final_policy(data, keys: list = ["mean_t_exec", "std_t_exec"], save_dir: str = "",
                                     default: int = None, save_name: str = "mean_execution_times", ylabel: str = None,
                                     scale_wrt_default: bool = True, xlabels: list = None) -> None:
    """
    create an errorbar plot for the execution times (or other quantities). If the std. deviation of a case is zero,
    then the case is plotted as scatter plot. If 'scale_wrt_default' is set to 'True', but no index of the default
    setting is passed, then 'scale_wrt_default' will be set to 'False' the absolute times (or other quantities) are
    plotted.

    :param data: the data loaded with the 'get_mean_and_std_exec_time' function
    :param keys: the keys of the dict which should be plotted, it is assumed that the 1. is the mean and 2. is std. dev.
    :param save_dir: directory to which the plots should be saved to
    :param default: index of the default setting, if given all y-values are scaled wrt this value
    :param save_name: name of the plot
    :param ylabel: y-label for the y-axis
    :param scale_wrt_default: flag if all y-values should be scaled wrt a default case / setting
    :param xlabels: list containing the labels for the x-axis
    :return: None
    """

    if ylabel is None and scale_wrt_default:
        ylabel = r"$t^*_{exec}$"
    elif ylabel is None and not scale_wrt_default:
        ylabel = r"$t_{exec}$   $[s]$"
    else:
        pass

    scale_wrt_default = False if default is None else scale
    xlabels = [f"case #{i}" for i in range(len(data[keys[0]]))] if xlabels is None else xlabels

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(zip(data[keys[0]], data[keys[1]])):
        # scale all execution times with the execution time of the default settings
        if scale_wrt_default:
            # if we don't have a std. (e.g. N_dt), then just make a scatter plot
            if r[1] == 0:
                ax.scatter(xlabels[i], r[0] / data[keys[0]][default], marker="o", alpha=1, color=color[i])
            else:
                ax.errorbar(xlabels[i], r[0] / data[keys[0]][default], yerr=r[1] / data[keys[1]][default],
                            barsabove=True, fmt="o", capsize=5, color=color[i])
            ax.set_ylabel(ylabel, fontsize=13)

        # no scaling
        else:
            # if we don't have a std. (e.g. N_dt), then just make a scatter plot
            if r[1] == 0:
                ax.scatter(xlabels[i], r[0], marker="o", alpha=1, color=color[i], facecolors=color[i])
            else:
                ax.errorbar(xlabels[i], r[0], yerr=r[1], barsabove=True, fmt="o", capsize=5, color=color[i])
            ax.set_ylabel(ylabel, fontsize=13)
    fig.tight_layout()
    ax.grid(visible=True, which="major", linestyle="-", alpha=0.45, color="black", axis="y")
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(visible=True, which="minor", linestyle="--", alpha=0.35, color="black", axis="y")
    if scale_wrt_default:
        plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    else:
        plt.savefig(join(save_dir, f"{save_name}_abs.png"), dpi=340)

    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_probabilities(probs: list, time_steps, save_dir: str = "", save_name: str = "probabilities_vs_dt",
                       sf: float = 1, param: Union[str, list] = "smoother", legend: list = None) -> None:
    """
    plot the loaded probabilities with respect to the time step and simulation

    :param probs: loaded probabilities of the policy output for each case
    :param time_steps: loaded time steps run for each case
    :param save_dir: directory to which the plots should be saved to
    :param save_name: name of the plot
    :param sf: scaling factor for scaling the x-axis (non-dimensionalizing the time step)
    :param param: labels for the probabilities, if 'smoother' the probabilities correspond to all available smoother
    :param legend: list containing the legend entries
    :return: None
    """
    if param == "smoother":
        label = ["$FDIC$", "$DIC$", "$DICGaussSeidel$", "$symGaussSeidel$", "$nonBlockingGaussSeidel$", "$GaussSeidel$"]
    elif param is not None:
        if type(param) != list:
            label = [param]
        else:
            label = param
    else:
        label = 20 * [""]
    # determine how many cases we have, which are using a policy (otherwise we don't have probabilities to plot)
    n_traj = sum([1 for i, p in enumerate(probs) if p is not None])
    counter, set_legend = 0, False
    xmax = round(max([dt.tail(1).item() for i, dt in enumerate(time_steps) if probs[i] is not None]) / sf, 0)

    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsfonts}"})

    fig, ax = plt.subplots(nrows=n_traj, figsize=(8, 3*n_traj), sharey="col", sharex="col")

    for i, p in enumerate(probs):
        # reset color cycle for each case, so that all probs have the same color
        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        if p is not None:
            # loop over all available probabilities
            for j in range(p.size()[1]):
                if not set_legend:
                    ax[counter].plot(time_steps[i] / sf, gaussian_filter1d(p[:, j], 5), color=color[j], label=label[j])

                    # for interpolateCorrection, add a horizontal line at p = 0.5 = decision boundary
                    if counter == 0 and j == 0:
                        ax[counter].hlines(0.5, 0, xmax, color="red", ls="-.", label="$\mathbb{P} = 0.5$")
                    else:
                        ax[counter].hlines(0.5, 0, xmax, color="red", ls="-.")
                else:
                    ax[counter].plot(time_steps[i] / sf, gaussian_filter1d(p[:, j], 5), color=color[j])

                    # for interpolateCorrection, add a horizontal line at p = 0.5 = decision boundary
                    ax[counter].hlines(0.5, 0, xmax, color="red", ls="-.")

                ax[counter].set_xlim(0, xmax)
                ax[counter].set_yscale("log")
                ax[counter].set_ylabel(r"$\mathbb{P}$", fontsize=13)
                ax[counter].annotate(legend[i], xy=(xmax + xmax * 0.025, 0.5), fontsize=13,
                                     xycoords=ax[counter].get_xaxis_transform())
            counter += 1
            set_legend = True

    ax[-1].set_xlabel(r"$t \, / \, T$", fontsize=13)
    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=3)
    fig.subplots_adjust(top=0.92)
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def compare_residuals(load_dir: str, simulations: list, save_dir: str, sf: float = 1, legend: list = None) -> None:
    # all runs have the same residuals since we used the same policy, seed, starting settings etc., so just take
    # randomly the one available and load the residuals for each case
    file_dir = join("postProcessing", "residuals", "0", "agentSolverSettings.dat")
    residuals = [load_residuals(join(glob(join(load_dir, s, "*"))[0], file_dir)) for s in simulations]

    # take 6 of the 7 features and plot them ('n_pimple_iter' not changing wrt solver settings, so no need to plot it)
    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # in case no legend is given, use empty strings
    legend = len(residuals) * [""] if legend is None else legend

    # we want to plot everything but the numerical time and N pimple iter as parameter for the y-axis
    keys = [k for k in residuals[0].keys() if k != "time" and k != "n_solver_iter"]

    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})

    counter = 0
    xmax = round(max([dt["time"].tail(1).item() for i, dt in enumerate(residuals)]) / sf, 0)

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(7, 8), sharex="all")
    for row in range(3):
        for col in range(2):
            for r in range(len(residuals)):
                if col == 0 and row == 0:
                    ax[row][col].plot(residuals[r]["time"] / sf, gaussian_filter1d(residuals[r][keys[counter]], 5),
                                      label=legend[r], color=color[r])
                else:
                    ax[row][col].plot(residuals[r]["time"] / sf, gaussian_filter1d(residuals[r][keys[counter]], 5),
                                      color=color[r])

            ax[row][col].set_ylabel(map_keys_to_labels(keys[counter]))
            ax[row][col].set_xlim(0, xmax)
            ax[-1][col].set_xlabel(r"$t \, / \, T$", fontsize=13)

            if keys[counter] != "max_gamg_iter" and keys[counter] != "min_convergence_rate" and not "ratio" in keys[counter]:
                ax[row][col].set_yscale("log")

            counter += 1

    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=2)
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_dir, f"comparison_residuals.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # main path to all the cases and save path
    load_path = join("..", "run", "drl", "combined_smoother_interpolateCorrection_nFinestSweeps",
                     "results_cylinder2D")
    save_path = join(load_path, "plots")

    # names of top-level directory containing the simulations run with different settings
    cases = ["default_settings_no_policy", "nonBlockingGaussSeidel_local",
             # "DIC_local",
             "trained_policy_b16_PPO_every_2nd_dt_validation_every_dt",
             "trained_policy_b16_PPO_every_2nd_dt_validation_every_dt_local",
             "trained_policy_b16_PPO_every_10th_dt_validation_every_dt"]

    # xticks for the plots
    xticks = ["$DICGaussSeidel$\n$(no$ $policy)$", "$nonBlocking$\n$GaussSeidel$\n$(no$ $policy)$",
              # "$DIC$\n$(no$ $policy)$",
              "$final$ $policy$\n$(2$ $\Delta t, HPC)$", "$final$ $policy$\n$(2$ $\Delta t, local)$",
              "$final$ $policy$\n$(10$ $\Delta t, HPC)$", "$final$ $policy$\n$(10$ $\Delta t, local)$"]

    # which case contains the default setting -> used for scaling the execution times
    default_idx = 0

    # flag if the avg. execution time and corresponding std. deviation should be scaled wrt default setting
    scale = True

    # scaling factor for num. time, here: approx. period length of vortex shedding frequency @ Re = 1000
    factor = 1 / 20

    # factor for weirOverflow case
    # factor = 1 / 0.4251

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    results = get_mean_and_std_exec_time(load_path, cases)

    # don't plot probabilities of actions or 'nCellsInCoarsestLevel' for random policy incase we have any
    try:
        results["mean_probs"][cases.index("random_policy")] = None
        results["n_cells"][cases.index("random_policy")] = None
    except ValueError:
        pass

    # plot the properties of the residuals wrt time step and case,replace all new lines in the legend with spaces if
    # present, because otherwise the legend is too big
    compare_residuals(load_path, cases, save_dir=save_path, sf=factor, legend=[i.replace("\n", " ") for i in xticks])

    # plot the avg. execution times and the corresponding std. deviation
    plot_avg_exec_times_final_policy(results, save_dir=save_path, scale_wrt_default=scale, default=default_idx,
                                     xlabels=xticks)

    # plot the avg. amount of time steps without the std. deviation (std. dev. is zero for same settings)
    plot_avg_exec_times_final_policy(results, keys=["mean_n_dt", "std_n_dt"], ylabel=r"$N_{\Delta t}$",
                                     save_dir=save_path, scale_wrt_default=scale, default=default_idx,
                                     save_name="mean_n_dt", xlabels=xticks)

    # plot the probability (policy output) wrt time step and setting (e.g. probability for each available smoother)
    plot_probabilities(results["mean_probs"], results["t"], save_dir=save_path, sf=factor, legend=xticks,
                       param=["$no$ $if$ $\mathbb{P} \le 0.5,$ $else$ $yes$", "$FDIC$", "$DIC$", "$DICGaussSeidel$",
                              "$symGaussSeidel$", "$nonBlockingGaussSeidel$", "$GaussSeidel$"])

    # plot 'nCellsInCoarsestLevel' if it is available
    if results["n_cells"]:
        plot_nFinestSweeps(results["n_cells"], results["t"], save_dir=save_path, sf=factor,
                           legend=[i.replace("\n", " ") for i in xticks])

    # make sure all cases have the same amount of time steps as the default case, if not then take the 1st N time steps
    # which are available for all cases (difference for 'weirOverflow' is ~10 dt and therefore not visible anyway)
    min_n_dt = min([len(i) for i in results["t"]])

    # plot the execution time wrt time step
    fig, ax = plt.subplots(nrows=2, figsize=(6, 6), sharex="col")
    for i, r in enumerate(zip(results["mean_t_per_dt"], results["std_t_per_dt"])):
        # scale all execution times with the execution time of the default settings
        if scale:
            ax[0].scatter(results["t"][i][:min_n_dt] / factor,
                          r[0][:min_n_dt] / results["mean_t_per_dt"][default_idx][:min_n_dt], marker=".")
            ax[1].scatter(results["t"][i][:min_n_dt] / factor,
                          r[1][:min_n_dt] / results["std_t_per_dt"][default_idx][:min_n_dt], marker=".")
            ax[0].set_ylabel(r"$\mu{(t^*_{exec})}$", fontsize=13)
            ax[1].set_ylabel(r"$\sigma{(t^*_{exec})}$", fontsize=13)

        # no scaling
        else:
            ax[0].scatter(results["t"][i][:min_n_dt] / factor, r[0][:min_n_dt], marker=".")
            ax[1].scatter(results["t"][i][:min_n_dt] / factor, r[1][:min_n_dt], marker=".")

            ax[0].set_ylabel(r"$\mu{(t_{exec})}$   $[s]$", fontsize=13)
            ax[1].set_ylabel(r"$\sigma{(t_{exec})}$   $[s]$", fontsize=13)
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")

    ax[1].set_xlabel(r"$t \, / \, T$", fontsize=13)
    fig.tight_layout()

    # replace all new lines in the legend with spaces if present, because otherwise the legend is too big
    xticks = [i.replace("\n", " ") for i in xticks]

    fig.legend(xticks, loc="upper center", framealpha=1.0, ncol=2)
    fig.subplots_adjust(top=0.86)
    if scale:
        plt.savefig(join(save_path, "execution_times_vs_dt.png"), dpi=340)
    else:
        plt.savefig(join(save_path, "execution_times_vs_dt_abs.png"), dpi=340)

    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
