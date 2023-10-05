"""
    compute and plot the auto-correlation of the statistical properties of the residuals
"""
import pandas as pd
import torch as pt
import matplotlib.pyplot as plt

from typing import List
from os.path import join
from seaborn import heatmap
from os import makedirs, path
from statsmodels.tsa.stattools import acf

from get_residuals_from_log import get_GAMG_residuals, map_keys_to_labels


def compute_auto_correlations(data: List[dict], key_list: list) -> list:
    """
    compute the auto-correlation of data for each key and element in the list, it is assumed that each list element is
    a dict containing the data, e.g. as: [{'key1': ..., 'key2': ...}, {'key1': ..., 'key2': ...}, ...]

    :param data: the data of which the auto-correlation should be computed
    :param key_list: a list containing the keys of the dict for which the auto-correlation should be computed
    :return: the auto-correlations wrt the data and keys
    """
    corr = []
    for k in key_list:
        tmp = []
        for i, case in enumerate(data):
            tmp.append(acf(case[k]))
        corr.append(tmp)
    return corr


if __name__ == "__main__":
    # main path to all the cases
    # load_path = join(r"..", "run", "drl", "smoother", "results_weirOverflow")
    load_path = join(r"..", "run", "parameter_study", "influence_solver_settings", "smoother")

    # list with the cases
    cases = ["surfaceMountedCube_DICGaussSeidel", "mixerVesselAMI_DICGaussSeidel",
             "../../../drl/smoother/results_weirOverflow/DICGaussSeidel_local/run_1",
             "../../../drl/smoother/results_cylinder2D/DICGaussSeidel_local/run_1"]

    # legend entries for the plot
    # legend = ["$FDIC$", "$DIC$", "$DICGaussSeidel$", "$symGaussSeidel$", "$nonBlockingGaussSeidel$", "$GaussSeidel$"]
    legend = ["$surfaceMountedCube$", "$mixerVesselAMI$", "$weirOverflow$", "$cylinder2D$"]
    # legend = ["$no$", "$yes$"]

    # save names for correlation plots
    save_name = ["surfaceMountedCube", "mixerVesselAMI", "weirOverflow", "cylinder2D"]

    # position of the legend (in which subplot)
    pos_legend = [1, 2]

    # portrait (wide = false) or landscape (wide = true)
    wide = True

    # save path for the plots
    save_path = join(r"..", "run", "drl", "autocorrelations_new_features", "plots")

    # dictionary keys of the properties of which the auto-correlations should be plotted (old features)
    # keys = ["n_solver_iter", "sum_gamg_iter", "max_gamg_iter", "init_residual", "min_convergence_rate",
    #         "median_convergence_rate", "avg_convergence_rate", "max_convergence_rate"]

    # new features
    keys = ["ratio_solver_iter", "ratio_gamg_iter", "init_residual_scaled", "min_convergence_rate_scaled",
            "median_convergence_rate_scaled", "max_convergence_rate_scaled"]

    assert len(keys) % 2 == 0, "the length of the 'keys'-list needs to be an even number!"

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # load the filtered log data for each case
    log_data = []
    for c in cases:
        if c.startswith("surfaceMountedCube"):
            load_path_tmp = join(load_path, c, "fullCase", "log_data_filtered.pt")
        else:
            load_path_tmp = join(load_path, c, "log_data_filtered.pt")

        try:
            # check if log file was already processed
            log_data.append(pt.load(load_path_tmp))
        except FileNotFoundError:
            # adjust load path
            load_path_tmp = join(load_path_tmp.split("log_data_filtered")[0])

            # else filter log file wrt GAMG & save the data from the log file
            pt.save(get_GAMG_residuals(load_path_tmp), join(load_path_tmp, "log_data_filtered.pt"))
            log_data.append(pt.load(join(load_path_tmp, "log_data_filtered.pt")))

    # compute the new input features for the policy network
    for l in log_data:
        # ratio GAMG iterations
        l["ratio_gamg_iter"] = (l["sum_gamg_iter"] - l["max_gamg_iter"]) / (l["sum_gamg_iter"] + l["max_gamg_iter"])

        # ratio PIMPLE iterations / max. allowed PIMPLE iterations (max. is always 50)
        l["ratio_solver_iter"] = l["n_solver_iter"] / 50

        # scaled initial residual
        l["init_residual_scaled"] = (pt.sigmoid(l["init_residual"]) - 0.5) / 8e-4

        # scaled convergence rates
        l["median_convergence_rate_scaled"] = (pt.sigmoid(l["median_convergence_rate"]) - 0.5) / 1.5e-4
        l["max_convergence_rate_scaled"] = (pt.sigmoid(l["max_convergence_rate"]) - 0.5) / 5e-4
        l["min_convergence_rate_scaled"] = (pt.sigmoid(l["min_convergence_rate"]) - 0.5) / 2e-5

    # compute the auto-correlation
    auto_corr = compute_auto_correlations(log_data, keys)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})

    # plot cross-correlations between the specified keys as heatmap (linear correlations only)
    log_data = [pd.DataFrame.from_dict(l) for l in log_data]

    # remove all unused keys
    unused_keys = [k for k in log_data[0].keys() if k not in keys]
    [l.drop(unused_keys, inplace=True, axis=1) for l in log_data]

    # plot a heatmap for each case
    labels = [map_keys_to_labels(k) for k in keys]
    for i, l in enumerate(log_data):
        fig, ax = plt.subplots(figsize=(7, 6))
        heatmap(l.corr(), vmin=-1, vmax=1, center=0, cmap="Greens", square=True, annot=True, xticklabels=labels,
                yticklabels=labels, cbar=True, linewidths=0.30, linecolor="white", ax=ax, cbar_kws={"shrink": 0.75},
                fmt=".2g")
        fig.tight_layout()
        plt.savefig(join(save_path, f"correlations_{save_name[i]}.png"), dpi=340)
        plt.show(block=False)
        plt.pause(2)
        plt.close(fig)

    # plot auto correlation for the specified keys
    counter = 0
    n_rows = 2 if wide else int(len(auto_corr) / 2)
    n_cols = int(len(auto_corr) / 2) if wide else 2
    figsize = (10, 6) if wide else (6, 10)

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, sharex="all", sharey="all")
    for row in range(n_rows):
        for col in range(n_cols):
            for c in range(len(cases)):
                if [row, col] == pos_legend:
                    ax[row][col].plot(range(len(auto_corr[counter][c])), auto_corr[counter][c], marker="x",
                                      label=legend[c])
                else:
                    ax[row][col].plot(range(len(auto_corr[counter][c])), auto_corr[counter][c], marker="x")
            ax[row][col].set_title(map_keys_to_labels(keys[counter]))

            # it is sufficient to look at the correlations of the 1st 20 time steps
            ax[row][col].set_xlim(0, 20)
            counter += 1
        ax[row][0].set_ylabel(r"$R_{ii}$", fontsize=13)

    for i in range(n_cols):
        ax[-1][i].set_xlabel("$time$ $delay$", fontsize=13)

    fig.tight_layout()
    ax[pos_legend[0]][pos_legend[1]].legend(loc="upper right", framealpha=1.0, ncol=1)
    fig.subplots_adjust(hspace=0.2)
    if wide:
        plt.savefig(join(save_path, f"auto_correlations_residuals_new_features_wide.png"), dpi=340)
    else:
        plt.savefig(join(save_path, f"auto_correlations_residuals_new_features.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
