"""
    compute and plot the auto-correlation of the statistical properties of the residuals
"""
import torch as pt
import matplotlib.pyplot as plt

from os.path import join
from os import makedirs, path
from statsmodels.tsa.stattools import acf
from typing import List

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
            tmp.append(acf(pt.tensor(case[k])))
        corr.append(tmp)
    return corr


if __name__ == "__main__":
    # main path to all the cases
    load_path = join(r"..", "run", "drl", "interpolateCorrection", "results_weirOverflow")

    # list with the cases
    cases = [join("no", "run_1"), join("yes", "run_1")]

    # legend entries for the plot
    legend = ["$no$", "$yes$"]

    # save path for the plots
    save_path = join(load_path, "plots")

    # dictionary keys of the properties of which the auto-correlations should be plotted
    keys = ["n_solver_iter", "sum_gamg_iter", "max_gamg_iter", "init_residual", "min_convergence_rate",
            "median_convergence_rate", "avg_convergence_rate", "max_convergence_rate"]

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
            load_path = join(load_path_tmp.split("log_data_filtered")[0])

            # else filter log file wrt GAMG & save the data from the log file
            pt.save(get_GAMG_residuals(load_path_tmp), join(load_path_tmp, "log_data_filtered.pt"))
            log_data.append(pt.load(join(load_path_tmp, "log_data_filtered.pt")))

    # compute the auto-correlation
    auto_corr = compute_auto_correlations(log_data, keys)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})

    # plot auto correlation for the specified keys
    counter = 0
    fig, ax = plt.subplots(ncols=2, nrows=int(len(auto_corr) / 2), figsize=(6, 10), sharex="all", sharey="all")
    for row in range(int(len(auto_corr) / 2)):
        for col in range(2):
            for c in range(len(cases)):
                if col == 0 and row == 0:
                    ax[row][col].plot(range(len(auto_corr[counter][c])), auto_corr[counter][c], marker="x",
                                      label=legend[c])
                else:
                    ax[row][col].plot(range(len(auto_corr[counter][c])), auto_corr[counter][c], marker="x")
            ax[row][col].set_title(map_keys_to_labels(keys[counter]))
            counter += 1
        ax[row][0].set_ylabel(r"$R_{ii}$", fontsize=13)

    ax[-1][0].set_xlabel("$time$ $delay$", fontsize=13)
    ax[-1][1].set_xlabel("$time$ $delay$", fontsize=13)
    fig.tight_layout()
    ax[0][0].legend(loc="lower right", framealpha=1.0, ncol=2)
    fig.subplots_adjust(hspace=0.2)
    plt.savefig(join(save_path, f"auto_correlations_residuals.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
