"""
    load and plot the results of the PPO-training
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from typing import Union
from os import path, makedirs


def load_rewards(load_dir: str) -> dict:
    """
    loads the observations_*.pt files and computes the mean and std. deviation of the results wrt episodes

    :param load_dir: path to the top-level directory of the case for which the results should be loaded
    :return: dict containing the mean rewards, cl & cd values along with their corresponding std. deviation
    """
    files = sorted(glob(join(load_dir, "observations_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    obs = [pt.load(join(f)) for f in files]
    params = ["rewards", "actions", "t_per_dt"]
    obs_out = {}

    # add keys for mean and std. deviation to dict
    for key in params:
        obs_out.update({f"{key}_mean": []})
        obs_out.update({f"{key}_std": []})

    for episode in range(len(obs)):
        # get the min. amount of time steps for each episode (in case dt != const. we can't just stack the tensors)
        # omit all failed trajectories
        n_dt = [i["rewards"].size()[0] if "rewards" in i else None for i in obs[episode]]
        tmp = pt.zeros((sum([i for i in n_dt if i is not None]), ))

        for key in params:
            for i, runner in enumerate(obs[episode]):
                # we want the quantities' avg. wrt episode, so it's ok to stack all trajectories in one tensor
                # since we avg. over all of them anyway
                if n_dt[i] is not None:
                    if i == 0:
                        tmp[:n_dt[i]] = obs[episode][i][key]
                    else:
                        start = sum([i for i in n_dt[:i] if i is not None])
                        end = sum([i for i in n_dt[:i+1] if i is not None])
                        tmp[start:end] = obs[episode][i][key]
                else:
                    continue

            # compute the mean and st. deviation wrt episode
            obs_out[f"{key}_mean"].append(pt.mean(tmp).item())
            obs_out[f"{key}_std"].append(pt.std(tmp).item())

    return obs_out


def resort_results(data):
    """
    resort the loaded results from list(dict) to dict(list) in order to plot the results easier / more efficient
    :param data: the loaded results from the trainings
    :return: the resorted data
    """
    data_out = {}
    for key in list(data[0].keys()):
        data_out[key] = [i[key] for i in results]

    return data_out


def plot_rewards_vs_episode(reward_mean: Union[list, pt.Tensor], reward_std: Union[list, pt.Tensor],
                            n_cases: int = 0, save_dir: str = "plots", legend_list: list = None) -> None:
    """
    plots the mean rewards received throughout the training periode and the corresponding standard deviation

    :param reward_mean: mean rewards received over the training periode
    :param reward_std: corresponding standard deviation of the rewards received over the training periode
    :param n_cases: number of cases to compare (= number of imported data)
    :param save_dir: path to which directory the plot should be saved
    :param legend_list: list containing the legend entries
    :return: None
    """
    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig, ax = plt.subplots(nrows=2, figsize=(6, 4), sharex="col")
    for i in range(2):
        for c in range(n_cases):
            if i == 0:
                if legend_list:
                    ax[i].plot(range(len(reward_mean[c])), reward_mean[c], color=color[c], label=legend_list[c])
                else:
                    ax[i].plot(range(len(reward_mean[c])), reward_mean[c], color=color[c], label=f"case {c}")
                ax[i].set_ylabel(r"$\mu(r)$")

            else:
                if legend_list:
                    ax[i].plot(range(len(reward_std[c])), reward_std[c], color=color[c], label=legend_list[c])
                else:
                    ax[i].plot(range(len(reward_std[c])), reward_std[c], color=color[c], label=f"case {c}")
                ax[i].set_ylabel(r"$\sigma(r)$")

            ax[i].set_xlim(0, max([len(i) for i in reward_mean]))

    ax[1].set_xlabel("$e$")
    fig.tight_layout()
    ax[0].legend(loc="lower left", framealpha=1.0, ncol=4)
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(join(save_dir, "rewards_vs_episode.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # main path to all the cases and save path
    load_path = join("..", "run", "drl", "smoother", "results_weirOverflow")
    save_path = join(load_path, "plots")

    # names of top-level directory containing the PPO-trainings
    cases = ["e100_r5_b5_f70_cluster_fixed_dt_issue", "e100_r5_b10_f70_cluster_fixed_dt_issue",
             "e100_r5_b20_f70_cluster_fixed_dt_issue"]

    # legend entries for the plots
    legend = ["$b = 5$", "$b = 10$", "$b = 20$"]

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load the results of the PPO-training
    results = [load_rewards(join(load_path, c)) for c in cases]

    # re-arrange for easier plotting
    results = resort_results(results)

    # plot mean rewards wrt to episode and the corresponding std. deviation
    plot_rewards_vs_episode(reward_mean=results["rewards_mean"], reward_std=results["rewards_std"],
                            n_cases=len(cases), legend_list=legend, save_dir=save_path)
