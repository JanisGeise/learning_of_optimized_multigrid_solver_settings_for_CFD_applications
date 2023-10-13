"""
    plot the rewards of trajectories of the PPO-training wrt the trajectory length
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs


def load_rewards(load_dir: str) -> list:
    """
    loads and filter the observations_*.pt files of the PPO-training

    :param load_dir: path to the top-level directory of the case for which the results should be loaded
    :return: list containing the rewards wrt episodes and buffer size
    """
    files = sorted(glob(join(load_dir, "observations_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    observations = [pt.load(join(f)) for f in files]

    # remove everything but the rewards and numerical time step for each episode and each trajectory in the buffer
    obs_out = [[traj["rewards"] for traj in o] for o in observations]

    return obs_out


def plot_rewards_vs_trajectory(data: list, episodes: list, save_dir: str, save_name: str = "rewards_vs_traj",
                               n_traj: int = None) -> None:
    """
    plot the rewards vs. trajectory length for the specified episodes

    :param data: loaded rewards
    :param episodes: specified episodes which should be plotted
    :param save_dir: directory to which the plot should be saved to
    :param save_name: save name of the plot
    :param n_traj: how many trajectories of the buffer should be plotted, if None then all available traj. are plotted
    :return: None
    """

    # how many trajectories of the buffer should be plotted for each episode
    n_traj = len(data[0]) if n_traj is None else n_traj

    fig, ax = plt.subplots(nrows=len(episodes), figsize=(6, 8), sharex="col", sharey="col")
    for j, e in enumerate(episodes):
        for buffer in range(n_traj):
            if j == 0:
                ax[j].plot(pt.tensor(range(len(data[e][buffer]))) / len(data[e][buffer]), data[e][buffer],
                           label=f"$trajectory$ $no.$ ${buffer}$")
            else:
                ax[j].plot(pt.tensor(range(len(data[e][buffer]))) / len(data[e][buffer]), data[e][buffer])
        ax[j].set_ylabel("$r$")
        ax[j].set_title(f"$episode$ ${e+1}$")

    ax[-1].set_xlabel(r"$l^*_{traj}$  $[\%]$")
    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1.0, ncol=3)
    fig.subplots_adjust(top=0.88, hspace=0.4)
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # main path to all the cases and save path
    load_path = join("..", "run", "drl", "combined_smoother_and_interpolateCorrection", "results_weirOverflow")
    save_path = join(load_path, "plots")

    # names of top-level directory containing the PPO-trainings
    cases = ["e100_r10_b20_f80_new_features", "e100_r9_b36_f80_new_features_const_sampling"]

    # index of the episodes to plot
    which_episodes = [0, 24, 49, 74, 99]

    # how many trajectories of the buffer should be plotted -> None = all available trajectories in the buffer
    n_t = 6

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load the rewards and time step wrt episode for each case
    rewards = [load_rewards(join(load_path, c)) for c in cases]

    # plot the rewards for some episodes wrt numerical time step for each case
    [plot_rewards_vs_trajectory(t, which_episodes, save_path, f"reward_vs_traj_{cases[i]}", n_traj=n_t) for i, t in
     enumerate(rewards)]
