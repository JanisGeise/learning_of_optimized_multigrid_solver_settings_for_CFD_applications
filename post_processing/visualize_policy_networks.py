"""
    script for visualizing the policy network architecture at different stages of the thesis (qualitatively)

    if torchviz should be used for a detailed plot of the policies, it needs to be installed first as:
        - local installation of graphviz, run the command 'sudo apt-get install graphviz'
        - installation of the torchviz package, run the command 'pip install torchviz'
        - uncomment the function call 'render_policies_torchviz'

    otherwise just comment 'import torchviz' out
"""
import torchviz
import torch as pt
import matplotlib.pyplot as plt

from os.path import join
from os import path, makedirs

from matplotlib.patches import Circle, Rectangle


class PolicyInterpolateCorrectionOnly(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self._n_states = 7
        self._n_actions = 1
        self._n_layers = 2
        self._n_neurons = 64
        self._activation = pt.nn.functional.relu
        self._n_output = 1

        # set up policy network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(self._n_neurons, self._n_neurons))
                self._layers.append(pt.nn.LayerNorm(self._n_neurons))
        self._last_layer = pt.nn.Linear(self._n_neurons, self._n_output)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for layer in self._layers:
            x = self._activation(layer(x))

        # map to intervall [0, 1] since we want to output a binary probability
        return pt.sigmoid(self._last_layer(x))


class PolicySmootherOnly(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self._n_states = 7
        self._n_actions = 1
        self._n_layers = 2
        self._n_neurons = 64
        self._activation = pt.nn.functional.relu
        self._n_output = 6

        # set up policy network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(self._n_neurons, self._n_neurons))
                self._layers.append(pt.nn.LayerNorm(self._n_neurons))
        self._last_layer = pt.nn.Linear(self._n_neurons, self._n_output)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for layer in self._layers:
            x = self._activation(layer(x))

        # classification for smoother, all probabilities add up to 1
        return pt.nn.functional.softmax(self._last_layer(x), dim=1)


class PolicyInterpolateCorrectionAndSmoother(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self._n_states = 7
        self._n_actions = 2
        self._n_layers = 2
        self._n_neurons = 64
        self._activation = pt.nn.functional.relu
        self._n_output = 7

        # set up policy network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(self._n_neurons, self._n_neurons))
                self._layers.append(pt.nn.LayerNorm(self._n_neurons))
        self._last_layer = pt.nn.Linear(self._n_neurons, self._n_output)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for layer in self._layers:
            x = self._activation(layer(x))

        # map the 1st output to intervall [0, 1] since we want a binary probability (interpolateCorrection)
        bin_choice = pt.sigmoid(self._last_layer(x)[:, 0]).unsqueeze(0)

        # use the remaining output neurons for classification for smoother, all probabilities add up to 1
        # transpose because dim 0 can be different when cat (bin_choice & classification), but here dim 1 is different
        classification = pt.nn.functional.softmax(self._last_layer(x)[:, 1:], dim=1).transpose(0, 1)

        # the output size of pt.cat([bin_choice,  classification], dim=0) = [7, len_traj],
        # but we want [len_traj, 7], so transpose
        return pt.cat([bin_choice, classification], dim=0).transpose(0, 1)


def render_policies_torchviz(save_dir: str) -> None:
    """
    plot the architecture of the policy network using 'torchviz' for each configuration of the policy

    :param save_dir: path to the directory where the plots should be saved to
    :return: None
    """
    # test input for tracing the network architecture, the amount of features remains constant for all policies
    test_input = pt.rand((1, 7))

    # policy network for 'interpolateCorrection' only
    policy = PolicyInterpolateCorrectionOnly()
    dot = torchviz.make_dot(policy.forward(test_input), params=dict(policy.named_parameters()))
    dot.render("policy_interpolateCorrection", save_dir, format="png", cleanup=True)

    # policy network for 'smoother' only
    policy = PolicySmootherOnly()
    dot = torchviz.make_dot(policy.forward(test_input), params=dict(policy.named_parameters()))
    dot.render("policy_smoother", save_dir, format="png", cleanup=True)

    # policy network for 'interpolateCorrection' only
    policy = PolicyInterpolateCorrectionAndSmoother()
    dot = torchviz.make_dot(policy.forward(test_input), params=dict(policy.named_parameters()))
    dot.render("policy_interpolateCorrection_and_smoother", save_dir, format="png", cleanup=True)


def plot_policies(save_dir: str, n_inputs: int = 7, n_outputs: int = 1, save_name=None) -> None:
    """
    plot a qualitative network architecture of the policy visualizing input & output quantities

    :param save_dir: path to the directory where the plots should be saved to
    :param n_inputs: number of input neurons (number of features), is const. for all policies
    :param n_outputs: number of output neurons
    :param save_name: name of the plot
    :return: None
    """

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts}"})

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.set_facecolor("white")

    # radius of neurons
    r = 0.02

    # names of the available smoother, GS = GaussSeidel, otherwise names are too long...
    msg = ["$\mathbb{P}(GS)$", "$\mathbb{P}(nonBlockingGS)$", "$\mathbb{P}(symGS)$", "$\mathbb{P}(DICGS)$",
           "$\mathbb{P}(DIC)$", "$\mathbb{P}(FDIC)$"]

    # names of labels for policy input
    if n_inputs == 6:
        names = [r"$\frac{N_{PIMPLE}}{N_{PIMPLE, max}}$",
                 r"$\frac{\sum{N_{GAMG}-N_{GAMG, \, max}}}{\sum{N_{GAMG} + N_{GAMG, \, max}}}$",
                 "$sigmoid(|\Delta \\boldsymbol{R}_{median}|)^*$", "$-ln(\\boldsymbol{R}_0)^*$",
                 "$-ln(|\Delta \\boldsymbol{R}_{max}|)^*$", "$sigmoid(|\Delta \\boldsymbol{R}_{min}|)^*$"]
    else:
        names = [r"$N_{PIMPLE}$", r"$N_{GAMG, \, max}$", r"$\sum{N_{GAMG}}$",
                 "$|ln(|\Delta \\boldsymbol{R}_{median}|)|$", "$|ln(\\boldsymbol{R}_0)|$",
                 "$|ln(|\Delta \\boldsymbol{R}_{max}|)|$", "$|ln(|\Delta \\boldsymbol{R}_{min}|)|$"]

    # add neurons of input layer
    rectangle = Rectangle((-2 * r, -r), width=4*r, height=1+2*r, edgecolor="grey", facecolor="grey", alpha=0.4)
    ax.add_patch(rectangle)
    ax.annotate(f"$input$\n$layer$\n$(1 \\times {n_inputs})$", (r, -0.48), annotation_clip=False, color="black",
                ha="center")
    pos_in = pt.linspace(r, 1-r, n_inputs)
    pos_out = pt.linspace(r, 1-r, n_outputs)
    for n in range(1, n_inputs+1):
        circle = Circle((0, pos_in[n-1]), radius=r, color="green", zorder=10)
        ax.add_patch(circle)

        # add label for each input parameter
        ax.arrow(-7 * r, pos_in[-n], 3*r, 0, color="green", head_width=0.02, clip_on=False, overhang=0.3)

        if n_inputs == 6:
            ax.annotate(names[n - 1], (-42 * r, pos_in[-n]), annotation_clip=False, color="green")
        else:
            ax.annotate(names[n-1], (-35 * r, pos_in[-n]), annotation_clip=False, color="green")

    # add rectangles for the two hidden layers
    rectangle = Rectangle((-r + 0.4, -r - 0.2), width=4*r, height=1.4+2*r, edgecolor="grey", facecolor="grey", alpha=0.4)
    ax.add_patch(rectangle)
    ax.annotate("$hidden$\n$layer$\n$(1 \\times 64)$", (0.4, -0.48), annotation_clip=False, color="black", ha="center")

    rectangle = Rectangle((-r + 0.8, -r - 0.2), width=4*r, height=1.4+2*r, edgecolor="grey", facecolor="grey", alpha=0.4)
    ax.add_patch(rectangle)
    ax.annotate("$hidden$\n$layer$\n$(1 \\times 64)$", (0.8, -0.48), annotation_clip=False, color="black", ha="center")

    # add a few (here: 8) neurons for each hidden layer
    pos_h = list(pt.linspace(0.8, 1.2 - r, 4)) + list(pt.linspace(-0.2 + r, 0.2, 4))
    for n in range(len(pos_h)):
        # 1st hidden layer
        circle = Circle((0.4 + r, pos_h[n]), radius=r, color="black", zorder=10)
        ax.add_patch(circle)
        # 2nd hidden layer
        circle = Circle((0.8 + r, pos_h[n]), radius=r, color="black", zorder=10)
        ax.add_patch(circle)

    ax.annotate("$.$\n$.$\n$.$", (0.4 + r/2, 0.5), annotation_clip=False, color="black", fontsize=20, va="center")
    ax.annotate("$.$\n$.$\n$.$", (0.8 + r/2, 0.5), annotation_clip=False, color="black", fontsize=20, va="center")

    # connect input layer with 1st hidden layer
    y = [[(i, k) for k in pos_h] for i in pos_in]
    [[ax.plot((r, 0.4), k, color="grey", lw=0.5) for k in i] for i in y]

    # connect the two hidden layers
    y = [[(i, k) for k in pos_h if i != k] for i in pos_h]
    [[ax.plot((0.4 + 2 * r, 0.8), k, color="grey", lw=0.5) for k in i] for i in y]

    # output layer, n_outputs = 1 corresponds to 'interpolateCorrection'
    if n_outputs == 1:
        rectangle = Rectangle((-r + 1.2, pt.mean(pos_in) + r), width=4*r, height=4*n_outputs*r,
                              edgecolor="grey", facecolor="grey", alpha=0.4)
        ax.add_patch(rectangle)

        # add the output neuron and some annotations
        circle = Circle((1.2 + r, pt.mean(pos_in) + 3 * r), radius=r, color="red", zorder=10)
        ax.add_patch(circle)

        ax.arrow(1.2 + 6 * r, pt.mean(pos_in) + 3 * r, 14 * r, 0, color="red", head_width=0.02,  clip_on=False,
                 overhang=0.3)
        ax.annotate("$sigmoid$", (1.2 + 6 * r, pt.mean(pos_in) + 5 * r), annotation_clip=False, color="red")
        ax.annotate("$\mathbb{P} \in [0, 1]$", (1.2 + 24 * r, pt.mean(pos_in) + 2 * r), annotation_clip=False,
                    color="red")

        # connect last hidden layer with output neuron
        [ax.plot([0.8+2*r, 1.2], [i, pt.mean(pos_in) + 3 * r], color="grey", lw=0.5) for i in pos_h]

        # add small legend
        ax.annotate("$\\boldsymbol{R} \equiv Residual$", (0.3, 1.45), annotation_clip=False, color="green")
        ax.annotate("$\mathbb{P} \: \equiv probability$", (0.3, 1.35), annotation_clip=False, color="red")

        save_name = "policy_network_interpolateCorrection"

    # n_outputs = 6 corresponds to 'smoother'
    elif n_outputs == 6:
        h = 5 * r * n_outputs + pos_in.mean()
        rectangle = Rectangle((-r + 1.2, pos_in.mean()-r-h/2), width=4*r, height=h, edgecolor="grey", facecolor="grey",
                              alpha=0.4)
        ax.add_patch(rectangle)

        # draw the output neurons
        for n in range(n_outputs):
            circle = Circle((1.2 + r, pos_out[n] + pos_out.mean()-h/2), radius=r, color="red", zorder=10)
            ax.add_patch(circle)

        # connect last hidden layer with output layer
        y = [[(i, k + pos_out.mean()-h/2) for k in pos_out] for i in pos_h]
        [[ax.plot((0.8+r, 1.2), k, color="grey", lw=0.5) for k in i] for i in y]

        # annotate the output neurons
        [ax.annotate(msg[m], (1.2 + 4 * r, pos_out[m] + pos_out.mean() - r - h/2), annotation_clip=False, color="red")
         for m in range(n_outputs)]

        # add legend
        ax.annotate("$\\boldsymbol{R} \enspace \equiv Residual$", (0.3, 1.45), annotation_clip=False, color="green")
        ax.annotate("$\mathbb{P} \enspace \; \equiv probability$\n$GS \equiv GaussSeidel$", (0.3, 1.3),
                    annotation_clip=False, color="red")

        save_name = "policy_network_smoother"

    # n_outputs = 7 corresponds to combination of 'interpolateCorrection' and 'smoother'
    elif n_outputs == 7:
        rectangle = Rectangle((-r + 1.2, -r), width=4 * r, height=1 + 2 * r, edgecolor="grey", facecolor="grey",
                              alpha=0.4)
        ax.add_patch(rectangle)

        # draw the output neurons
        for n in range(n_outputs):
            circle = Circle((1.2 + r, pos_out[n]), radius=r, color="red", zorder=10)
            ax.add_patch(circle)

        # connect last hidden layer with output layer
        y = [[(i, k) for k in pos_out] for i in pos_h]
        [[ax.plot((0.8+r, 1.2), k, color="grey", lw=0.5) for k in i] for i in y]

        # drax the sigmoid annotation for the 1st output neuron
        ax.annotate("$\mathbb{P} \in [0, 1]$", (1.2 + 4 * r, pos_out[-1] - r), annotation_clip=False, color="red")

        # annotate the remaining output neurons
        [ax.annotate(msg[m], (1.2 + 4 * r, pos_out[m]-r), annotation_clip=False, color="red") for m in range(len(pos_out[:-1]))]

        # add legend
        ax.annotate("$\\boldsymbol{R} \enspace \equiv Residual$", (0.3, 1.45), annotation_clip=False, color="green")
        ax.annotate("$\mathbb{P} \enspace \; \equiv probability$\n$GS \equiv GaussSeidel$", (0.3, 1.3),
                    annotation_clip=False, color="red")

        save_name = "policy_network_interpolateCorrection_and_smoother" if save_name is None else save_name

    ax.annotate("$output$\n$layer$\n$(1 \\times$" + f"${n_outputs}$" + "$)$", (1.2 + r, -0.48), annotation_clip=False,
                color="black", ha="center")

    plt.axis("off")
    plt.axis("equal")
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # save directory for the plots
    save_path = join(r"..", "run", "drl", "policy_design")

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # for a detailed insight using 'torchviz' uncomment (for presentations this is kinda overkill...)
    # render_policies_torchviz(save_path)

    # plot the policy networks more qualitatively, here for 'interpolateCorrection'
    plot_policies(save_path)

    # for 'smoother'
    plot_policies(save_path, n_outputs=6)

    # for combination of 'interpolateCorrection' and 'smoother'
    plot_policies(save_path, n_outputs=7)

    # for combination of 'interpolateCorrection' and 'smoother', new input features
    plot_policies(save_path, n_inputs=6, n_outputs=7,
                  save_name="policy_network_interpolateCorrection_and_smoother_new_features")

