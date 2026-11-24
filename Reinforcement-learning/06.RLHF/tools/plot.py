from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def plot_training(reward_training, policy_training, path):
    figure = Figure(figsize=(7.6, 5), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    reward_axis, policy_axis = figure.subplots(2, 1)

    reward_axis.plot(
        [row["epoch"] for row in reward_training],
        [row["loss"] for row in reward_training],
        color="#255A94",
    )
    reward_axis.set(xlabel="Epoch", ylabel="Pairwise loss", title="Reward model")
    policy_axis.plot(
        [row["update"] for row in policy_training],
        [row["sampled_accuracy"] for row in policy_training],
        color="#176C63",
    )
    policy_axis.set(xlabel="Update", ylabel="Sampled accuracy", title="PPO policy", ylim=(0, 1.02))
    policy_axis.yaxis.set_major_formatter(PercentFormatter(1))

    for axis in (reward_axis, policy_axis):
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=160)
