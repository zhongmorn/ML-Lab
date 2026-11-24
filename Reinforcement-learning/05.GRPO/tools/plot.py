from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def plot_training(training, path):
    updates = [row["update"] for row in training]
    figure = Figure(figsize=(7.6, 5), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    reward_axis, group_axis = figure.subplots(2, 1, sharex=True)

    reward_axis.plot(updates, [row["mean_reward"] for row in training], color="#255A94")
    reward_axis.set(ylabel="Mean reward", title="GRPO training on a four-action task")
    group_axis.plot(updates, [row["mixed_group_rate"] for row in training], color="#176C63")
    group_axis.set(xlabel="Update", ylabel="Mixed groups")

    for axis in (reward_axis, group_axis):
        axis.set_ylim(0, 1.02)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=160)
