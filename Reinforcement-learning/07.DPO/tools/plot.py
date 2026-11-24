from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def plot_training(training, path):
    epochs = [row["epoch"] for row in training]
    figure = Figure(figsize=(7.6, 5), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    loss_axis, accuracy_axis = figure.subplots(2, 1, sharex=True)

    loss_axis.plot(epochs, [row["loss"] for row in training], color="#255A94")
    loss_axis.set(ylabel="Preference loss", title="DPO on a fixed preference dataset")
    accuracy_axis.plot(epochs, [row["preference_accuracy"] for row in training], color="#176C63")
    accuracy_axis.set(xlabel="Epoch", ylabel="Preference accuracy", ylim=(0, 1.02))
    accuracy_axis.yaxis.set_major_formatter(PercentFormatter(1))

    for axis in (loss_axis, accuracy_axis):
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=160)
