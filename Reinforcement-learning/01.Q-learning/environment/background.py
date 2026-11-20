from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch, Patch

from . import gridworld


PAPER_COLOR = "#f8f6f0"
INK_COLOR = "#243448"
CELL_STYLES = {
    "S": ("Start", "#ede1f5"),
    "G": ("Food", "#e4ecd8"),
    "#": ("Wall", "#dce2e8"),
    "X": ("Trap", "#f4dcd4"),
    ".": ("Open cell", "#f0ece3"),
}


def draw_background(ax, *, legend_title=None):
    sheet = imread(Path(__file__).with_name("maze-icons.png"))
    # 按左上、右上、左下、右下取出起点、食物、墙和陷阱的图标。
    middle_row, middle_col = sheet.shape[0] // 2, sheet.shape[1] // 2
    icons = {
        "S": sheet[:middle_row, :middle_col],
        "G": sheet[:middle_row, middle_col:],
        "#": sheet[middle_row:, :middle_col],
        "X": sheet[middle_row:, middle_col:],
    }

    ax.set_facecolor(PAPER_COLOR)
    for row, line in enumerate(gridworld.LAYOUT):
        for col, marker in enumerate(line):
            _, color = CELL_STYLES.get(marker, CELL_STYLES["."])
            ax.add_patch(FancyBboxPatch(
                (col - 0.48, row - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=color, edgecolor="#d7d5ce", linewidth=0.6, zorder=0,
            ))
            if marker in icons:
                ax.imshow(
                    icons[marker], origin="upper", interpolation="antialiased", zorder=1,
                    extent=(col - 0.44, col + 0.44, row + 0.44, row - 0.44),
                )

    # 第 0 行在最上方。
    size = gridworld.SIZE
    ax.set(xlim=(-0.5, size - 0.5), ylim=(size - 0.5, -0.5), aspect="equal")
    ax.set(xticks=range(size), yticks=range(size))
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=7, labelsize=10, labelcolor=INK_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        handles=[Patch(label=label, facecolor=color, edgecolor="#d7d5ce")
                 for label, color in CELL_STYLES.values()],
        title=legend_title, title_fontsize=8, labelcolor=INK_COLOR,
        loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=5,
        facecolor="#fcfbf8", edgecolor="#e1dcd2", columnspacing=1.2, fontsize=8.5,
    )


def save_maze_map(output_path):
    figure = Figure(figsize=(6.0, 6.4), layout="constrained", facecolor=PAPER_COLOR)
    figure.get_layout_engine().set(w_pad=0.3, h_pad=0.22)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    draw_background(axes)
    size = gridworld.SIZE
    axes.set_title(f"{size} × {size} maze", loc="left", color=INK_COLOR, fontsize=18, fontweight="bold", pad=14)
    figure.savefig(output_path, dpi=150)


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parents[1] / "docs" / "figures" / "maze-map.png"
    save_maze_map(output_path)
