"""Black-and-white figure style with Polish number formatting."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

from .config import FIGURES  # noqa: E402

INK, DARK, MID, LIGHT = "#111111", "#4d4d4d", "#8c8c8c", "#bdbdbd"
SHADES = ["#d0d0d0", "#a8a8a8", "#808080", "#585858", "#303030", "#000000"]

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.family": "DejaVu Sans", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6, "grid.color": "#cccccc",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": "#222222", "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.8, "lines.markersize": 5,
})


def _pl(x, pos=None):
    return f"{x:.10g}".replace("-", "−").replace(".", ",")


def save(fig, name: str) -> None:
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(_pl))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
