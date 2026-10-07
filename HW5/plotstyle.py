"""One matplotlib style for every figure: validated categorical order, hairline grid, thin marks, no dual axes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL = "#b9b8b2"
BLUE, ORANGE, RED = SERIES[0], SERIES[1], SERIES[7]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 1.6, "legend.frameon": False,
    "legend.fontsize": 8, "legend.labelcolor": INK2, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
