"""
figures/make_msbe_figure.py

Paper-style figure of the Bellman-error (MSBE) loss across the 5-stage curriculum
of the reference run. Reads figures/msbe_loss.csv (one evaluation point every 500
episodes) and writes figures/msbe_loss.png and figures/msbe_loss.pdf.

Run (matplotlib is not a project dependency, so use an ephemeral environment):
    uv run --with matplotlib python figures/make_msbe_figure.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent

# (stage label, first episode, last episode, defence probability beta)
STAGES = [
    ("1", 0, 8000, "0.00"),
    ("2A", 8000, 18000, "0.35"),
    ("2B", 18000, 30000, "0.70"),
    ("2C", 30000, 42000, "0.85"),
    ("2D", 42000, 60000, "1.00"),
]
MEAN_LOSS = 0.9  # x 1e-3, mean over the whole curriculum as reported by the training logs

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "savefig.dpi": 200,
})

with open(HERE / "msbe_loss.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
episodes = [int(r["episode"]) for r in rows]
loss = [float(r["msbe_loss"]) * 1e3 for r in rows]  # plot in units of 1e-3

fig, ax = plt.subplots(figsize=(6.2, 3.6))

# stage bands and boundaries
for i, (name, a, b, beta) in enumerate(STAGES):
    if i % 2 == 0:
        ax.axvspan(a, b, color="0.94", lw=0, zorder=0)
    if i > 0:
        ax.axvline(a, color="0.55", lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.text((a + b) / 2, 1.015, f"Stage {name}\n" + rf"$\beta={beta}$", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9.5, linespacing=1.25)

ax.plot(episodes, loss, color="#1f3f8f", lw=1.0, marker="o", ms=2.2, zorder=3,
        label="MSBE (every 500 episodes)")
ax.axhline(MEAN_LOSS, color="#b03a2e", lw=0.9, ls="--", zorder=2,
           label=r"mean $=0.9\times10^{-3}$")

peak = max(range(len(loss)), key=loss.__getitem__)
ax.annotate(rf"peak ${loss[peak]:.1f}\times10^{{-3}}$", xy=(episodes[peak], loss[peak]),
            xytext=(episodes[peak] - 2500, loss[peak] + 0.45), ha="right", fontsize=10,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.25"))

ax.set_xlim(0, 60000)
ax.set_ylim(0, 3.0)
ax.xaxis.set_major_locator(MultipleLocator(10000))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{int(v / 1000)}k"))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
ax.set_xlabel("Training episode (continuous across the five stages)")
ax.set_ylabel(r"MSBE loss ($\times10^{-3}$)")
ax.legend(loc="upper right", fontsize=9.5, handlelength=2.0)

fig.tight_layout()
fig.savefig(HERE / "msbe_loss.png", facecolor="white")
fig.savefig(HERE / "msbe_loss.pdf", facecolor="white")
print("saved", HERE / "msbe_loss.png", "and", HERE / "msbe_loss.pdf")
