"""
figures/make_reward_moves_figure.py

Paper-style figure of the greedy evaluation of the reference run across the 5-stage
curriculum: (a) mean evaluation reward and (b) mean game length in moves, evaluated on
40 games every 500 episodes. Reads figures/eval_reward_moves.csv and writes
figures/reward_moves.png and figures/reward_moves.pdf.

Run (matplotlib is not a project dependency, so use an ephemeral environment):
    uv run --with matplotlib python figures/make_reward_moves_figure.py
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
REWARD_COLOR = "#1b7f5a"
MOVES_COLOR = "#c0562b"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 11.5,
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

with open(HERE / "eval_reward_moves.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
episodes = [int(r["episode"]) for r in rows]
reward = [float(r["eval_reward"]) for r in rows]
moves = [float(r["mean_moves"]) for r in rows]

fig, (ax_r, ax_m) = plt.subplots(2, 1, figsize=(6.6, 3.9), sharex=True,
                                 gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1]})


def decorate(ax, top=False):
    for i, (name, a, b, beta) in enumerate(STAGES):
        if i % 2 == 0:
            ax.axvspan(a, b, color="0.94", lw=0, zorder=0)
        if i > 0:
            ax.axvline(a, color="0.55", lw=0.6, ls=(0, (3, 3)), zorder=1)
        if top:
            ax.text((a + b) / 2, 1.03, f"Stage {name}\n" + rf"$\beta={beta}$", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9.5, linespacing=1.25)


decorate(ax_r, top=True)
decorate(ax_m)

ax_r.plot(episodes, reward, color=REWARD_COLOR, lw=1.0, marker="o", ms=2.2, zorder=3)
ax_r.set_ylim(0, 1.25)
ax_r.yaxis.set_major_locator(MultipleLocator(0.25))
ax_r.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
ax_r.set_ylabel("(a) Eval. reward")

ax_m.plot(episodes, moves, color=MOVES_COLOR, lw=1.0, marker="s", ms=2.2, zorder=3)
ax_m.set_ylim(0, 16)
ax_m.yaxis.set_major_locator(MultipleLocator(4))
ax_m.set_ylabel("(b) Moves per game")

ax_m.set_xlim(0, 60000)
ax_m.xaxis.set_major_locator(MultipleLocator(10000))
ax_m.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{int(v / 1000)}k"))
ax_m.set_xlabel("Training episode (continuous across the five stages)")
ax_r.tick_params(labelbottom=False)

fig.align_ylabels([ax_r, ax_m])
fig.savefig(HERE / "reward_moves.png", facecolor="white", bbox_inches="tight", pad_inches=0.08)
fig.savefig(HERE / "reward_moves.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.08)
print("saved", HERE / "reward_moves.png", "and", HERE / "reward_moves.pdf")
