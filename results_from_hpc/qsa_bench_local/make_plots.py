"""Make plots for the clean-slate qsa_bench local pack."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
plots = root / "plots"
plots.mkdir(parents=True, exist_ok=True)

ks = [1, 2, 3]
models = [
    "kqsa-mono",
    "kqsa-poly",
    "kcsa-mono",
    "kcsa-poly",
    "nlcsa-iso",
    "nlcsa-gen",
]
styles = {
    "kqsa-mono": ("C0", "o", "-"),
    "kqsa-poly": ("C0", "s", "--"),
    "kcsa-mono": ("C1", "o", "-"),
    "kcsa-poly": ("C1", "s", "--"),
    "nlcsa-iso": ("C2", "^", "-"),
    "nlcsa-gen": ("C3", "v", "-"),
}


def load_quantum(k):
    data = json.loads((root / f"T16_d8_k{k}.json").read_text(encoding="utf-8"))
    return {r["name"]: r for r in data["rows"]["quantum"]}


by_k = {k: load_quantum(k) for k in ks}
ch_an = 2.5929  # analytic chance L1, complex d=8


def plot_metric(metric, fname, ylabel, title, include_chance=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in models:
        ys = [by_k[k][m][metric] for k in ks]
        c, mk, ls = styles[m]
        ax.plot(ks, ys, color=c, marker=mk, linestyle=ls, label=m,
                linewidth=1.8, markersize=7)
    if include_chance:
        ax.axhline(ch_an, color="0.4", linestyle=":", linewidth=1.2,
                   label="analytic chance L1")
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    out = plots / fname
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


plot_metric(
    "L1", "train_L1_vs_k.png", r"train $L_1$",
    r"Clean-slate shared pipeline — quantum TFIM ($T{=}16$, $d{=}8$)",
    include_chance=True,
)
plot_metric(
    "L1_test", "test_L1_vs_k.png", r"test $L_1$",
    r"Clean-slate shared pipeline — quantum TFIM test ($T{=}16$, $d{=}8$)",
)
plot_metric(
    "L_B", "train_LB_vs_k.png", r"train $\mathcal{L}_B$",
    r"Clean-slate — train $\mathcal{L}_B$ (kQSA/kCSA objective; nl reported)",
)
plot_metric(
    "gain", "train_gain_vs_k.png", r"gain $=$ chance $-$ $L_1$",
    r"Clean-slate — learning signal (gain > 0 means optimized)",
)

data2 = json.loads((root / "T16_d8_k2.json").read_text(encoding="utf-8"))
cq = data2["rows"]["quantum"]
cc = data2["rows"]["classical"]


def bar_compare(rows, title, fname):
    names = [r["name"] for r in rows]
    L1 = [r["L1"] for r in rows]
    L1t = [r.get("L1_test", np.nan) for r in rows]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w / 2, L1, w, label="train L1", color="C0")
    ax.bar(x + w / 2, L1t, w, label="test L1", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel(r"$L_1$")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = plots / fname
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


bar_compare(
    cq,
    r"Quantum TFIM — $L_1$ train/test at $k{=}2$ (shared pipeline)",
    "quantum_k2_L1_bars.png",
)
bar_compare(
    cc,
    r"Classical Markov — $L_1$ train/test at $k{=}2$ (complex ansatz)",
    "classical_k2_L1_bars.png",
)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
for ax, pair, title in [
    (axes[0], ("kqsa-mono", "kcsa-mono"), "monomial"),
    (axes[1], ("kqsa-poly", "kcsa-poly"), "polynomial"),
]:
    for m in pair:
        ys = [by_k[k][m]["L1"] for k in ks]
        c, mk, ls = styles[m]
        ax.plot(ks, ys, color=c, marker=mk, linestyle=ls, label=m,
                linewidth=1.8, markersize=7)
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
axes[0].set_ylabel(r"train $L_1$")
fig.suptitle(
    r"kQSA vs kCSA on identical pipeline (quantum, $T{=}16$, $d{=}8$)",
    y=1.02,
)
fig.tight_layout()
out = plots / "kqsa_vs_kcsa_L1.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
print("done")
