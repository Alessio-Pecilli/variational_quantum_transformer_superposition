#!/usr/bin/env python3
"""Rebuild Friday monomial complexity plots (ō vs T linear; ō vs d log + advantage)."""
from __future__ import annotations

from pathlib import Path

from run_study import _plot_vs_T, _plot_vs_d, _plot_vs_T_by_d, _plot_vs_d_by_T

# From Nuovo Documento di testo.txt / job finished 2026-07-16 ~18:05
ROWS = [
    # vs T, d=16
    {"label": "T2_d16_k2", "T": 2, "d": 16, "k": 2, "obar": 0.299136, "reference_haar": 0.015625, "reference_advantage": 0.125, "final_loss": 2.4237},
    {"label": "T4_d16_k2", "T": 4, "d": 16, "k": 2, "obar": 0.200828, "reference_haar": 0.015625, "reference_advantage": 0.125, "final_loss": 3.2256},
    {"label": "T8_d16_k2", "T": 8, "d": 16, "k": 2, "obar": 0.170267, "reference_haar": 0.015625, "reference_advantage": 0.125, "final_loss": 3.5518},
    {"label": "T16_d16_k2", "T": 16, "d": 16, "k": 2, "obar": 0.094199, "reference_haar": 0.015625, "reference_advantage": 0.125, "final_loss": 4.7381},
    {"label": "T32_d16_k2", "T": 32, "d": 16, "k": 2, "obar": 0.092227, "reference_haar": 0.015625, "reference_advantage": 0.125, "final_loss": 4.7784},
    {"label": "T2_d16_k3", "T": 2, "d": 16, "k": 3, "obar": 0.281290, "reference_haar": 0.00390625, "reference_advantage": 0.06629126, "final_loss": 2.5492},
    {"label": "T4_d16_k3", "T": 4, "d": 16, "k": 3, "obar": 0.163230, "reference_haar": 0.00390625, "reference_advantage": 0.06629126, "final_loss": 3.6451},
    {"label": "T8_d16_k3", "T": 8, "d": 16, "k": 3, "obar": 0.126816, "reference_haar": 0.00390625, "reference_advantage": 0.06629126, "final_loss": 4.1470},
    {"label": "T16_d16_k3", "T": 16, "d": 16, "k": 3, "obar": 0.069428, "reference_haar": 0.00390625, "reference_advantage": 0.06629126, "final_loss": 5.3550},
    {"label": "T32_d16_k3", "T": 32, "d": 16, "k": 3, "obar": 0.068032, "reference_haar": 0.00390625, "reference_advantage": 0.06629126, "final_loss": 5.3871},
    # vs d, T=32
    {"label": "T32_d2_k2", "T": 32, "d": 2, "k": 2, "obar": 0.398930, "reference_haar": 0.3535534, "reference_advantage": 1.0, "final_loss": 2.9346},
    {"label": "T32_d4_k2", "T": 32, "d": 4, "k": 2, "obar": 0.293566, "reference_haar": 0.125, "reference_advantage": 0.5, "final_loss": 2.4595},
    {"label": "T32_d8_k2", "T": 32, "d": 8, "k": 2, "obar": 0.266611, "reference_haar": 0.04419417, "reference_advantage": 0.25, "final_loss": 2.6477},
    {"label": "T32_d16_k2", "T": 32, "d": 16, "k": 2, "obar": 0.092227, "reference_haar": 0.015625, "reference_advantage": 0.125, "final_loss": 4.7784},
    {"label": "T32_d32_k2", "T": 32, "d": 32, "k": 2, "obar": 0.019911, "reference_haar": 0.005524272, "reference_advantage": 0.0625, "final_loss": 7.8448},
    {"label": "T32_d2_k3", "T": 32, "d": 2, "k": 3, "obar": 0.338295, "reference_haar": 0.25, "reference_advantage": 1.5, "final_loss": 2.1733},
    {"label": "T32_d4_k3", "T": 32, "d": 4, "k": 3, "obar": 0.238125, "reference_haar": 0.0625, "reference_advantage": 0.5303301, "final_loss": 2.8780},
    {"label": "T32_d8_k3", "T": 32, "d": 8, "k": 3, "obar": 0.205606, "reference_haar": 0.015625, "reference_advantage": 0.1875, "final_loss": 3.1685},
    {"label": "T32_d16_k3", "T": 32, "d": 16, "k": 3, "obar": 0.068032, "reference_haar": 0.00390625, "reference_advantage": 0.06629126, "final_loss": 5.3871},
    {"label": "T32_d32_k3", "T": 32, "d": 32, "k": 3, "obar": 0.012418, "reference_haar": 0.0009765625, "reference_advantage": 0.0234375, "final_loss": 8.7900},
]


def main() -> None:
    out = Path("results_from_hpc/report_20260716_monomial")
    out.mkdir(parents=True, exist_ok=True)

    t_k2 = [r for r in ROWS if r["d"] == 16 and r["k"] == 2]
    t_k3 = [r for r in ROWS if r["d"] == 16 and r["k"] == 3]
    d_k2 = [r for r in ROWS if r["T"] == 32 and r["k"] == 2]
    d_k3 = [r for r in ROWS if r["T"] == 32 and r["k"] == 3]

    _plot_vs_T(
        [("k=2 (trained)", t_k2), ("k=3 (trained)", t_k3)],
        d_fixed=16,
        out_path=out / "mean_O_vs_T.png",
    )
    _plot_vs_d(
        [("k=2 (trained)", d_k2), ("k=3 (trained)", d_k3)],
        T_fixed=32,
        out_path=out / "mean_O_vs_d.png",
    )
    _plot_vs_T_by_d(t_k2, k_fixed=2, out_path=out / "mean_O_vs_T_by_d.png")
    _plot_vs_d_by_T(d_k2, k_fixed=2, out_path=out / "mean_O_vs_d_by_T.png")
    print(f"wrote plots under {out}")


if __name__ == "__main__":
    main()
