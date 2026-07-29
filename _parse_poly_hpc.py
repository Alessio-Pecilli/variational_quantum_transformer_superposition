#!/usr/bin/env python3
"""Parse poly HPC log robustly (MPI-interleaved stdout)."""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path(r"C:\Users\Ale\Desktop\Nuovo Documento di testo.txt")
OUT = Path(__file__).resolve().parent / "_poly_hpc_parsed.json"
text = SRC.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

# --- Job meta ---
started = next(l for l in lines if "STARTED" in l and "JOB" in l)
finished = next(l for l in lines if "JOB FINISHED" in l)
kernel = next(l for l in lines if l.startswith("kernel_mode="))
conv = next(l for l in lines if "Convergenza:" in l)
srun_c = next(l for l in lines if "run_study.py" in l and "srun" in l)

# --- Complexity from WARN + RIASSUNTO tables (rank-0 authoritative) ---
# WARN lines: "  T8_d2_k2: loss=3.2003 > 3.0  (obar=0.4015)"
warn = {}
for l in lines:
    m = re.match(
        r"\s+(T\d+_d\d+_k\d+): loss=([0-9.eE+-]+) > [0-9.]+  \(obar=([0-9.eE+-]+)\)",
        l,
    )
    if m:
        warn[m.group(1)] = {"loss": float(m.group(2)), "obar": float(m.group(3))}

# Parse RIASSUNTO tables: vs T and vs d
# Find "SWEEP vs T" then table rows "   T          obar ..."
def parse_table_after(marker: str, col0_name: str):
    rows = []
    try:
        i0 = next(i for i, l in enumerate(lines) if marker in l and i < 700)
    except StopIteration:
        return rows
    # find header
    i = i0
    while i < min(i0 + 80, len(lines)):
        if re.search(rf"^\s*{col0_name}\s+obar", lines[i]):
            i += 1
            break
        i += 1
    while i < len(lines):
        l = lines[i]
        if not l.strip() or l.startswith("  Atteso") or l.startswith("FILE") or l.startswith("SWEEP") or l.startswith("---") or l.startswith("RIASSUNTO") or l.startswith("==="):
            if rows and (l.startswith("SWEEP") or l.startswith("FILE") or l.startswith("===") or l.startswith("RIASSUNTO")):
                break
            if not l.strip() or l.startswith("  Atteso") or l.startswith("  T>="):
                i += 1
                continue
            if l.startswith("---"):
                # interleaved training noise inside riassunto — skip until next numeric row or end
                i += 1
                continue
        m = re.match(
            r"^\s*(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)(?:\s+(\d+))?(?:\s+(si|no)\s+(si|no))?",
            l,
        )
        if m:
            rows.append(
                {
                    col0_name: int(m.group(1)),
                    "obar": float(m.group(2)),
                    "haar": float(m.group(3)),
                    "adv": float(m.group(4)),
                    "n_qubits": int(m.group(5)) if m.group(5) and m.group(5).isdigit() else None,
                    "above_haar": (m.group(6) == "si") if m.group(6) else float(m.group(2)) > float(m.group(3)),
                    "above_adv": (m.group(7) == "si") if m.group(7) else float(m.group(2)) > float(m.group(4)),
                }
            )
        elif rows and re.match(r"^\s+[A-Za-z]", l):
            break
        i += 1
    return rows

vs_T = parse_table_after("SWEEP vs T", "T")
vs_d = parse_table_after("SWEEP vs d", "d")

# Rebuild full complexity map from individual result lines that include label
# Strategy: for each "--- LABEL |" find the NEXT obar line that appears BEFORE another
# "--- T" label AND after embedding checks for THIS block. Because of MPI, also
# require that n_qubits in the header matches haar formula consistency? Better:
# use metrics from WARN + reconstruct from label components + haar formula.

from math import factorial
import math

def haar_floor(d, k):
    return float(d ** (-(k + 1) / 2))

def adv_thr(d, k):
    return float(math.sqrt(k * factorial(k) / (d ** k)))

# Collect all labels announced in phase 1
phase1_end = next(i for i, l in enumerate(lines) if "PHASE 1 DONE" in l)
labels = []
for l in lines[:phase1_end]:
    m = re.match(r"--- (T\d+_d\d+_k\d+) \| T=(\d+) d=(\d+) k=(\d+) n_qubits=(\d+)", l)
    if m:
        labels.append(
            {
                "label": m.group(1),
                "T": int(m.group(2)),
                "d": int(m.group(3)),
                "k": int(m.group(4)),
                "n_qubits": int(m.group(5)),
            }
        )
# unique by label (keep first)
seen = {}
for lab in labels:
    seen.setdefault(lab["label"], lab)
labels = list(seen.values())

# Prefer WARN for loss/obar when present; else try RIASSUNTO vs T/d for the main sweeps
complexity = []
for lab in labels:
    label, T, d, k = lab["label"], lab["T"], lab["d"], lab["k"]
    haar = haar_floor(d, k)
    adv = adv_thr(d, k)
    if label in warn:
        obar = warn[label]["obar"]
        loss = warn[label]["loss"]
        conv = False
        source = "warn"
    else:
        # try vs_T table (d=16 implied in broken header but table is d=16 from values)
        obar = loss = None
        source = None
        # match vs T if d==16 and k==2 (table is k=2 only in first table)
        if d == 16 and k == 2:
            for r in vs_T:
                if r["T"] == T:
                    obar, loss = r["obar"], None
                    source = "riassunto_vs_T"
                    break
        if obar is None and T == 32 and k == 2:
            for r in vs_d:
                if r["d"] == d:
                    obar = r["obar"]
                    source = "riassunto_vs_d"
                    break
        # For converged points (loss<=3) not in WARN — scan for unique matching
        # by searching result lines and validating haar matches expected for (d,k)
        if obar is None:
            for l in lines[:phase1_end]:
                m2 = re.search(
                    r"obar=([0-9.eE+-]+)\s+mean_O=([0-9.eE+-]+)\s+haar=([0-9.eE+-]+)\s+"
                    r"adv=([0-9.eE+-]+)\s+loss=([0-9.eE+-]+)\s+conv=(\w+)\s+\(([0-9.]+)s\)",
                    l,
                )
                if not m2:
                    continue
                if abs(float(m2.group(3)) - haar) < 1e-12 and abs(float(m2.group(4)) - adv) < 1e-5:
                    # ambiguous: many points share same (d,k). Keep candidates.
                    pass

        # Second pass: associate by scanning with stack of open jobs keyed by (haar,adv) FIFO? 
        # Simpler: extract ALL result tuples, then for each label find result where
        # haar/adv match AND we haven't assigned that result yet — still ambiguous for same (d,k).
        conv = None
        loss = loss

    # fill from a careful non-interleaved approach: match label line to result
    # only if no other "--- T" between label and result
    if label not in warn:
        for i, l in enumerate(lines[:phase1_end]):
            if not l.startswith(f"--- {label} |"):
                continue
            # look ahead until next --- T label
            for j in range(i + 1, min(i + 40, phase1_end)):
                if re.match(r"--- T\d+_d\d+_k\d+ \|", lines[j]):
                    break
                m2 = re.search(
                    r"obar=([0-9.eE+-]+)\s+mean_O=([0-9.eE+-]+)\s+haar=([0-9.eE+-]+)\s+"
                    r"adv=([0-9.eE+-]+)\s+loss=([0-9.eE+-]+)\s+conv=(\w+)\s+\(([0-9.]+)s\)",
                    lines[j],
                )
                if m2:
                    obar = float(m2.group(1))
                    loss = float(m2.group(5))
                    conv = m2.group(6) == "True"
                    source = "contiguous_block"
                    break
            break

    if obar is None:
        continue

    complexity.append(
        {
            "label": label,
            "T": T,
            "d": d,
            "k": k,
            "n_qubits": lab["n_qubits"],
            "obar": obar,
            "haar": haar,
            "adv": adv,
            "loss": loss,
            "conv": conv if conv is not None else (loss is not None and loss <= 3),
            "above_haar": obar > haar,
            "above_adv": obar > adv,
            "ratio_haar": obar / haar if haar > 0 else None,
            "source": source if label not in warn else "warn",
        }
    )

# For points still missing loss, leave None
# Also add warn-only completeness: ensure all warn labels present
for label, w in warn.items():
    if any(c["label"] == label for c in complexity):
        continue
    tm = re.match(r"T(\d+)_d(\d+)_k(\d+)", label)
    T, d, k = map(int, tm.groups())
    haar, adv = haar_floor(d, k), adv_thr(d, k)
    complexity.append(
        {
            "label": label,
            "T": T,
            "d": d,
            "k": k,
            "n_qubits": None,
            "obar": w["obar"],
            "haar": haar,
            "adv": adv,
            "loss": w["loss"],
            "conv": False,
            "above_haar": w["obar"] > haar,
            "above_adv": w["obar"] > adv,
            "ratio_haar": w["obar"] / haar,
            "source": "warn_only",
        }
    )

# Deduplicate by label preferring contiguous_block / warn
by = {}
prio = {"contiguous_block": 3, "warn": 3, "warn_only": 2, "riassunto_vs_T": 1, "riassunto_vs_d": 1, None: 0}
for c in complexity:
    prev = by.get(c["label"])
    if prev is None or prio.get(c.get("source"), 0) >= prio.get(prev.get("source"), 0):
        by[c["label"]] = c
complexity = sorted(by.values(), key=lambda r: (r["k"], r["d"], r["T"]))

# --- Baseline SUMMARY JSON blocks ---
def extract_summaries():
    out = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "SUMMARY":
            # find opening {
            j = i + 1
            while j < len(lines) and lines[j].strip() != "{":
                j += 1
            if j >= len(lines):
                break
            depth = 0
            buf = []
            for k in range(j, len(lines)):
                # strip UCX warn lines
                if "UCX" in lines[k] or "parser.c" in lines[k]:
                    continue
                buf.append(lines[k])
                depth += lines[k].count("{") - lines[k].count("}")
                if depth == 0 and buf:
                    raw = "\n".join(buf)
                    try:
                        data = json.loads(raw)
                        out.append(data)
                    except json.JSONDecodeError as e:
                        out.append({"_parse_error": str(e), "_raw_head": raw[:200]})
                    i = k + 1
                    break
            else:
                break
            continue
        i += 1
    return out

summaries = extract_summaries()
baselines = []
for s in summaries:
    if "_parse_error" in s:
        continue
    cfg = s.get("config", {})
    for m in s.get("models", []):
        baselines.append(
            {
                "T": cfg.get("T"),
                "d": cfg.get("d"),
                "k": cfg.get("k"),
                "epochs": cfg.get("epochs"),
                "n_seeds": m.get("n_seeds"),
                "model": m.get("model"),
                "final_loss_mean": m.get("final_loss_mean"),
                "final_loss_std": m.get("final_loss_std"),
                "qsa_csa_gap": s.get("qsa_csa_max_epoch_gap_first_seed"),
                "output_dir": cfg.get("output_dir"),
            }
        )

# Split phase2 (single k=2 definitive) vs phase3 (ppl_vs_k)
phase2 = [b for b in baselines if b.get("output_dir") and "ppl_vs_k" not in str(b["output_dir"])]
phase3 = [b for b in baselines if b.get("output_dir") and "ppl_vs_k" in str(b["output_dir"])]

payload = {
    "job": {
        "id": "49628674",
        "started": started,
        "finished": finished,
        "duration_note": "17:05:10 -> 17:54:23 (~49 min)",
        "kernel_mode": "poly",
        "convergence": conv,
        "srun_complexity": srun_c.strip()[:300],
        "n_complexity_labels": len(labels),
        "n_complexity_parsed": len(complexity),
        "n_above_haar": sum(1 for r in complexity if r["above_haar"]),
        "n_below_haar": sum(1 for r in complexity if not r["above_haar"]),
        "n_above_adv": sum(1 for r in complexity if r["above_adv"]),
        "n_loss_le_3": sum(1 for r in complexity if r["loss"] is not None and r["loss"] <= 3),
        "n_loss_gt_3": sum(1 for r in complexity if r["loss"] is not None and r["loss"] > 3),
        "n_conv_true": sum(1 for r in complexity if r.get("conv") is True),
        "n_summaries": len(summaries),
    },
    "complexity": complexity,
    "riassunto_vs_T": vs_T,
    "riassunto_vs_d": vs_d,
    "baselines_k2": phase2,
    "ppl_vs_k": phase3,
    "warn": warn,
}

OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("JOB", payload["job"]["id"], payload["job"]["duration_note"])
print("kernel poly |", conv)
print(f"complexity parsed {len(complexity)}/{len(labels)}")
print(
    f"aboveH={payload['job']['n_above_haar']} belowH={payload['job']['n_below_haar']} "
    f"aboveA={payload['job']['n_above_adv']} loss<=3={payload['job']['n_loss_le_3']} "
    f"loss>3={payload['job']['n_loss_gt_3']} conv={payload['job']['n_conv_true']}"
)

print("\n=== ALL complexity ===")
for r in complexity:
    loss_s = f"{r['loss']:.4f}" if r["loss"] is not None else "NA"
    print(
        f"{r['label']:14s} obar={r['obar']:.6f} haar={r['haar']:.4e} "
        f"ratio={r['ratio_haar']:.2f} loss={loss_s:>8s} "
        f"H={'Y' if r['above_haar'] else 'N'} A={'Y' if r['above_adv'] else 'N'} "
        f"conv={r['conv']} src={r['source']}"
    )

print("\n=== BELOW Haar ===")
for r in complexity:
    if not r["above_haar"]:
        print(f"  {r['label']} obar={r['obar']:.6f} < haar={r['haar']:.6e} loss={r['loss']}")

print("\n=== Baselines k=2 (phase2) ===")
for b in phase2:
    print(f"  {b['model']:7s} {b['final_loss_mean']:.4f} +/- {b['final_loss_std']:.4f} (n={b['n_seeds']})")

print("\n=== PPL vs k (phase3) ===")
byk = {}
for b in phase3:
    byk.setdefault(b["k"], []).append(b)
for k in sorted(byk):
    print(f"k={k}")
    for b in byk[k]:
        print(f"  {b['model']:7s} {b['final_loss_mean']:.4f} +/- {b['final_loss_std']:.4f}")

print("\nRIASSUNTO vs T rows:", vs_T)
print("RIASSUNTO vs d rows:", vs_d)
print("wrote", OUT)
