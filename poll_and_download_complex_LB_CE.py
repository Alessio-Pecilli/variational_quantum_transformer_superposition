#!/usr/bin/env python3
"""Poll Leonardo job for complex LB+CE_unif campaign and download pack."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HOST = "apecilli@login.leonardo.cineca.it"
REMOTE = "/leonardo_work/IscrC_QuSALa/variational_quantum_transformer_sovrapposition"
OUT = "results/classical_sequences/complex_LB_CE_T32_d16_ks1-2-3-5_L16_n10"
JOBS = ("51675806",)
LOCAL_PACK = (
    Path(__file__).resolve().parent
    / "results_from_hpc"
    / "final_campaign_classical_complex"
    / "run_LB_CE_n10"
)
POLL_S = 300


def ssh(cmd: str) -> str:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, cmd],
        capture_output=True,
        text=True,
    )
    return (r.stdout or "") + (r.stderr or "")


def job_done(state: str) -> bool:
    return state.strip() in {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
    }


def states() -> dict[str, str]:
    out = {}
    for j in JOBS:
        s = ssh(f"sacct -j {j} -n -X -o State | head -1").strip().split()
        out[j] = s[0] if s else "UNKNOWN"
    return out


def main() -> int:
    if "PLACEHOLDER" in JOBS[0]:
        print("Update JOBS in this script with the real Slurm id first.", file=sys.stderr)
        return 2
    LOCAL_PACK.mkdir(parents=True, exist_ok=True)
    (LOCAL_PACK / "plots").mkdir(parents=True, exist_ok=True)
    print(f"polling every {POLL_S}s for jobs {JOBS} ...", flush=True)
    while True:
        st = states()
        n = ssh(f"ls {REMOTE}/{OUT}/plots 2>/dev/null | wc -l").strip()
        print(f"{time.strftime('%H:%M:%S')} states={st} plots={n}", flush=True)
        if all(job_done(st[j]) for j in JOBS):
            print(f"jobs finished: {st}", flush=True)
            break
        time.sleep(POLL_S)

    for p in [
        "train_LB_vs_k.png",
        "test_LB_vs_k.png",
        "train_L1_CE_unif_vs_k.png",
        "test_L1_CE_unif_vs_k.png",
    ]:
        subprocess.run(
            ["scp", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{OUT}/plots/{p}", str(LOCAL_PACK / "plots" / p)],
            check=False,
        )
    subprocess.run(
        ["scp", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{OUT}/summary.json", str(LOCAL_PACK / "summary.json")],
        check=False,
    )
    subprocess.run(
        ["scp", "-o", "BatchMode=yes", "-r", f"{HOST}:{REMOTE}/{OUT}/aggregates", str(LOCAL_PACK)],
        check=False,
    )
    (LOCAL_PACK / "DOWNLOAD_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f"\n{st}\n", encoding="utf-8")
    print("download complete ->", LOCAL_PACK, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
