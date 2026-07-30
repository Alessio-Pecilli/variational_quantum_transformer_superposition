#!/usr/bin/env python3
"""Poll Leonardo until L_B finalize is done, then download pack plots."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HOST = "apecilli@login.leonardo.cineca.it"
REMOTE = "/leonardo_work/IscrC_QuSALa/variational_quantum_transformer_sovrapposition"
OUT = "results/final_loss/LB_T16_d8_ks1-2-3-5-6_L16_n10_test"
JOBS = ("51004497", "51073063")
LOCAL_PACK = Path(__file__).resolve().parent / "results_from_hpc" / "final_campaign_LB"
POLL_S = 900


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
    LOCAL_PACK.mkdir(parents=True, exist_ok=True)
    (LOCAL_PACK / "plots").mkdir(parents=True, exist_ok=True)
    print(f"polling every {POLL_S}s ...", flush=True)
    while True:
        st = states()
        n = ssh(f"find {REMOTE}/{OUT} -name metrics.json 2>/dev/null | wc -l").strip()
        print(f"{time.strftime('%H:%M:%S')} states={st} metrics={n}", flush=True)
        if all(job_done(st[j]) for j in JOBS):
            print("jobs finished; downloading pack", flush=True)
            break
        time.sleep(POLL_S)

    # download key artifacts
    plots = [
        "final_aligned_loss_vs_k.png",
        "final_aligned_loss_vs_k_test.png",
        "final_LB_vs_Lhalf_mono_train.png",
        "final_LB_vs_Lhalf_poly_train.png",
        "final_Lhalf_vs_k_train.png",
        "final_Lhalf_vs_k_test.png",
        "final_L1_vs_k_train.png",
        "final_L1_vs_k_test.png",
        "final_LB_training_curves_k3.png",
    ]
    for p in plots:
        src = f"{HOST}:{REMOTE}/{OUT}/plots/{p}"
        dst = LOCAL_PACK / "plots" / p
        subprocess.run(["scp", "-o", "BatchMode=yes", src, str(dst)], check=False)
    subprocess.run(
        ["scp", "-o", "BatchMode=yes", f"{HOST}:{REMOTE}/{OUT}/summary.json", str(LOCAL_PACK / "summary.json")],
        check=False,
    )
    # marker
    (LOCAL_PACK / "DOWNLOAD_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f"\n{st}\n", encoding="utf-8")
    print("download complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
