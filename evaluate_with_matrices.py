#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Valutazione distribuita qiskit-free di un modello salvato (`best_params_native.npy`).

Riusa la stessa pipeline PennyLane+JAX del training e distribuisce i campioni
tra rank MPI come il vecchio script HPC.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from mpi4py import MPI
except Exception:
    class FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, value, root=0):
            return value

        def Barrier(self):
            return None

    class FakeMPI:
        COMM_WORLD = FakeComm()

    MPI = FakeMPI()

from config import OPTIMIZATION_CONFIG, TEST_ONLY_CONFIG
from jax_training_pipeline import run_saved_model_evaluation


def setup_logging(rank: int):
    logger = logging.getLogger("vqt_eval_jax")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [rank=%(rank)d] %(message)s"
        )

        class RankFilter(logging.Filter):
            def filter(self, record):
                record.rank = rank
                return True

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.addFilter(RankFilter())
        logger.addHandler(console)

        Path("logs").mkdir(exist_ok=True)
        logfile = logging.FileHandler(
            f"logs/evaluation_rank_{rank}.log",
            encoding="utf-8",
        )
        logfile.setFormatter(formatter)
        logfile.addFilter(RankFilter())
        logger.addHandler(logfile)

    return logger


def _resolve_matrices_dir() -> Path:
    matrices_dir = Path(TEST_ONLY_CONFIG.get("matrices_dir", Path.cwd()))
    if (matrices_dir / "matrices").exists():
        return matrices_dir / "matrices"
    return matrices_dir


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger = setup_logging(rank)

    matrices_dir = _resolve_matrices_dir()
    cfg = dict(OPTIMIZATION_CONFIG)

    if rank == 0:
        logger.info("=" * 72)
        logger.info("VALUTAZIONE DISTRIBUITA JAX+PENNYLANE")
        logger.info("=" * 72)
        logger.info(f"[MPI] ranks={size}")
        logger.info(f"[LOAD] matrici da {matrices_dir}")

    metrics = run_saved_model_evaluation(
        matrices_dir=matrices_dir,
        cfg=cfg,
        logger=logger,
        comm=comm,
        rank=rank,
        size=size,
    )

    if rank == 0 and metrics is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = Path(f"evaluation_summary_{timestamp}.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("EVALUATION SUMMARY\n")
            handle.write(f"timestamp={timestamp}\n")
            handle.write(f"mpi_ranks={size}\n")
            handle.write(f"matrices_dir={matrices_dir}\n")
            for key, value in metrics.items():
                handle.write(f"{key}={value}\n")
        logger.info(f"[SAVE] Summary salvato in {summary_path}")

    try:
        comm.Barrier()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
