#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import faulthandler
import logging
import sys
import traceback
from pathlib import Path

import numpy as np

faulthandler.enable(file=sys.stderr, all_threads=True)
np.seterr(all="warn")

try:
    from mpi4py import MPI
except Exception:
    class FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            return None

    class FakeMPI:
        COMM_WORLD = FakeComm()

    MPI = FakeMPI()

from config import OPTIMIZATION_CONFIG
from jax_training_pipeline import run_training


def setup_logging(rank: int):
    logger = logging.getLogger("vqt_jax")
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
        logfile = logging.FileHandler(f"logs/rank_{rank}.log", encoding="utf-8")
        logfile.setFormatter(formatter)
        logfile.addFilter(RankFilter())
        logger.addHandler(logfile)

    return logger


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger = setup_logging(rank)
    cfg = dict(OPTIMIZATION_CONFIG)
    cfg.setdefault("seed", 42)

    try:
        if rank == 0:
            logger.info(f"[MPI] Avvio backend distribuito JAX+PennyLane su {size} rank")
        result = run_training(logger=logger, cfg=cfg, comm=comm, rank=rank, size=size)
    except Exception as exc:
        logger.error("Errore critico durante l'esecuzione JAX")
        logger.error(f"Tipo: {type(exc).__name__}, Msg: {exc}")
        for line in traceback.format_exc().splitlines():
            logger.error(line)
        result = 1
    finally:
        try:
            comm.Barrier()
        except Exception:
            pass

    if result == 0:
        logger.info(f"[Rank {rank}] Uscita pulita (exit=0)")
    else:
        logger.error(f"[Rank {rank}] Uscita con errore (exit={result})")
    return result


if __name__ == "__main__":
    sys.exit(main())
