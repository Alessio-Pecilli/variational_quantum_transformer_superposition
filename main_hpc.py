#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import faulthandler
import argparse
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

import config
import jax_training_pipeline as jtp
from config import DATASET_CONFIG, OPTIMIZATION_CONFIG
from jax_training_pipeline import run_training
from pennylane_jax_vqt import create_layout


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Main VQT HPC/local entrypoint with optional run overrides."
    )
    parser.add_argument(
        "--circuit-mode",
        choices=("section2", "legacy"),
        default=None,
        help="section2 = QSA Section 2 (default) | legacy = circuito notebook originale",
    )
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--sentence-length", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--max-run-minutes", type=float, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--non-linear-order", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--log-frequency", type=int, default=None)
    parser.add_argument("--eval-frequency", type=int, default=None)
    parser.add_argument("--batch-log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to best_params_native.npy or to a checkpoint/matrices directory.",
    )

    parser.add_argument("--prune-inactive-branches", action="store_true")
    parser.add_argument(
        "--control-readout-mode",
        choices=("auto", "hadamard", "uniform_active"),
        default=None,
    )
    parser.add_argument("--analytic-readout", dest="analytic_readout", action="store_true")
    parser.add_argument("--no-analytic-readout", dest="analytic_readout", action="store_false")
    parser.set_defaults(analytic_readout=None)

    parser.add_argument("--quantum-device", type=str, default=None)
    parser.add_argument("--quantum-diff-method", type=str, default=None)
    parser.add_argument("--analytic-quantum-device", type=str, default=None)
    parser.add_argument("--analytic-quantum-diff-method", type=str, default=None)

    parser.add_argument("--train-embedding", dest="train_embedding", action="store_true")
    parser.add_argument("--freeze-embedding", dest="train_embedding", action="store_false")
    parser.set_defaults(train_embedding=None)
    parser.add_argument("--train-rotation", dest="train_rotation", action="store_true")
    parser.add_argument("--freeze-rotation", dest="train_rotation", action="store_false")
    parser.set_defaults(train_rotation=None)

    parser.add_argument("--dry-layout", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: dict, args):
    scalar_overrides = {
        "circuit_mode": args.circuit_mode,
        "run_label": args.run_label,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "num_layers": args.num_layers,
        "non_linear_order": args.non_linear_order,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_clip_norm": args.gradient_clip_norm,
        "log_frequency": args.log_frequency,
        "eval_frequency": args.eval_frequency,
        "batch_log_interval": args.batch_log_interval,
        "seed": args.seed,
        "resume_checkpoint": args.resume_checkpoint,
        "control_readout_mode": args.control_readout_mode,
        "analytic_readout": args.analytic_readout,
        "quantum_device": args.quantum_device,
        "quantum_diff_method": args.quantum_diff_method,
        "analytic_quantum_device": args.analytic_quantum_device,
        "analytic_quantum_diff_method": args.analytic_quantum_diff_method,
        "train_embedding": args.train_embedding,
        "train_rotation": args.train_rotation,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            cfg[key] = value

    if args.prune_inactive_branches:
        cfg["prune_inactive_branches"] = True

    if args.sentence_length is not None:
        DATASET_CONFIG["sentence_length"] = args.sentence_length
        jtp.DATASET_CONFIG["sentence_length"] = args.sentence_length
    if args.max_sentences is not None:
        DATASET_CONFIG["max_sentences"] = args.max_sentences
        jtp.DATASET_CONFIG["max_sentences"] = args.max_sentences
    if args.max_run_minutes is not None:
        DATASET_CONFIG["max_run_minutes"] = args.max_run_minutes
        jtp.DATASET_CONFIG["max_run_minutes"] = args.max_run_minutes

    return cfg


def print_layout_preview(cfg: dict):
    sentence_length = int(DATASET_CONFIG.get("sentence_length", 5))
    embedding_dim = int(cfg.get("embedding_dim", 4))
    circuit_mode = str(cfg.get("circuit_mode", "section2")).lower()

    if circuit_mode == "section2":
        from qsa_section2_circuit import qubit_budget

        k = int(cfg.get("non_linear_order", 2))
        n_qubits = qubit_budget(sentence_length, embedding_dim, k)
        print(
            f"mode=section2 label={cfg.get('run_label')} T={sentence_length} d={embedding_dim} "
            f"k={k} n_qubits={n_qubits} "
            f"train_embedding={cfg.get('train_embedding')} "
            f"max_sentences={DATASET_CONFIG.get('max_sentences')} "
            f"epochs={cfg.get('epochs')}"
        )
        return

    layout = create_layout(
        sequence_length=sentence_length,
        feature_dimension=embedding_dim,
        num_layers=int(cfg.get("num_layers", 2)),
        non_linear_order=int(cfg.get("non_linear_order", 2)),
        prune_inactive_branches=bool(cfg.get("prune_inactive_branches", False)),
        active_branch_count=max(sentence_length - 1, 0),
    )
    total_qubits = layout.control_qubits + (layout.non_linear_order + 1) * layout.feature_qubits
    print(
        f"label={cfg.get('run_label')} T={sentence_length} d={embedding_dim} "
        f"nlo={layout.non_linear_order} prune={layout.prune_inactive_branches} "
        f"active_branches={layout.active_branch_count} qc={layout.control_qubits} "
        f"qf={layout.feature_qubits} total_qubits={total_qubits} "
        f"padded_T={layout.padded_sequence_length} "
        f"quantum_params={layout.total_parameters} "
        f"readout={cfg.get('control_readout_mode')} "
        f"analytic={cfg.get('analytic_readout')} "
        f"train_embedding={cfg.get('train_embedding')} "
        f"train_rotation={cfg.get('train_rotation')} "
        f"max_sentences={DATASET_CONFIG.get('max_sentences')} "
        f"epochs={cfg.get('epochs')}"
    )


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger = setup_logging(rank)
    cfg = dict(OPTIMIZATION_CONFIG)
    cfg.setdefault("seed", 42)
    cfg = apply_overrides(cfg, args)

    if rank == 0:
        print_layout_preview(cfg)
    if args.dry_layout:
        return 0

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
