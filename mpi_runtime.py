"""Thin MPI helpers for Section-2 HPC runners (seed / sweep sharding)."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple


def init_mpi(enabled: bool = True) -> Tuple[Any, int, int]:
    """Return (comm, rank, size). If disabled or mpi4py missing, single-rank stub."""
    if not enabled:
        return None, 0, 1
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, int(comm.Get_rank()), int(comm.Get_size())
    except Exception:
        return None, 0, 1


def shard_items(items: List[Any], rank: int, size: int) -> List[Any]:
    return list(items[rank::size])


def gather_list(comm, local: List[Any], root: int = 0) -> Optional[List[Any]]:
    """Gather python lists to root; non-root returns None."""
    if comm is None:
        return local
    parts = comm.gather(local, root=root)
    if int(comm.Get_rank()) != root:
        return None
    out: List[Any] = []
    for part in parts:
        out.extend(part)
    return out


def barrier(comm) -> None:
    if comm is not None and hasattr(comm, "Barrier"):
        try:
            comm.Barrier()
        except Exception:
            pass
