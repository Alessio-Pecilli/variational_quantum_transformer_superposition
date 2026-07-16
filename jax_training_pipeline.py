from __future__ import annotations

import math
import io
import random
import time
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext, redirect_stdout
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from config import (
    DATASET_CONFIG,
    OPTIMIZATION_CONFIG,
    QUANTUM_STATES_CONFIG,
    TRAINING_SENTENCES,
    TEST_ONLY_CONFIG,
    get_training_sentences,
)
from encoding import Encoding
from pennylane_jax_vqt import (
    QuantumParameterLayout,
    ansatz_matrix,
    apply_adam_step,
    compute_overlap_loss,
    create_layout,
    flatten_quantum_parameters,
    get_overlap_probability_fn,
    get_qnode,
    get_state_qnode,
    initialize_quantum_parameters,
    materialize_text_matrices,
    prepare_quantum_batch,
    prepare_text_batch,
    split_quantum_parameters,
    zeros_like_tree,
)
from quantum_projection import create_projector_from_config, get_projection_shape
from quantum_utils import clear_memory


def _as_numpy(value):
    return np.asarray(jax.device_get(value))


def _as_float(value) -> float:
    return float(np.asarray(jax.device_get(value)))


def _copy_tree(tree):
    return jax.tree_util.tree_map(lambda leaf: jnp.array(leaf, copy=True), tree)


def _comm_size(comm) -> int:
    if comm is None:
        return 1
    try:
        return int(comm.Get_size())
    except Exception:
        return 1


def _broadcast(comm, value, root: int = 0):
    if comm is None or not hasattr(comm, "bcast"):
        return value
    return comm.bcast(value, root=root)


def _barrier(comm):
    if comm is None or not hasattr(comm, "Barrier"):
        return None
    try:
        return comm.Barrier()
    except Exception:
        return None


def _mpi_sum_op():
    try:
        from mpi4py import MPI

        return MPI.SUM
    except Exception:
        return None


def _allreduce_ndarray(comm, value: np.ndarray) -> np.ndarray:
    if _comm_size(comm) <= 1:
        return np.asarray(value)

    send = np.ascontiguousarray(np.asarray(value))
    recv = np.zeros_like(send)
    op = _mpi_sum_op()

    try:
        comm.Allreduce(send, recv, op=op)
    except Exception:
        if op is None:
            reduced = comm.allreduce(send)
        else:
            reduced = comm.allreduce(send, op=op)
        recv = np.asarray(reduced)

    return recv


def _allreduce_scalar(comm, value) -> float:
    reduced = _allreduce_ndarray(comm, np.asarray(value))
    return float(np.asarray(reduced).reshape(()))


def _allreduce_tree(comm, tree):
    if _comm_size(comm) <= 1:
        return tree

    def reduce_leaf(leaf):
        reduced = _allreduce_ndarray(comm, jax.device_get(leaf))
        return jnp.asarray(reduced, dtype=leaf.dtype)

    return jax.tree_util.tree_map(reduce_leaf, tree)


def _shard_items(items, rank: int, size: int):
    return list(items[rank::size])


def _shard_array(array: np.ndarray, rank: int, size: int) -> np.ndarray:
    return np.asarray(array)[rank::size]


def _maybe_silence_stdout(enabled: bool):
    return redirect_stdout(io.StringIO()) if enabled else nullcontext()


def _overlaps_to_losses(overlaps: jnp.ndarray) -> jnp.ndarray:
    overlaps = jnp.clip(jnp.real(overlaps), 1e-12, 1.0)
    return -jnp.log(overlaps)


def _normalize_sentence_length(sentence: str, sentence_length: int) -> Optional[str]:
    words = sentence.split()
    if len(words) < sentence_length:
        return None
    if len(words) > sentence_length:
        words = words[:sentence_length]
    return " ".join(words)


def _load_text_sentences(logger, seed: Optional[int] = None) -> List[str]:
    sentence_length = DATASET_CONFIG.get("sentence_length", 5)
    saved_random_state = None
    if seed is not None:
        saved_random_state = random.getstate()
        random.seed(seed)

    try:
        try:
            raw_sentences = get_training_sentences()
        except Exception as exc:
            logger.warning(f"[DATA] Fallback a TRAINING_SENTENCES: {exc}")
            raw_sentences = TRAINING_SENTENCES
    finally:
        if saved_random_state is not None:
            random.setstate(saved_random_state)

    normalized = []
    for sentence in raw_sentences:
        fixed = _normalize_sentence_length(sentence, sentence_length)
        if fixed is not None:
            normalized.append(fixed)

    if not normalized:
        raise ValueError("Nessuna frase valida disponibile per il training.")

    logger.info(
        f"[DATA] Frasi testuali caricate: {len(normalized)} | lunghezza={sentence_length}"
    )
    return normalized


def _split_text_sentences(sentences: List[str], cfg: dict):
    sentences = list(sentences)
    if not sentences:
        raise ValueError("Impossibile suddividere un dataset testuale vuoto.")

    test_fraction = float(cfg.get("test_fraction", DATASET_CONFIG.get("test_fraction", 0.2)))
    shuffle_before_split = bool(DATASET_CONFIG.get("shuffle_before_split", True))
    seed = int(cfg.get("seed", 42))

    if shuffle_before_split and len(sentences) > 1:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(sentences))
        sentences = [sentences[idx] for idx in order]

    if len(sentences) == 1 or test_fraction <= 0.0:
        return sentences, []

    test_count = int(round(len(sentences) * test_fraction))
    test_count = max(1, min(len(sentences) - 1, test_count))
    train_sentences = sentences[:-test_count]
    test_sentences = sentences[-test_count:]
    return train_sentences, test_sentences


def _split_array_dataset(items: np.ndarray, cfg: dict):
    items = np.asarray(items)
    if items.shape[0] == 0:
        raise ValueError("Impossibile suddividere un dataset vuoto.")

    test_fraction = float(cfg.get("test_fraction", DATASET_CONFIG.get("test_fraction", 0.2)))
    shuffle_before_split = bool(DATASET_CONFIG.get("shuffle_before_split", True))
    seed = int(cfg.get("seed", 42))

    if shuffle_before_split and items.shape[0] > 1:
        rng = np.random.default_rng(seed)
        items = items[rng.permutation(items.shape[0])]

    if items.shape[0] == 1 or test_fraction <= 0.0:
        return items, np.empty((0,) + items.shape[1:], dtype=items.dtype)

    test_count = int(round(items.shape[0] * test_fraction))
    test_count = max(1, min(items.shape[0] - 1, test_count))
    return items[:-test_count], items[-test_count:]


def _load_quantum_sequences(logger, seed: int) -> np.ndarray:
    from quantum_annealing import TFIMHamiltonian

    qs_cfg = QUANTUM_STATES_CONFIG
    num_sequences = int(qs_cfg.get("num_states", 10))
    sequence_length = int(qs_cfg.get("max_time", DATASET_CONFIG.get("sentence_length", 5)))
    use_projection = qs_cfg.get("use_projection", True) and qs_cfg.get("use_Projector", True)
    source_qubits = qs_cfg.get("source_qubits", qs_cfg.get("num_qubits", 2))
    direct_qubits = qs_cfg.get("num_qubits", qs_cfg.get("target_qubits", 2))
    num_qubits = source_qubits if use_projection else direct_qubits

    logger.info(
        f"[DATA] Generazione stati quantistici: sequences={num_sequences} "
        f"sequence_length={sequence_length} qubits={num_qubits}"
    )

    with redirect_stdout(io.StringIO()):
        hamiltonian = TFIMHamiltonian(num_qubits=num_qubits, seed=seed)
        sequences = hamiltonian.generate_sentences(
            num_sentences=num_sequences,
            max_time=sequence_length,
            seed=seed,
        )
    logger.info(f"[DATA] Stati quantistici generati: shape={sequences.shape}")
    return np.asarray(sequences)


def _prepare_text_training_data(
    sentences: List[str],
    cfg: dict,
    encoding: Optional[Encoding] = None,
    suppress_output: bool = False,
):
    if encoding is None:
        with _maybe_silence_stdout(suppress_output):
            encoding = Encoding(sentences, embeddingDim=cfg["embedding_dim"])

    sentence_length = DATASET_CONFIG.get("sentence_length", 5)
    if sentences:
        token_batch = jnp.stack([encoding.encode_tokens(sentence) for sentence in sentences])
        sentence_length = int(token_batch.shape[1])
    else:
        token_batch = jnp.zeros((0, sentence_length), dtype=jnp.int32)

    positional_encoding = encoding._positionalEncoding(sentence_length)
    return encoding, {
        "token_batch": token_batch,
        "positional_encoding": positional_encoding,
        "sentences": sentences,
    }


def _prepare_quantum_training_data(quantum_sequences: np.ndarray):
    return {
        "state_batch": jnp.asarray(quantum_sequences, dtype=jnp.complex128),
    }


def _build_layout(
    cfg: dict,
    use_quantum_states: bool,
    text_data: Optional[dict],
    quantum_data: Optional[dict],
) -> QuantumParameterLayout:
    if use_quantum_states:
        feature_dimension = (
            2 ** QUANTUM_STATES_CONFIG.get("target_qubits", 2)
            if QUANTUM_STATES_CONFIG.get("use_projection", True)
            and QUANTUM_STATES_CONFIG.get("use_Projector", True)
            else int(quantum_data["state_batch"].shape[-1])
        )
        sequence_length = int(quantum_data["state_batch"].shape[1])
        active_branch_count = max(sequence_length - 1, 0)
    else:
        feature_dimension = int(cfg["embedding_dim"])
        sequence_length = int(text_data["token_batch"].shape[1])
        active_branch_count = max(sequence_length - 1, 0)

    return create_layout(
        sequence_length=sequence_length,
        feature_dimension=feature_dimension,
        num_layers=cfg["num_layers"],
        non_linear_order=int(cfg.get("non_linear_order", 2)),
        prune_inactive_branches=bool(cfg.get("prune_inactive_branches", False)),
        active_branch_count=active_branch_count,
    )


def _initialize_trainable_params(
    layout: QuantumParameterLayout,
    cfg: dict,
    seed: int,
    encoding: Optional[Encoding],
    projection_shape: Optional[Tuple[int, int]],
):
    key = jax.random.PRNGKey(seed)
    quantum_params = initialize_quantum_parameters(key, layout)
    if encoding is not None:
        return {
            **quantum_params,
            "embedding": jnp.asarray(encoding.embeddingMatrix, dtype=jnp.float64),
            "rotation": jnp.asarray(encoding.rotationMatrix, dtype=jnp.float64),
        }

    params = dict(quantum_params)
    if projection_shape is not None:
        projector = create_projector_from_config(QUANTUM_STATES_CONFIG)
        initial_projection = projector.get_initial_params().reshape(projection_shape)
        params["projection"] = jnp.asarray(initial_projection, dtype=jnp.float64)
    return params


def _flatten_native_parameters(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    pieces = [
        flatten_quantum_parameters(
            params["weights_v"],
            params["weights_w"],
            params["weights_c"],
        )
    ]
    if "embedding" in params:
        pieces.append(jnp.ravel(params["embedding"]))
        pieces.append(jnp.ravel(params["rotation"]))
    if "projection" in params:
        pieces.append(jnp.ravel(params["projection"]))
    return jnp.concatenate(pieces)


def _unflatten_native_parameters(
    flat_params: np.ndarray,
    layout: QuantumParameterLayout,
    embedding_shape: Optional[Tuple[int, int]] = None,
    rotation_shape: Optional[Tuple[int, int]] = None,
    projection_shape: Optional[Tuple[int, int]] = None,
):
    flat_params = jnp.asarray(flat_params, dtype=jnp.float64)
    quantum_size = layout.total_parameters
    weights_v, weights_w, weights_c = split_quantum_parameters(
        flat_params[:quantum_size],
        layout,
    )
    offset = quantum_size
    params = {
        "weights_v": weights_v,
        "weights_w": weights_w,
        "weights_c": weights_c,
    }

    if embedding_shape is not None and rotation_shape is not None:
        embedding_size = int(np.prod(embedding_shape))
        rotation_size = int(np.prod(rotation_shape))
        params["embedding"] = flat_params[offset:offset + embedding_size].reshape(embedding_shape)
        offset += embedding_size
        params["rotation"] = flat_params[offset:offset + rotation_size].reshape(rotation_shape)
        offset += rotation_size

    if projection_shape is not None:
        projection_size = int(np.prod(projection_shape))
        params["projection"] = flat_params[offset:offset + projection_size].reshape(projection_shape)

    return params


def _resolve_checkpoint_path(path_like) -> Path:
    checkpoint_path = Path(path_like)
    if checkpoint_path.is_dir():
        full = checkpoint_path / "best_checkpoint.npz"
        direct = checkpoint_path / "best_params_native.npy"
        nested = checkpoint_path / "matrices" / "best_params_native.npy"
        if full.exists():
            return full
        if direct.exists():
            return direct
        if nested.exists():
            return nested
    return checkpoint_path


def _unflatten_like(
    flat_params: np.ndarray,
    layout: QuantumParameterLayout,
    template_params: Dict[str, jnp.ndarray],
):
    return _unflatten_native_parameters(
        flat_params=flat_params,
        layout=layout,
        embedding_shape=template_params["embedding"].shape if "embedding" in template_params else None,
        rotation_shape=template_params["rotation"].shape if "rotation" in template_params else None,
        projection_shape=template_params["projection"].shape if "projection" in template_params else None,
    )


def _save_training_checkpoint(
    checkpoint_dir: Path,
    params: Dict[str, jnp.ndarray],
    first_moment: Dict[str, jnp.ndarray],
    second_moment: Dict[str, jnp.ndarray],
    best_epoch: int,
    best_loss: float,
    optimizer_step: int,
    run_label,
):
    flat_params = _as_numpy(_flatten_native_parameters(params))
    flat_first = _as_numpy(_flatten_native_parameters(first_moment))
    flat_second = _as_numpy(_flatten_native_parameters(second_moment))
    np.save(checkpoint_dir / "best_params_native.npy", flat_params)
    np.savez(
        checkpoint_dir / "best_checkpoint.npz",
        params=flat_params,
        first_moment=flat_first,
        second_moment=flat_second,
        best_epoch=np.asarray(best_epoch, dtype=np.int64),
        best_loss=np.asarray(best_loss, dtype=np.float64),
        optimizer_step=np.asarray(optimizer_step, dtype=np.int64),
        run_label=np.asarray(str(run_label or "")),
    )
    with open(checkpoint_dir / "checkpoint_info.txt", "w", encoding="utf-8") as handle:
        handle.write(f"best_epoch={best_epoch}\n")
        handle.write(f"best_loss={best_loss:.12f}\n")
        handle.write(f"optimizer_step={optimizer_step}\n")
        handle.write(f"run_label={run_label}\n")


def _read_checkpoint_info(checkpoint_path: Path) -> dict:
    info_path = checkpoint_path / "checkpoint_info.txt" if checkpoint_path.is_dir() else checkpoint_path.parent / "checkpoint_info.txt"
    info = {}
    if not info_path.exists():
        return info
    with open(info_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
    return info


def _load_resume_state(
    checkpoint,
    layout: QuantumParameterLayout,
    template_params: Dict[str, jnp.ndarray],
    logger,
    rank: int = 0,
):
    if not checkpoint:
        return None, None, None, 0, None

    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    loaded = np.load(checkpoint_path)
    first_moment = None
    second_moment = None
    optimizer_step = 0
    resume_best_loss = None
    if checkpoint_path.suffix == ".npz":
        flat_params = loaded["params"]
        first_moment = loaded["first_moment"] if "first_moment" in loaded else None
        second_moment = loaded["second_moment"] if "second_moment" in loaded else None
        optimizer_step = int(np.asarray(loaded["optimizer_step"]).item()) if "optimizer_step" in loaded else 0
        resume_best_loss = float(np.asarray(loaded["best_loss"]).item()) if "best_loss" in loaded else None
    else:
        flat_params = loaded
        info = _read_checkpoint_info(checkpoint_path)
        if "optimizer_step" in info:
            optimizer_step = int(info["optimizer_step"])
        elif "best_epoch" in info:
            optimizer_step = int(info["best_epoch"])
        if "best_loss" in info:
            resume_best_loss = float(info["best_loss"])

    expected_size = int(_as_numpy(_flatten_native_parameters(template_params)).size)
    actual_size = int(np.asarray(flat_params).size)
    if actual_size != expected_size:
        raise ValueError(
            f"Checkpoint incompatibile: size={actual_size}, atteso={expected_size}. "
            "Usare la stessa configurazione di T, d, layer, pruning e parametri trainabili."
        )

    params = _unflatten_like(flat_params, layout, template_params)
    first_tree = _unflatten_like(first_moment, layout, template_params) if first_moment is not None else None
    second_tree = _unflatten_like(second_moment, layout, template_params) if second_moment is not None else None
    if rank == 0:
        logger.info(f"[RESUME] Parametri inizializzati da checkpoint: {checkpoint_path}")
        if first_tree is not None and second_tree is not None:
            logger.info(f"[RESUME] Stato Adam ripristinato | optimizer_step={optimizer_step}")
        else:
            logger.info("[RESUME] Stato Adam non presente nel checkpoint; riparto dai pesi migliori con momenti azzerati")
    return params, first_tree, second_tree, optimizer_step, resume_best_loss


def _compute_param_counts(params: Dict[str, jnp.ndarray], layout: QuantumParameterLayout):
    n_quantum = layout.total_parameters
    n_embedding = int(np.prod(params["embedding"].shape)) if "embedding" in params else 0
    n_rotation = int(np.prod(params["rotation"].shape)) if "rotation" in params else 0
    n_projection = int(np.prod(params["projection"].shape)) if "projection" in params else 0
    return n_quantum, n_embedding, n_rotation, n_projection


def _sum_tree(accumulator, increment):
    return jax.tree_util.tree_map(lambda acc, inc: acc + inc, accumulator, increment)


def _sanitize_tree(tree):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.nan_to_num(leaf, nan=0.0, posinf=0.0, neginf=0.0),
        tree,
    )


def _clip_grad_tree(tree, max_norm: float):
    sanitized = _sanitize_tree(tree)
    if max_norm <= 0:
        return sanitized, 0.0
    squared_norm = jax.tree_util.tree_reduce(
        lambda acc, leaf: acc + jnp.sum(jnp.abs(leaf) ** 2),
        sanitized,
        initializer=jnp.asarray(0.0, dtype=jnp.float64),
    )
    global_norm = jnp.sqrt(squared_norm)
    clip_coef = jnp.minimum(1.0, max_norm / jnp.maximum(global_norm, 1e-12))
    clipped = jax.tree_util.tree_map(lambda leaf: leaf * clip_coef, sanitized)
    return clipped, float(global_norm)


def _slice_text_batch(text_data: dict, start: int, stop: int) -> dict:
    return {
        "token_batch": text_data["token_batch"][start:stop],
        "positional_encoding": text_data["positional_encoding"],
    }


def _slice_quantum_batch(quantum_data: dict, start: int, stop: int) -> dict:
    return {
        "state_batch": quantum_data["state_batch"][start:stop],
    }


def _batch_ranges(total: int, batch_size: int):
    if total <= 0:
        return
    batch_size = max(1, min(int(batch_size), total))
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def _stacked_qnode_eval(qnode, x_batch, tilde_batch, prep_batch, weights_v, weights_w, weights_c):
    overlaps = [
        qnode(x_item, tilde_item, prep_item, weights_v, weights_w, weights_c)
        for x_item, tilde_item, prep_item in zip(x_batch, tilde_batch, prep_batch)
    ]
    if not overlaps:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.stack(overlaps)


def _stacked_state_eval(state_qnode, prep_batch, weights_v, weights_w):
    states = [state_qnode(prep_item, weights_v, weights_w) for prep_item in prep_batch]
    if not states:
        return jnp.zeros((0,), dtype=jnp.complex128)
    return jnp.stack(states)


def _get_backend_settings(cfg: dict) -> Tuple[str, str, str]:
    return (
        str(cfg.get("quantum_device", "lightning.qubit")),
        str(cfg.get("quantum_interface", "jax")),
        str(cfg.get("quantum_diff_method", "adjoint")),
    )


def _get_readout_settings(cfg: dict) -> Tuple[str, bool]:
    return (
        str(cfg.get("control_readout_mode", "auto")),
        bool(cfg.get("analytic_readout", True)),
    )


def _get_text_trainability(cfg: dict) -> Tuple[bool, bool]:
    return (
        bool(cfg.get("train_embedding", True)),
        bool(cfg.get("train_rotation", True)),
    )


def _get_analytic_backend_settings(cfg: dict, default_interface: str) -> Tuple[str, str, str]:
    return (
        str(cfg.get("analytic_quantum_device", "default.qubit")),
        default_interface,
        str(cfg.get("analytic_quantum_diff_method", "backprop")),
    )


def _build_text_functions(layout: QuantumParameterLayout, cfg: dict):
    device_name, interface, diff_method = _get_backend_settings(cfg)
    readout_mode, analytic_readout = _get_readout_settings(cfg)
    train_embedding, train_rotation = _get_text_trainability(cfg)
    analytic_device_name, analytic_interface, analytic_diff_method = _get_analytic_backend_settings(
        cfg,
        interface,
    )
    qnode = get_qnode(
        layout,
        device_name=device_name,
        interface=interface,
        diff_method=diff_method,
        readout_mode=readout_mode,
    )
    state_qnode = get_state_qnode(
        layout,
        device_name=analytic_device_name,
        interface=analytic_interface,
        diff_method=analytic_diff_method,
        readout_mode=readout_mode,
    )
    overlap_probability = get_overlap_probability_fn(
        layout,
        device_name=analytic_device_name,
        interface=analytic_interface,
        diff_method=analytic_diff_method,
        readout_mode=readout_mode,
    )

    def _loss_sum_impl(params, token_batch, positional_encoding):
        embedding_param = params["embedding"]
        rotation_param = params["rotation"]
        if not train_embedding:
            embedding_param = jax.lax.stop_gradient(embedding_param)
        if not train_rotation:
            rotation_param = jax.lax.stop_gradient(rotation_param)
        x_batch, tilde_batch, prep_batch, _, _, _ = prepare_text_batch(
            token_batch=token_batch,
            positional_encoding=positional_encoding,
            raw_embedding=embedding_param,
            raw_rotation=rotation_param,
            layout=layout,
        )
        if analytic_readout:
            states = _stacked_state_eval(
                state_qnode,
                prep_batch,
                params["weights_v"],
                params["weights_w"],
            )
            overlaps = jax.vmap(overlap_probability, in_axes=(0, 0, 0, None))(
                states,
                x_batch,
                tilde_batch,
                params["weights_c"],
            )
        else:
            overlaps = _stacked_qnode_eval(
                qnode,
                x_batch,
                tilde_batch,
                prep_batch,
                params["weights_v"],
                params["weights_w"],
                params["weights_c"],
            )
        return jnp.sum(_overlaps_to_losses(overlaps))

    def compute_loss_sum(params, token_batch, positional_encoding):
        return _loss_sum_impl(params, token_batch, positional_encoding)

    def infer_overlaps(params, token_batch, positional_encoding):
        x_batch, tilde_batch, prep_batch, _, _, _ = prepare_text_batch(
            token_batch=token_batch,
            positional_encoding=positional_encoding,
            raw_embedding=params["embedding"],
            raw_rotation=params["rotation"],
            layout=layout,
        )
        if analytic_readout:
            states = _stacked_state_eval(
                state_qnode,
                prep_batch,
                params["weights_v"],
                params["weights_w"],
            )
            return jax.vmap(overlap_probability, in_axes=(0, 0, 0, None))(
                states,
                x_batch,
                tilde_batch,
                params["weights_c"],
            )
        return _stacked_qnode_eval(
            qnode,
            x_batch,
            tilde_batch,
            prep_batch,
            params["weights_v"],
            params["weights_w"],
            params["weights_c"],
        )

    def loss_and_grad(params, token_batch, positional_encoding):
        return jax.value_and_grad(_loss_sum_impl)(params, token_batch, positional_encoding)

    @jax.jit
    def optimizer_step(params, grads, first_moment, second_moment, step, learning_rate):
        return apply_adam_step(
            params=params,
            grads=grads,
            first_moment=first_moment,
            second_moment=second_moment,
            step=step,
            learning_rate=learning_rate,
        )

    return compute_loss_sum, infer_overlaps, loss_and_grad, optimizer_step


def _build_quantum_functions(layout: QuantumParameterLayout, cfg: dict):
    device_name, interface, diff_method = _get_backend_settings(cfg)
    readout_mode, analytic_readout = _get_readout_settings(cfg)
    analytic_device_name, analytic_interface, analytic_diff_method = _get_analytic_backend_settings(
        cfg,
        interface,
    )
    qnode = get_qnode(
        layout,
        device_name=device_name,
        interface=interface,
        diff_method=diff_method,
        readout_mode=readout_mode,
    )
    state_qnode = get_state_qnode(
        layout,
        device_name=analytic_device_name,
        interface=analytic_interface,
        diff_method=analytic_diff_method,
        readout_mode=readout_mode,
    )
    overlap_probability = get_overlap_probability_fn(
        layout,
        device_name=analytic_device_name,
        interface=analytic_interface,
        diff_method=analytic_diff_method,
        readout_mode=readout_mode,
    )

    def _loss_sum_impl(params, state_batch):
        x_batch, tilde_batch, prep_batch = prepare_quantum_batch(
            state_batch=state_batch,
            layout=layout,
            raw_projection=params.get("projection"),
        )
        if analytic_readout:
            states = _stacked_state_eval(
                state_qnode,
                prep_batch,
                params["weights_v"],
                params["weights_w"],
            )
            overlaps = jax.vmap(overlap_probability, in_axes=(0, 0, 0, None))(
                states,
                x_batch,
                tilde_batch,
                params["weights_c"],
            )
        else:
            overlaps = _stacked_qnode_eval(
                qnode,
                x_batch,
                tilde_batch,
                prep_batch,
                params["weights_v"],
                params["weights_w"],
                params["weights_c"],
            )
        return jnp.sum(_overlaps_to_losses(overlaps))

    def compute_loss_sum(params, state_batch):
        return _loss_sum_impl(params, state_batch)

    def infer_overlaps(params, state_batch):
        x_batch, tilde_batch, prep_batch = prepare_quantum_batch(
            state_batch=state_batch,
            layout=layout,
            raw_projection=params.get("projection"),
        )
        if analytic_readout:
            states = _stacked_state_eval(
                state_qnode,
                prep_batch,
                params["weights_v"],
                params["weights_w"],
            )
            return jax.vmap(overlap_probability, in_axes=(0, 0, 0, None))(
                states,
                x_batch,
                tilde_batch,
                params["weights_c"],
            )
        return _stacked_qnode_eval(
            qnode,
            x_batch,
            tilde_batch,
            prep_batch,
            params["weights_v"],
            params["weights_w"],
            params["weights_c"],
        )

    def loss_and_grad(params, state_batch):
        return jax.value_and_grad(_loss_sum_impl)(params, state_batch)

    @jax.jit
    def optimizer_step(params, grads, first_moment, second_moment, step, learning_rate):
        return apply_adam_step(
            params=params,
            grads=grads,
            first_moment=first_moment,
            second_moment=second_moment,
            step=step,
            learning_rate=learning_rate,
        )

    return compute_loss_sum, infer_overlaps, loss_and_grad, optimizer_step


def _prepare_output_dirs(timestamp: str, seed: int, run_label: Optional[str] = None) -> Tuple[Path, Path, Path, Path]:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    suffix = f"_seed{seed}"
    if run_label:
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(run_label))
        suffix += f"_{safe_label}"
    run_dir = results_dir / f"run_{timestamp}{suffix}"
    run_dir.mkdir(exist_ok=True)
    matrices_dir = run_dir / "matrices"
    parameters_dir = run_dir / "parameters"
    summaries_dir = run_dir / "summaries"
    plots_dir = run_dir / "plots"
    matrices_dir.mkdir(exist_ok=True)
    parameters_dir.mkdir(exist_ok=True)
    summaries_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    return run_dir, matrices_dir, parameters_dir, summaries_dir, plots_dir


def _save_matrices_and_metadata(
    params: Dict[str, jnp.ndarray],
    layout: QuantumParameterLayout,
    cfg: dict,
    matrices_dir: Path,
    run_id: str,
    use_quantum_states: bool,
    text_train_sentences: Optional[List[str]] = None,
    text_test_sentences: Optional[List[str]] = None,
):
    quantum_flat = flatten_quantum_parameters(
        params["weights_v"],
        params["weights_w"],
        params["weights_c"],
    )
    best_params_native = _as_numpy(_flatten_native_parameters(params))
    U_matrix = _as_numpy(ansatz_matrix(params["weights_v"], layout.feature_qubits, layout.num_layers))
    W_matrix = _as_numpy(ansatz_matrix(params["weights_w"], layout.feature_qubits, layout.num_layers))

    if "embedding" in params:
        E_matrix, V_rotation, F_matrix = materialize_text_matrices(
            params["embedding"],
            params["rotation"],
        )
        E_matrix = _as_numpy(E_matrix)
        V_rotation = _as_numpy(V_rotation)
        F_matrix = _as_numpy(F_matrix)
    else:
        E_matrix = np.array([])
        V_rotation = np.array([])
        F_matrix = np.array([])

    P_matrix = _as_numpy(params["projection"]) if "projection" in params else np.array([])

    np.save(matrices_dir / "best_params_native.npy", best_params_native)
    np.save(matrices_dir / "U_matrix.npy", U_matrix)
    np.save(matrices_dir / "W_matrix.npy", W_matrix)
    np.save(matrices_dir / "E_matrix.npy", E_matrix)
    np.save(matrices_dir / "F_matrix.npy", F_matrix)
    np.save(matrices_dir / "V_rotation.npy", V_rotation)
    np.save(matrices_dir / "P_matrix.npy", P_matrix)

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "seed": cfg.get("seed", 42),
        "run_label": cfg.get("run_label"),
        "use_quantum_states": use_quantum_states,
        "sequence_length": layout.sequence_length,
        "padded_sequence_length": layout.padded_sequence_length,
        "feature_dimension": layout.feature_dimension,
        "feature_qubits": layout.feature_qubits,
        "control_qubits": layout.control_qubits,
        "num_layers": layout.num_layers,
        "non_linear_order": layout.non_linear_order,
        "active_branch_count": layout.active_branch_count,
        "prune_inactive_branches": layout.prune_inactive_branches,
        "control_readout_mode": cfg.get("control_readout_mode", "auto"),
        "analytic_readout": cfg.get("analytic_readout", True),
        "analytic_quantum_device": cfg.get("analytic_quantum_device", "default.qubit"),
        "analytic_quantum_diff_method": cfg.get("analytic_quantum_diff_method", "backprop"),
        "train_embedding": cfg.get("train_embedding", True),
        "train_rotation": cfg.get("train_rotation", True),
        "weights_v_shape": layout.weights_v_shape,
        "weights_w_shape": layout.weights_w_shape,
        "weights_c_shape": layout.weights_c_shape,
        "quantum_parameter_count": quantum_flat.size,
        "embedding_shape": E_matrix.shape if E_matrix.size else None,
        "rotation_shape": V_rotation.shape if V_rotation.size else None,
        "projection_shape": P_matrix.shape if P_matrix.size else None,
        "text_train_sentences": list(text_train_sentences or []),
        "text_test_sentences": list(text_test_sentences or []),
    }
    np.save(matrices_dir / "metadata.npy", metadata)
    return best_params_native, metadata


def _save_summary_and_plot(
    logger,
    output_dir: Path,
    optimization_loss_history: List[float],
    best_loss: float,
    best_epoch: int,
    n_params_total: int,
    n_params_quantum: int,
    n_params_embedding: int,
    n_params_rotation: int,
    n_params_projection: int,
    train_metrics: Optional[dict],
    test_metrics: Optional[dict],
    eval_history: List[dict],
    stop_reason: str,
    elapsed_seconds: float,
):
    summaries_dir = output_dir / "summaries"
    plots_dir = output_dir / "plots"

    np.savetxt(
        summaries_dir / "optimization_loss_history.txt",
        np.asarray(optimization_loss_history, dtype=np.float64),
    )

    eval_history_path = summaries_dir / "evaluation_history.csv"
    with open(eval_history_path, "w", encoding="utf-8") as handle:
        handle.write(
            "epoch,elapsed_minutes,optimization_loss,train_loss,train_ppl,test_loss,test_ppl\n"
        )
        for row in eval_history:
            handle.write(
                f"{row['epoch']},{row['elapsed_minutes']:.4f},{row['optimization_loss']:.8f},"
                f"{row['train_loss']:.8f},{row['train_ppl']:.8f},"
                f"{row['test_loss']:.8f},{row['test_ppl']:.8f}\n"
            )

    with open(summaries_dir / "training_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(f"epochs_completed={len(optimization_loss_history)}\n")
        handle.write(f"best_loss={best_loss:.8f}\n")
        handle.write(f"best_epoch={best_epoch}\n")
        handle.write(f"stop_reason={stop_reason}\n")
        handle.write(f"elapsed_minutes={elapsed_seconds / 60.0:.4f}\n")
        handle.write(f"n_params_total={n_params_total}\n")
        handle.write(f"n_params_quantum={n_params_quantum}\n")
        handle.write(f"n_params_embedding={n_params_embedding}\n")
        handle.write(f"n_params_rotation={n_params_rotation}\n")
        handle.write(f"n_params_projection={n_params_projection}\n")
        if train_metrics is not None:
            handle.write(f"train_num_sequences={train_metrics.get('num_sequences', 0)}\n")
            handle.write(f"train_loss_mean={train_metrics.get('loss_mean', float('nan')):.8f}\n")
            handle.write(f"train_perplexity={train_metrics.get('perplexity', float('nan')):.8f}\n")
        if test_metrics is not None:
            handle.write(f"test_num_sequences={test_metrics.get('num_sequences', 0)}\n")
            handle.write(f"test_loss_mean={test_metrics.get('loss_mean', float('nan')):.8f}\n")
            handle.write(f"test_perplexity={test_metrics.get('perplexity', float('nan')):.8f}\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = np.arange(1, len(optimization_loss_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, optimization_loss_history, color="blue", linewidth=2, label="Optimization Loss")
        if eval_history:
            eval_epochs = np.asarray([row["epoch"] for row in eval_history], dtype=np.int32)
            train_losses = np.asarray([row["train_loss"] for row in eval_history], dtype=np.float64)
            test_losses = np.asarray([row["test_loss"] for row in eval_history], dtype=np.float64)
            plt.plot(eval_epochs, train_losses, color="green", linewidth=2, marker="o", label="Train Loss")
            plt.plot(eval_epochs, test_losses, color="red", linewidth=2, marker="s", label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Test Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "training_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

        if eval_history:
            eval_epochs = np.asarray([row["epoch"] for row in eval_history], dtype=np.int32)
            train_ppl = np.asarray([row["train_ppl"] for row in eval_history], dtype=np.float64)
            test_ppl = np.asarray([row["test_ppl"] for row in eval_history], dtype=np.float64)
            plt.figure(figsize=(10, 6))
            plt.plot(eval_epochs, train_ppl, color="green", linewidth=2, marker="o", label="Train PPL")
            plt.plot(eval_epochs, test_ppl, color="red", linewidth=2, marker="s", label="Test PPL")
            plt.xlabel("Epoch")
            plt.ylabel("Perplexity")
            plt.title("Train and Test Perplexity")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "training_perplexity.png", dpi=300, bbox_inches="tight")
            plt.close()
    except Exception as exc:
        logger.warning(f"[SAVE] Grafico training non creato: {exc}")


def _perplexity_from_overlaps(overlaps: np.ndarray) -> float:
    overlaps = np.clip(np.real(overlaps), 1e-12, 1.0)
    mean_sqrt = float(np.mean(np.sqrt(overlaps)))
    return float(np.exp(-2.0 * np.log(mean_sqrt))) if mean_sqrt > 0 else float("inf")


def _aggregate_overlap_metrics(local_overlaps: np.ndarray, comm, rank: int):
    probabilities = np.clip(np.real(local_overlaps), 1e-12, 1.0)
    local_count = int(probabilities.size)
    local_loss_sum = float(np.sum(-np.log(probabilities)))
    local_prob_sum = float(np.sum(probabilities))
    local_sqrt_sum = float(np.sum(np.sqrt(probabilities)))

    total_count = int(round(_allreduce_scalar(comm, np.asarray(local_count, dtype=np.float64))))
    total_loss_sum = _allreduce_scalar(comm, np.asarray(local_loss_sum, dtype=np.float64))
    total_prob_sum = _allreduce_scalar(comm, np.asarray(local_prob_sum, dtype=np.float64))
    total_sqrt_sum = _allreduce_scalar(comm, np.asarray(local_sqrt_sum, dtype=np.float64))

    if rank != 0:
        return None

    if total_count == 0:
        raise ValueError("Nessun campione disponibile per la valutazione distribuita.")

    mean_sqrt = total_sqrt_sum / total_count
    perplexity = float(np.exp(-2.0 * np.log(mean_sqrt))) if mean_sqrt > 0 else float("inf")
    return {
        "num_sequences": total_count,
        "loss_mean": total_loss_sum / total_count,
        "perplexity": perplexity,
        "avg_probability": total_prob_sum / total_count,
    }


def _evaluate_text_batch(params, layout, sentences: List[str], cfg: dict, comm=None, rank: int = 0, size: int = 1):
    local_sentences = _shard_items(sentences, rank, size)
    with _maybe_silence_stdout(rank != 0):
        encoding = Encoding(sentences, embeddingDim=cfg["embedding_dim"])
    _, text_data = _prepare_text_training_data(
        local_sentences,
        cfg,
        encoding=encoding,
        suppress_output=True,
    )
    _, infer_overlaps, _, _ = _build_text_functions(layout, cfg)

    if text_data["token_batch"].shape[0] == 0:
        overlaps = np.empty((0,), dtype=np.float64)
    else:
        eval_batch_size = int(cfg.get("eval_batch_size", cfg.get("train_batch_size", 4)))
        overlap_batches = []
        total = int(text_data["token_batch"].shape[0])
        for start, stop in _batch_ranges(total, eval_batch_size):
            batch = _slice_text_batch(text_data, start, stop)
            overlap_batches.append(
                _as_numpy(infer_overlaps(params, batch["token_batch"], batch["positional_encoding"]))
            )
        overlaps = np.concatenate(overlap_batches, axis=0) if overlap_batches else np.empty((0,), dtype=np.float64)

    return _aggregate_overlap_metrics(overlaps, comm, rank)


def _evaluate_quantum_batch(
    params,
    layout,
    sequences: np.ndarray,
    cfg: Optional[dict] = None,
    comm=None,
    rank: int = 0,
    size: int = 1,
):
    local_sequences = _shard_array(sequences, rank, size)
    cfg = dict(OPTIMIZATION_CONFIG if cfg is None else cfg)
    _, infer_overlaps, _, _ = _build_quantum_functions(layout, cfg)

    if local_sequences.shape[0] == 0:
        overlaps = np.empty((0,), dtype=np.float64)
    else:
        eval_batch_size = int(cfg.get("eval_batch_size", cfg.get("train_batch_size", 4)))
        quantum_data = {"state_batch": jnp.asarray(local_sequences, dtype=jnp.complex128)}
        overlap_batches = []
        total = int(quantum_data["state_batch"].shape[0])
        for start, stop in _batch_ranges(total, eval_batch_size):
            batch = _slice_quantum_batch(quantum_data, start, stop)
            overlap_batches.append(_as_numpy(infer_overlaps(params, batch["state_batch"])))
        overlaps = np.concatenate(overlap_batches, axis=0) if overlap_batches else np.empty((0,), dtype=np.float64)

    return _aggregate_overlap_metrics(overlaps, comm, rank)


def _resolve_text_splits_from_metadata(metadata: dict, cfg: dict, logger):
    train_sentences = list(metadata.get("text_train_sentences") or [])
    test_sentences = list(metadata.get("text_test_sentences") or [])
    if train_sentences or test_sentences:
        return train_sentences, test_sentences

    all_sentences = _load_text_sentences(logger, seed=int(cfg.get("seed", 42)))
    return _split_text_sentences(all_sentences, cfg)


def _run_cross_validation(
    logger,
    params: Dict[str, jnp.ndarray],
    layout: QuantumParameterLayout,
    cfg: dict,
    use_quantum_states: bool,
    comm=None,
    rank: int = 0,
    size: int = 1,
):
    if rank == 0:
        logger.info("[CV] Inizio valutazione post-training")
    if use_quantum_states:
        if rank == 0:
            sequences = _load_quantum_sequences(logger, seed=cfg.get("seed", 42) + 999)
        else:
            sequences = None
        sequences = _broadcast(comm, sequences, root=0)
        return _evaluate_quantum_batch(params, layout, sequences, cfg=cfg, comm=comm, rank=rank, size=size)

    if rank == 0:
        sentences = _load_text_sentences(logger, seed=int(cfg.get("seed", 42)))
    else:
        sentences = None
    sentences = _broadcast(comm, sentences, root=0)
    return _evaluate_text_batch(params, layout, sentences, cfg, comm=comm, rank=rank, size=size)


def run_saved_model_evaluation(matrices_dir: Path, cfg: dict, logger, comm=None, rank: int = 0, size: int = 1):
    if rank == 0:
        if (matrices_dir / "metadata.npy").exists():
            metadata = np.load(matrices_dir / "metadata.npy", allow_pickle=True).item()
        else:
            raise FileNotFoundError(f"metadata.npy non trovato in {matrices_dir}")

        best_params_native = np.load(matrices_dir / "best_params_native.npy")
    else:
        metadata = None
        best_params_native = None

    metadata = _broadcast(comm, metadata, root=0)
    best_params_native = _broadcast(comm, best_params_native, root=0)
    cfg = dict(cfg)
    cfg.setdefault("control_readout_mode", metadata.get("control_readout_mode", "auto"))
    cfg.setdefault("analytic_readout", metadata.get("analytic_readout", True))
    cfg.setdefault("analytic_quantum_device", metadata.get("analytic_quantum_device", "default.qubit"))
    cfg.setdefault("analytic_quantum_diff_method", metadata.get("analytic_quantum_diff_method", "backprop"))
    cfg.setdefault("train_embedding", metadata.get("train_embedding", True))
    cfg.setdefault("train_rotation", metadata.get("train_rotation", True))
    use_quantum_states = bool(metadata.get("use_quantum_states", False))
    layout = create_layout(
        sequence_length=int(metadata["sequence_length"]),
        feature_dimension=int(metadata["feature_dimension"]),
        num_layers=int(metadata["num_layers"]),
        non_linear_order=int(metadata.get("non_linear_order", 1)),
        prune_inactive_branches=bool(metadata.get("prune_inactive_branches", False)),
        active_branch_count=int(metadata.get("active_branch_count", max(int(metadata["sequence_length"]) - 1, 0))),
    )

    params = _unflatten_native_parameters(
        flat_params=best_params_native,
        layout=layout,
        embedding_shape=tuple(metadata["embedding_shape"]) if metadata.get("embedding_shape") else None,
        rotation_shape=tuple(metadata["rotation_shape"]) if metadata.get("rotation_shape") else None,
        projection_shape=tuple(metadata["projection_shape"]) if metadata.get("projection_shape") else None,
    )

    if use_quantum_states:
        metrics = _run_cross_validation(
            logger=logger,
            params=params,
            layout=layout,
            cfg=cfg,
            use_quantum_states=use_quantum_states,
            comm=comm,
            rank=rank,
            size=size,
        )
    else:
        if rank == 0:
            train_sentences, test_sentences = _resolve_text_splits_from_metadata(metadata, cfg, logger)
            if not test_sentences:
                test_sentences = train_sentences
        else:
            test_sentences = None
        test_sentences = _broadcast(comm, test_sentences, root=0)
        metrics = _evaluate_text_batch(
            params,
            layout,
            test_sentences,
            cfg,
            comm=comm,
            rank=rank,
            size=size,
        )
    if rank == 0:
        logger.info(f"[TEST-ONLY] Metriche: {metrics}")
    return metrics


def run_training(logger, cfg: dict, comm=None, rank: int = 0, size: int = 1):
    circuit_mode = str(cfg.get("circuit_mode", "section2")).lower()
    if circuit_mode == "section2":
        from qsa_training import run_qsa_training

        return run_qsa_training(logger=logger, cfg=cfg, comm=comm, rank=rank, size=size)

    seed = int(cfg.get("seed", 42))
    use_quantum_states = QUANTUM_STATES_CONFIG.get("use_quantum_states", False)

    if TEST_ONLY_CONFIG.get("skip_training", False):
        matrices_dir = Path(TEST_ONLY_CONFIG.get("matrices_dir", Path.cwd()))
        if (matrices_dir / "matrices").exists():
            matrices_dir = matrices_dir / "matrices"
        if rank == 0:
            logger.info(f"[TEST-ONLY] Valutazione da {matrices_dir}")
        run_saved_model_evaluation(matrices_dir, cfg, logger, comm=comm, rank=rank, size=size)
        return 0

    if use_quantum_states:
        if rank == 0:
            all_quantum_sequences = _load_quantum_sequences(logger, seed=seed)
            train_sequences, test_sequences = _split_array_dataset(all_quantum_sequences, cfg)
            logger.info(
                f"[DATA] Split quantum_states | train={train_sequences.shape[0]} "
                f"| test={test_sequences.shape[0]}"
            )
        else:
            train_sequences = None
            test_sequences = None
        train_sequences = _broadcast(comm, train_sequences, root=0)
        test_sequences = _broadcast(comm, test_sequences, root=0)
        local_sequences = _shard_array(train_sequences, rank, size)
        text_data = None
        encoding = None
        quantum_data = _prepare_quantum_training_data(local_sequences)
        projection_shape = get_projection_shape(QUANTUM_STATES_CONFIG)
        train_text_sentences = None
        test_text_sentences = None
    else:
        if rank == 0:
            all_sentences = _load_text_sentences(logger, seed=seed)
            train_text_sentences, test_text_sentences = _split_text_sentences(all_sentences, cfg)
            logger.info(
                f"[DATA] Split frasi | train={len(train_text_sentences)} | test={len(test_text_sentences)}"
            )
        else:
            all_sentences = None
            train_text_sentences = None
            test_text_sentences = None
        all_sentences = _broadcast(comm, all_sentences, root=0)
        train_text_sentences = _broadcast(comm, train_text_sentences, root=0)
        test_text_sentences = _broadcast(comm, test_text_sentences, root=0)
        local_sentences = _shard_items(train_text_sentences, rank, size)
        with _maybe_silence_stdout(rank != 0):
            encoding = Encoding(all_sentences, embeddingDim=cfg["embedding_dim"])
        _, text_data = _prepare_text_training_data(
            local_sentences,
            cfg,
            encoding=encoding,
            suppress_output=True,
        )
        quantum_data = None
        projection_shape = None
        train_sequences = None
        test_sequences = None

    layout = _build_layout(cfg, use_quantum_states, text_data, quantum_data)
    params = _initialize_trainable_params(
        layout=layout,
        cfg=cfg,
        seed=seed,
        encoding=encoding,
        projection_shape=projection_shape,
    )
    first_moment = zeros_like_tree(params)
    second_moment = zeros_like_tree(params)
    resume_params, resume_first_moment, resume_second_moment, resume_step_offset, resume_best_loss = _load_resume_state(
        checkpoint=cfg.get("resume_checkpoint"),
        layout=layout,
        template_params=params,
        logger=logger,
        rank=rank,
    )
    if resume_params is not None:
        params = resume_params
    if resume_first_moment is not None:
        first_moment = resume_first_moment
    if resume_second_moment is not None:
        second_moment = resume_second_moment

    if use_quantum_states:
        _, _, loss_and_grad, optimizer_step = _build_quantum_functions(layout, cfg)
        local_count = int(quantum_data["state_batch"].shape[0])
    else:
        _, _, loss_and_grad, optimizer_step = _build_text_functions(layout, cfg)
        local_count = int(text_data["token_batch"].shape[0])

    global_count = int(round(_allreduce_scalar(comm, np.asarray(local_count, dtype=np.float64))))
    if global_count <= 0:
        raise ValueError("Nessun campione disponibile per il training distribuito.")

    learning_rate = float(cfg.get("learning_rate", 1e-3))
    epochs = int(cfg.get("epochs", 100))
    train_batch_size = max(1, int(cfg.get("train_batch_size", local_count if local_count > 0 else 1)))
    batch_log_interval = max(1, int(cfg.get("batch_log_interval", 1)))
    gradient_clip_norm = float(cfg.get("gradient_clip_norm", 0.0))
    log_frequency = max(1, int(cfg.get("log_frequency", 10)))
    eval_frequency = max(1, int(cfg.get("eval_frequency", log_frequency)))
    device_name, interface, diff_method = _get_backend_settings(cfg)
    max_run_minutes = float(
        DATASET_CONFIG.get(
            "max_run_minutes",
            float(cfg.get("max_hours", 0.0)) * 60.0,
        )
    )
    max_seconds = max_run_minutes * 60.0 if max_run_minutes > 0 else None

    optimization_loss_history = []
    eval_history = []
    best_loss = float(resume_best_loss) if resume_best_loss is not None else float("inf")
    best_params = _copy_tree(params)
    best_epoch = int(resume_step_offset) if resume_best_loss is not None else 0
    stop_reason = "epochs_completed"
    start_time = time.perf_counter()
    checkpoint_dir = None

    if rank == 0:
        run_label = cfg.get("run_label")
        safe_label = "unlabeled"
        if run_label:
            safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(run_label))
        checkpoint_dir = (
            Path("results")
            / "checkpoints"
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_label}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[TRAIN] Backend JAX+PennyLane+MPI | epochs={epochs} | lr={learning_rate} "
            f"| padded_sequence_length={layout.padded_sequence_length}"
        )
        if resume_step_offset > 0:
            logger.info(
                f"[RESUME] Continuazione da epoch={resume_step_offset}; "
                f"target totale={epochs}; epoche rimanenti={max(epochs - resume_step_offset, 0)}"
            )
        if run_label:
            logger.info(f"[TRAIN] Run label={run_label}")
        logger.info(
            f"[TRAIN] Device={device_name} | interface={interface} | diff_method={diff_method} "
            f"| train_batch_size={train_batch_size}"
        )
        logger.info(
            f"[MPI] ranks={size} | sample_globali={global_count} | "
            f"sample_locali_medi~={math.ceil(global_count / max(size, 1))}"
        )
        if max_seconds is not None:
            logger.info(
                f"[TRAIN] Time budget attivo: {max_run_minutes:.1f} minuti "
                f"(~{max_seconds / 3600.0:.2f} ore)"
            )
        logger.info("[TRAIN] Esecuzione quantistica in mini-batch con backend Lightning")

    for epoch in range(resume_step_offset + 1, epochs + 1):
        if local_count == 0:
            local_loss_sum = jnp.asarray(0.0, dtype=jnp.float64)
            local_grads = zeros_like_tree(params)
        else:
            local_loss_sum = jnp.asarray(0.0, dtype=jnp.float64)
            local_grads = zeros_like_tree(params)
            batch_ranges = list(_batch_ranges(local_count, train_batch_size))
            total_local_batches = len(batch_ranges)
            epoch_batch_start = time.perf_counter()
            for batch_idx, (start, stop) in enumerate(batch_ranges, start=1):
                if use_quantum_states:
                    batch = _slice_quantum_batch(quantum_data, start, stop)
                    batch_loss_sum, batch_grads = loss_and_grad(
                        params,
                        batch["state_batch"],
                    )
                else:
                    batch = _slice_text_batch(text_data, start, stop)
                    batch_loss_sum, batch_grads = loss_and_grad(
                        params,
                        batch["token_batch"],
                        batch["positional_encoding"],
                    )
                local_loss_sum = local_loss_sum + batch_loss_sum
                local_grads = _sum_tree(local_grads, batch_grads)
                if rank == 0 and (
                    batch_idx == 1
                    or batch_idx == total_local_batches
                    or batch_idx % batch_log_interval == 0
                ):
                    elapsed_batch = time.perf_counter() - epoch_batch_start
                    logger.info(
                        f"[TRAIN] epoch={epoch}/{epochs} | batch={batch_idx}/{total_local_batches} "
                        f"| sample_locali={stop}/{local_count} | elapsed_epoch={elapsed_batch:.1f}s"
                    )

        global_loss_sum = _allreduce_scalar(comm, np.asarray(_as_float(local_loss_sum), dtype=np.float64))
        global_grads = _allreduce_tree(comm, local_grads)
        mean_grads = jax.tree_util.tree_map(
            lambda grad: grad / global_count,
            global_grads,
        )
        mean_grads, grad_norm = _clip_grad_tree(mean_grads, gradient_clip_norm)
        params = _sanitize_tree(params)
        params, first_moment, second_moment = optimizer_step(
            params,
            mean_grads,
            first_moment,
            second_moment,
            epoch,
            learning_rate,
        )
        first_moment = _sanitize_tree(first_moment)
        second_moment = _sanitize_tree(second_moment)
        params = _sanitize_tree(params)

        loss_value = global_loss_sum / global_count
        optimization_loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = _copy_tree(params)
            best_epoch = epoch
            if rank == 0 and checkpoint_dir is not None:
                _save_training_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    params=best_params,
                    first_moment=first_moment,
                    second_moment=second_moment,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                    optimizer_step=epoch,
                    run_label=cfg.get("run_label"),
                )

        should_log = epoch == 1 or epoch % log_frequency == 0 or epoch == epochs
        should_eval = epoch == 1 or epoch % eval_frequency == 0 or epoch == epochs

        current_train_metrics = None
        current_test_metrics = None

        if should_eval:
            if use_quantum_states:
                current_train_metrics = _evaluate_quantum_batch(
                    params,
                    layout,
                    train_sequences,
                    cfg=cfg,
                    comm=comm,
                    rank=rank,
                    size=size,
                )
                current_test_metrics = _evaluate_quantum_batch(
                    params,
                    layout,
                    test_sequences,
                    cfg=cfg,
                    comm=comm,
                    rank=rank,
                    size=size,
                ) if len(test_sequences) > 0 else None
            else:
                current_train_metrics = _evaluate_text_batch(
                    params,
                    layout,
                    train_text_sentences,
                    cfg,
                    comm=comm,
                    rank=rank,
                    size=size,
                )
                current_test_metrics = _evaluate_text_batch(
                    params,
                    layout,
                    test_text_sentences,
                    cfg,
                    comm=comm,
                    rank=rank,
                    size=size,
                ) if len(test_text_sentences) > 0 else None

            if rank == 0 and current_train_metrics is not None:
                elapsed_minutes = (time.perf_counter() - start_time) / 60.0
                eval_history.append(
                    {
                        "epoch": epoch,
                        "elapsed_minutes": elapsed_minutes,
                        "optimization_loss": loss_value,
                        "train_loss": float(current_train_metrics["loss_mean"]),
                        "train_ppl": float(current_train_metrics["perplexity"]),
                        "test_loss": float(current_test_metrics["loss_mean"]) if current_test_metrics else float("nan"),
                        "test_ppl": float(current_test_metrics["perplexity"]) if current_test_metrics else float("nan"),
                    }
                )

        if rank == 0 and should_log:
            message = (
                f"[TRAIN] Epoch {epoch:04d}/{epochs:04d} | opt_loss={loss_value:.8f}"
                f" | grad_norm={grad_norm:.6f}"
            )
            if current_train_metrics is not None:
                message += (
                    f" | train_loss={current_train_metrics['loss_mean']:.8f}"
                    f" | train_ppl={current_train_metrics['perplexity']:.8f}"
                )
            if current_test_metrics is not None:
                message += (
                    f" | test_loss={current_test_metrics['loss_mean']:.8f}"
                    f" | test_ppl={current_test_metrics['perplexity']:.8f}"
                )
            logger.info(message)

        if max_seconds is not None:
            local_stop_flag = 1.0 if (time.perf_counter() - start_time) >= max_seconds else 0.0
            global_stop_flag = _allreduce_scalar(comm, np.asarray(local_stop_flag, dtype=np.float64))
            if global_stop_flag > 0.0:
                stop_reason = f"time_budget_reached_{max_run_minutes:.1f}m"
                if rank == 0:
                    logger.info(f"[TRAIN] Stop per budget temporale raggiunto a epoch={epoch}")
                break

    if use_quantum_states:
        final_train_metrics = _evaluate_quantum_batch(
            best_params,
            layout,
            train_sequences,
            cfg=cfg,
            comm=comm,
            rank=rank,
            size=size,
        )
        final_test_metrics = _evaluate_quantum_batch(
            best_params,
            layout,
            test_sequences,
            cfg=cfg,
            comm=comm,
            rank=rank,
            size=size,
        ) if len(test_sequences) > 0 else None
    else:
        if rank == 0:
            logger.info("[EVAL] Valutazione finale su train/test holdout")
        final_train_metrics = _evaluate_text_batch(
            best_params,
            layout,
            train_text_sentences,
            cfg,
            comm=comm,
            rank=rank,
            size=size,
        )
        final_test_metrics = _evaluate_text_batch(
            best_params,
            layout,
            test_text_sentences,
            cfg,
            comm=comm,
            rank=rank,
            size=size,
        ) if len(test_text_sentences) > 0 else None

    elapsed_seconds = time.perf_counter() - start_time

    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir, matrices_dir, parameters_dir, summaries_dir, plots_dir = _prepare_output_dirs(
            timestamp=timestamp,
            seed=seed,
            run_label=cfg.get("run_label"),
        )
        run_id = run_dir.name

        best_params_native, metadata = _save_matrices_and_metadata(
            params=best_params,
            layout=layout,
            cfg=cfg,
            matrices_dir=matrices_dir,
            run_id=run_id,
            use_quantum_states=use_quantum_states,
            text_train_sentences=train_text_sentences,
            text_test_sentences=test_text_sentences,
        )
        np.save(parameters_dir / "theta_finali_native.npy", best_params_native)

        n_quantum, n_embedding, n_rotation, n_projection = _compute_param_counts(best_params, layout)
        _save_summary_and_plot(
            logger=logger,
            output_dir=run_dir,
            optimization_loss_history=optimization_loss_history,
            best_loss=best_loss,
            best_epoch=best_epoch,
            n_params_total=n_quantum + n_embedding + n_rotation + n_projection,
            n_params_quantum=n_quantum,
            n_params_embedding=n_embedding,
            n_params_rotation=n_rotation,
            n_params_projection=n_projection,
            train_metrics=final_train_metrics,
            test_metrics=final_test_metrics,
            eval_history=eval_history,
            stop_reason=stop_reason,
            elapsed_seconds=elapsed_seconds,
        )

        logger.info(
            f"[EVAL] Final train | loss={final_train_metrics['loss_mean']:.8f} "
            f"| ppl={final_train_metrics['perplexity']:.8f}"
        )
        if final_test_metrics is not None:
            logger.info(
                f"[EVAL] Final test  | loss={final_test_metrics['loss_mean']:.8f} "
                f"| ppl={final_test_metrics['perplexity']:.8f}"
            )
        logger.info(f"[SAVE] Run completata: {run_dir}")
        logger.info(f"[SAVE] Matrici: {matrices_dir}")
        logger.info(f"[SAVE] Parametri: {parameters_dir}")
        logger.info(f"[SAVE] Summary: {summaries_dir}")
        logger.info(f"[SAVE] Plots: {plots_dir}")

    _barrier(comm)
    clear_memory()
    return 0
