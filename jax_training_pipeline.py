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
    get_vmapped_qnode,
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
    else:
        feature_dimension = int(cfg["embedding_dim"])
        sequence_length = int(text_data["token_batch"].shape[1])

    return create_layout(
        sequence_length=sequence_length,
        feature_dimension=feature_dimension,
        num_layers=cfg["num_layers"],
        non_linear_order=1,
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


def _compute_param_counts(params: Dict[str, jnp.ndarray], layout: QuantumParameterLayout):
    n_quantum = layout.total_parameters
    n_embedding = int(np.prod(params["embedding"].shape)) if "embedding" in params else 0
    n_rotation = int(np.prod(params["rotation"].shape)) if "rotation" in params else 0
    n_projection = int(np.prod(params["projection"].shape)) if "projection" in params else 0
    return n_quantum, n_embedding, n_rotation, n_projection


def _build_text_functions(layout: QuantumParameterLayout):
    vmapped_qnode = get_vmapped_qnode(layout)

    def _loss_sum_impl(params, token_batch, positional_encoding):
        x_batch, tilde_batch, prep_batch, _, _, _ = prepare_text_batch(
            token_batch=token_batch,
            positional_encoding=positional_encoding,
            raw_embedding=params["embedding"],
            raw_rotation=params["rotation"],
            layout=layout,
        )
        overlaps = vmapped_qnode(
            x_batch,
            tilde_batch,
            prep_batch,
            params["weights_v"],
            params["weights_w"],
            params["weights_c"],
        )
        return jnp.sum(_overlaps_to_losses(overlaps))

    @jax.jit
    def compute_loss_sum(params, token_batch, positional_encoding):
        return _loss_sum_impl(params, token_batch, positional_encoding)

    @jax.jit
    def infer_overlaps(params, token_batch, positional_encoding):
        x_batch, tilde_batch, prep_batch, _, _, _ = prepare_text_batch(
            token_batch=token_batch,
            positional_encoding=positional_encoding,
            raw_embedding=params["embedding"],
            raw_rotation=params["rotation"],
            layout=layout,
        )
        return vmapped_qnode(
            x_batch,
            tilde_batch,
            prep_batch,
            params["weights_v"],
            params["weights_w"],
            params["weights_c"],
        )

    @jax.jit
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


def _build_quantum_functions(layout: QuantumParameterLayout):
    vmapped_qnode = get_vmapped_qnode(layout)

    def _loss_sum_impl(params, state_batch):
        x_batch, tilde_batch, prep_batch = prepare_quantum_batch(
            state_batch=state_batch,
            layout=layout,
            raw_projection=params.get("projection"),
        )
        overlaps = vmapped_qnode(
            x_batch,
            tilde_batch,
            prep_batch,
            params["weights_v"],
            params["weights_w"],
            params["weights_c"],
        )
        return jnp.sum(_overlaps_to_losses(overlaps))

    @jax.jit
    def compute_loss_sum(params, state_batch):
        return _loss_sum_impl(params, state_batch)

    @jax.jit
    def infer_overlaps(params, state_batch):
        x_batch, tilde_batch, prep_batch = prepare_quantum_batch(
            state_batch=state_batch,
            layout=layout,
            raw_projection=params.get("projection"),
        )
        return vmapped_qnode(
            x_batch,
            tilde_batch,
            prep_batch,
            params["weights_v"],
            params["weights_w"],
            params["weights_c"],
        )

    @jax.jit
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


def _prepare_output_dirs(timestamp: str, seed: int) -> Tuple[Path, Path, Path, Path]:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    run_dir = results_dir / f"run_{timestamp}_seed{seed}"
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
        "use_quantum_states": use_quantum_states,
        "sequence_length": layout.sequence_length,
        "padded_sequence_length": layout.padded_sequence_length,
        "feature_dimension": layout.feature_dimension,
        "feature_qubits": layout.feature_qubits,
        "control_qubits": layout.control_qubits,
        "num_layers": layout.num_layers,
        "non_linear_order": layout.non_linear_order,
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
    _, infer_overlaps, _, _ = _build_text_functions(layout)

    if text_data["token_batch"].shape[0] == 0:
        overlaps = np.empty((0,), dtype=np.float64)
    else:
        overlaps = _as_numpy(
            infer_overlaps(params, text_data["token_batch"], text_data["positional_encoding"])
        )

    return _aggregate_overlap_metrics(overlaps, comm, rank)


def _evaluate_quantum_batch(params, layout, sequences: np.ndarray, comm=None, rank: int = 0, size: int = 1):
    local_sequences = _shard_array(sequences, rank, size)
    _, infer_overlaps, _, _ = _build_quantum_functions(layout)

    if local_sequences.shape[0] == 0:
        overlaps = np.empty((0,), dtype=np.float64)
    else:
        overlaps = _as_numpy(
            infer_overlaps(params, jnp.asarray(local_sequences, dtype=jnp.complex128))
        )

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
        return _evaluate_quantum_batch(params, layout, sequences, comm=comm, rank=rank, size=size)

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
    use_quantum_states = bool(metadata.get("use_quantum_states", False))
    layout = create_layout(
        sequence_length=int(metadata["sequence_length"]),
        feature_dimension=int(metadata["feature_dimension"]),
        num_layers=int(metadata["num_layers"]),
        non_linear_order=int(metadata.get("non_linear_order", 1)),
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

    if use_quantum_states:
        _, _, loss_and_grad, optimizer_step = _build_quantum_functions(layout)
        local_count = int(quantum_data["state_batch"].shape[0])
    else:
        _, _, loss_and_grad, optimizer_step = _build_text_functions(layout)
        local_count = int(text_data["token_batch"].shape[0])

    global_count = int(round(_allreduce_scalar(comm, np.asarray(local_count, dtype=np.float64))))
    if global_count <= 0:
        raise ValueError("Nessun campione disponibile per il training distribuito.")

    learning_rate = float(cfg.get("learning_rate", 1e-3))
    epochs = int(cfg.get("epochs", 100))
    log_frequency = max(1, int(cfg.get("log_frequency", 10)))
    eval_frequency = max(1, int(cfg.get("eval_frequency", log_frequency)))
    max_hours = float(cfg.get("max_hours", 0.0))
    max_seconds = max_hours * 3600.0 if max_hours > 0 else None

    optimization_loss_history = []
    eval_history = []
    best_loss = float("inf")
    best_params = _copy_tree(params)
    best_epoch = 0
    stop_reason = "epochs_completed"
    start_time = time.perf_counter()

    if rank == 0:
        logger.info(
            f"[TRAIN] Backend JAX+PennyLane+MPI | epochs={epochs} | lr={learning_rate} "
            f"| padded_sequence_length={layout.padded_sequence_length}"
        )
        logger.info(
            f"[MPI] ranks={size} | sample_globali={global_count} | "
            f"sample_locali_medi~={math.ceil(global_count / max(size, 1))}"
        )
        if max_seconds is not None:
            logger.info(f"[TRAIN] Time budget attivo: {max_hours:.2f} ore (~{max_seconds / 60.0:.1f} minuti)")
        logger.info("[TRAIN] Warmup JIT alla prima iterazione di ogni rank")

    for epoch in range(1, epochs + 1):
        if local_count == 0:
            local_loss_sum = jnp.asarray(0.0, dtype=jnp.float64)
            local_grads = zeros_like_tree(params)
        else:
            if use_quantum_states:
                local_loss_sum, local_grads = loss_and_grad(
                    params,
                    quantum_data["state_batch"],
                )
            else:
                local_loss_sum, local_grads = loss_and_grad(
                    params,
                    text_data["token_batch"],
                    text_data["positional_encoding"],
                )

        global_loss_sum = _allreduce_scalar(comm, np.asarray(_as_float(local_loss_sum), dtype=np.float64))
        global_grads = _allreduce_tree(comm, local_grads)
        mean_grads = jax.tree_util.tree_map(
            lambda grad: grad / global_count,
            global_grads,
        )
        params, first_moment, second_moment = optimizer_step(
            params,
            mean_grads,
            first_moment,
            second_moment,
            epoch,
            learning_rate,
        )

        loss_value = global_loss_sum / global_count
        optimization_loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = _copy_tree(params)
            best_epoch = epoch

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
                    comm=comm,
                    rank=rank,
                    size=size,
                )
                current_test_metrics = _evaluate_quantum_batch(
                    params,
                    layout,
                    test_sequences,
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
            message = f"[TRAIN] Epoch {epoch:04d}/{epochs:04d} | opt_loss={loss_value:.8f}"
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
                stop_reason = f"time_budget_reached_{max_hours:.2f}h"
                if rank == 0:
                    logger.info(f"[TRAIN] Stop per budget temporale raggiunto a epoch={epoch}")
                break

    if use_quantum_states:
        final_train_metrics = _evaluate_quantum_batch(
            best_params,
            layout,
            train_sequences,
            comm=comm,
            rank=rank,
            size=size,
        )
        final_test_metrics = _evaluate_quantum_batch(
            best_params,
            layout,
            test_sequences,
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
