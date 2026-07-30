"""
Classical baselines for comparison with k-QSA (Section 2).

k-QSA  — quantum-angle parametrization (RY layers → orthogonal W, V) + -log(mu).
k-CSA  — independent classical baseline: learnable d×d matrices W, V (no gate angles).
         Same dataset / training conditions; separate data_seed and model_seed.
nl-CSA — embedding → causal softmax attention → residual norm → FFN GELU → logits.
         Four ablations via nl_embedding_mode × nl_loss_mode.

Common evaluation: validation cross-entropy / perplexity on a held-out split
(train loss remains model-specific and must not be compared across architectures).
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from config import DATASET_CONFIG
from encoding import Encoding
from pennylane_jax_vqt import isometrize_matrix
from qsa_training import (
    QSATrainConfig,
    L_half_uniform_jax,
    _feature_qubits,
    _normalize_rows,
    classical_mu_jax,
    init_qsa_params,
    loss_B_jax,
    ortho_matrix_jax,
    sequence_xy_from_tokens,
)

jax.config.update("jax_enable_x64", True)


def count_tree_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.asarray(x).size for x in leaves))


def qsa_angle_param_count(d: int, layers: int) -> int:
    n = _feature_qubits(d)
    return 2 * layers * n


def kcsa_matrix_param_count(d: int) -> int:
    """k-CSA uses two full d×d matrices W and V (classical, not angle-parametrized)."""
    return 2 * d * d


@dataclass
class BaselineConfig:
    T: int
    d: int
    k: int = 2
    layers: int = 2
    epochs: int = 40
    learning_rate: float = 1e-3
    max_sentences: int = 64
    seed: int = 42
    data_seed: Optional[int] = None
    model_seed: Optional[int] = None
    train_embedding: bool = True
    epsilon: float = 1e-12
    run_label: Optional[str] = None
    output_dir: Optional[str] = None
    nl_rank: Optional[int] = None
    nl_learning_rate: Optional[float] = None
    nl_logit_scale: float = 8.0
    nl_embedding_mode: str = "isometric"  # "general" | "isometric"
    nl_loss_mode: str = "renyi"  # "ce" | "renyi"
    batch_size: Optional[int] = None
    checkpoint_every: int = 20
    resume: bool = True
    kernel_mode: str = "monomial"
    loss_objective: str = "mu"  # "mu" (-log mu) | "L_B" (objective B)
    val_fraction: float = 0.2
    test_data_seed: Optional[int] = None
    test_max_sentences: int = 200
    # Early stopping / plateau (diagnostic; avoids mega-runs)
    early_stop: bool = False
    max_epochs: Optional[int] = None
    loss_rel_tol: float = 1e-4
    convergence_patience: int = 12
    track_grad_norm: bool = True

    def __post_init__(self) -> None:
        if self.data_seed is None:
            self.data_seed = self.seed
        if self.model_seed is None:
            self.model_seed = self.seed + 1000
        if self.max_epochs is None:
            self.max_epochs = self.epochs

    @property
    def parametrization(self) -> dict:
        """Human-readable parametrization summary for reporting."""
        return {
            "k-QSA": (
                f"isometric embedding (vocab×d) + quantum-angle ortho blocks "
                f"(weights_w, weights_v: layers×log2(d) RY angles each); "
                f"angles={qsa_angle_param_count(self.d, self.layers)}"
            ),
            "k-CSA": (
                f"isometric embedding (vocab×d) + classical QR-orthogonal W, V "
                f"(learnable W_raw, V_raw: d×d each, orthogonalized via QR; "
                f"no quantum gate angles); raw params={kcsa_matrix_param_count(self.d)}"
            ),
            "nl-CSA": (
                f"embedding_mode={self.nl_embedding_mode}, loss_mode={self.nl_loss_mode}; "
                f"arch: embedding → causal softmax attn → residual norm → "
                f"FFN GELU → tied logits (k-independent)"
            ),
        }


def _load_sentences_with_seed(
    T: int, max_sentences: int, data_seed: int
) -> List[str]:
    """Seeded sentence sample from local PTB (fixed T words per line)."""
    rng = random.Random(data_seed)
    local = "ptb_sentences.txt"
    with open(local, "r", encoding="utf-8") as f:
        valid = [line.strip() for line in f if line.strip() and len(line.split()) == T]
    if len(valid) <= max_sentences:
        sentences = valid
    else:
        sentences = rng.sample(valid, max_sentences)
    if len(sentences) < max_sentences:
        expanded = list(sentences)
        while len(expanded) < max_sentences:
            expanded.extend(sentences)
        sentences = expanded[:max_sentences]
    return sentences


def _load_sentences(cfg: BaselineConfig) -> List[str]:
    """Seeded sentence sample; all models see the exact same data."""
    return _load_sentences_with_seed(cfg.T, cfg.max_sentences, int(cfg.data_seed))


def _split_train_val(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 7)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.size == 0:
        train_idx = perm[n_val - 1 :]
        val_idx = perm[:1]
    return train_idx, val_idx


def prepare_data_bundle(cfg: BaselineConfig) -> dict:
    """Dataset + train/val split shared by all baselines (data_seed only)."""
    DATASET_CONFIG["sentence_length"] = cfg.T
    DATASET_CONFIG["max_sentences"] = cfg.max_sentences
    sentences = _load_sentences(cfg)
    encoding = Encoding(sentences, embeddingDim=cfg.d, embeddingSeed=cfg.data_seed)
    token_batch = jnp.stack([encoding.encode_tokens(s) for s in sentences])
    pe = encoding._positionalEncoding(cfg.T)
    n = int(token_batch.shape[0])
    train_idx, val_idx = _split_train_val(n, cfg.val_fraction, cfg.data_seed)
    qcfg = QSATrainConfig(
        T=cfg.T,
        d=cfg.d,
        k=cfg.k,
        layers=cfg.layers,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        max_sentences=cfg.max_sentences,
        seed=cfg.data_seed,
        train_embedding=cfg.train_embedding,
        epsilon=cfg.epsilon,
        local_max_qubits=32,
        kernel_mode=cfg.kernel_mode,
    )
    return {
        "sentences": sentences,
        "encoding": encoding,
        "token_batch": token_batch,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "pe": pe,
        "qcfg": qcfg,
    }


def attach_test_bundle(bundle: dict, cfg: BaselineConfig) -> dict:
    """Hold-out PTB sample (different seed) encoded with the train vocabulary."""
    if cfg.test_data_seed is None:
        return bundle
    encoding = bundle["encoding"]
    test_sentences = _load_sentences_with_seed(
        cfg.T, int(cfg.test_max_sentences), int(cfg.test_data_seed)
    )
    test_token_batch = jnp.stack([encoding.encode_tokens(s) for s in test_sentences])
    bundle = dict(bundle)
    bundle["test_sentences"] = test_sentences
    bundle["test_token_batch"] = test_token_batch
    bundle["num_test"] = int(test_token_batch.shape[0])
    return bundle


def eval_batch_mean_loss(params, batch, pe, cfg: BaselineConfig, loss_fn) -> float:
    return float(jnp.mean(jax.vmap(lambda ids: loss_fn(params, ids, pe, cfg))(batch)))


def prepare_shared_bundle(cfg: BaselineConfig) -> dict:
    """Backward-compatible wrapper: data bundle + legacy k-QSA init (params0)."""
    bundle = prepare_data_bundle(cfg)
    key = jax.random.PRNGKey(cfg.model_seed)
    bundle["params0"] = init_qsa_params(bundle["encoding"].vocabSize, bundle["qcfg"], key)
    return bundle


# --------------------------------------------------------------------------- #
#  k-QSA / k-CSA losses
# --------------------------------------------------------------------------- #
def _embedding_forward(emb_raw, cfg: BaselineConfig):
    if not cfg.train_embedding:
        emb_raw = jax.lax.stop_gradient(emb_raw)
    return isometrize_matrix(emb_raw)


def mu_loss_sentence(params, token_ids, pe, cfg: BaselineConfig):
    """k-QSA: quantum-angle orthogonal W, V."""
    emb = _embedding_forward(params["embedding"], cfg)
    X, Y = sequence_xy_from_tokens(token_ids, emb, pe)
    W = ortho_matrix_jax(params["weights_w"], cfg.d)
    V = ortho_matrix_jax(params["weights_v"], cfg.d)
    mu = classical_mu_jax(X, Y, W, V, cfg.k, kernel_mode=cfg.kernel_mode)
    return -jnp.log(jnp.maximum(mu, cfg.epsilon))


def lb_kqsa_loss_sentence(params, token_ids, pe, cfg: BaselineConfig):
    """k-QSA train loss = L_B (real ansatz, no trainable phases)."""
    emb = _embedding_forward(params["embedding"], cfg)
    X, Y = sequence_xy_from_tokens(token_ids, emb, pe)
    W = ortho_matrix_jax(params["weights_w"], cfg.d)
    V = ortho_matrix_jax(params["weights_v"], cfg.d)
    return loss_B_jax(X, Y, W, V, cfg.k, kernel_mode=cfg.kernel_mode, epsilon=cfg.epsilon)


def l_half_uniform_kqsa_sentence(params, token_ids, pe, cfg: BaselineConfig):
    """k-QSA eval: L_half_uniform at fixed W,V (uniform Renyi-1/2 CE)."""
    emb = _embedding_forward(params["embedding"], cfg)
    X, Y = sequence_xy_from_tokens(token_ids, emb, pe)
    W = ortho_matrix_jax(params["weights_w"], cfg.d)
    V = ortho_matrix_jax(params["weights_v"], cfg.d)
    return L_half_uniform_jax(X, Y, W, V, cfg.k, kernel_mode=cfg.kernel_mode, epsilon=cfg.epsilon)


def _kqsa_loss_for_cfg(cfg: BaselineConfig):
    return lb_kqsa_loss_sentence if cfg.loss_objective == "L_B" else mu_loss_sentence


def _orthogonalize_classical(raw: jnp.ndarray) -> jnp.ndarray:
    """Classical QR orthogonalization (independent of quantum RY gate angles)."""
    Q, R = jnp.linalg.qr(raw)
    signs = jnp.sign(jnp.diag(R))
    signs = jnp.where(signs == 0, 1.0, signs)
    return Q * signs[jnp.newaxis, :]


def kcsa_mu_loss_sentence(params, token_ids, pe, cfg: BaselineConfig):
    """k-CSA: classical QR-orthogonal W, V (no quantum gate angles)."""
    emb = _embedding_forward(params["embedding"], cfg)
    X, Y = sequence_xy_from_tokens(token_ids, emb, pe)
    W = _orthogonalize_classical(params["W_raw"])
    V = _orthogonalize_classical(params["V_raw"])
    mu = classical_mu_jax(X, Y, W, V, cfg.k, kernel_mode=cfg.kernel_mode)
    return -jnp.log(jnp.maximum(mu, cfg.epsilon))


def lb_kcsa_loss_sentence(params, token_ids, pe, cfg: BaselineConfig):
    """k-CSA train loss = L_B."""
    emb = _embedding_forward(params["embedding"], cfg)
    X, Y = sequence_xy_from_tokens(token_ids, emb, pe)
    W = _orthogonalize_classical(params["W_raw"])
    V = _orthogonalize_classical(params["V_raw"])
    return loss_B_jax(X, Y, W, V, cfg.k, kernel_mode=cfg.kernel_mode, epsilon=cfg.epsilon)


def l_half_uniform_kcsa_sentence(params, token_ids, pe, cfg: BaselineConfig):
    emb = _embedding_forward(params["embedding"], cfg)
    X, Y = sequence_xy_from_tokens(token_ids, emb, pe)
    W = _orthogonalize_classical(params["W_raw"])
    V = _orthogonalize_classical(params["V_raw"])
    return L_half_uniform_jax(X, Y, W, V, cfg.k, kernel_mode=cfg.kernel_mode, epsilon=cfg.epsilon)


def _kcsa_loss_for_cfg(cfg: BaselineConfig):
    return lb_kcsa_loss_sentence if cfg.loss_objective == "L_B" else kcsa_mu_loss_sentence


def _l_half_uniform_for_model(model: str):
    if model == "k-QSA":
        return l_half_uniform_kqsa_sentence
    return l_half_uniform_kcsa_sentence


def init_kcsa_params(vocab_size: int, cfg: BaselineConfig, key: jax.Array) -> dict:
    key_e, key_w, key_v = jax.random.split(key, 3)
    emb = isometrize_matrix(
        jax.random.normal(key_e, (vocab_size, cfg.d), dtype=jnp.float64)
    )
    scale = (2.0 / max(cfg.d, 1)) ** 0.5
    W_raw = jax.random.normal(key_w, (cfg.d, cfg.d), dtype=jnp.float64) * scale
    V_raw = jax.random.normal(key_v, (cfg.d, cfg.d), dtype=jnp.float64) * scale
    return {"embedding": emb, "W_raw": W_raw, "V_raw": V_raw}


def eval_mu_val_perplexity(params, val_batch, pe, cfg: BaselineConfig, loss_fn) -> float:
    losses = jax.vmap(lambda ids: loss_fn(params, ids, pe, cfg))(val_batch)
    return float(jnp.exp(jnp.mean(losses)))


# --------------------------------------------------------------------------- #
#  nl-CSA architecture
# --------------------------------------------------------------------------- #
def _default_nl_rank(d: int, layers: int) -> int:
    return int(min(d, max(2, int(round(d ** 0.5)))))


def nl_model_param_count(d: int, layers: int, rank: int) -> int:
    """Model params excluding embedding: 6 matrices per layer of shape (d, r)."""
    return int(6 * layers * d * rank)


def nl_rank_for_budget(d: int, layers: int, target_params: int) -> int:
    """Choose integer rank so that 6*L*d*r ≈ target_params (at least 1)."""
    denom = max(6 * layers * d, 1)
    return max(1, int(round(target_params / denom)))


def _apply_nl_embedding(emb_raw, cfg: BaselineConfig):
    if not cfg.train_embedding:
        emb_raw = jax.lax.stop_gradient(emb_raw)
    if cfg.nl_embedding_mode == "isometric":
        return isometrize_matrix(emb_raw)
    return emb_raw


def init_nlcsa_params(vocab_size: int, cfg: BaselineConfig, key: jax.Array) -> Tuple[dict, int]:
    r = cfg.nl_rank if cfg.nl_rank is not None else _default_nl_rank(cfg.d, cfg.layers)
    keys = jax.random.split(key, 1 + 6 * cfg.layers)
    raw_emb = jax.random.normal(keys[0], (vocab_size, cfg.d), dtype=jnp.float64)
    if cfg.nl_embedding_mode == "isometric":
        emb = isometrize_matrix(raw_emb)
    else:
        emb = raw_emb
    scale = (2.0 / max(cfg.d, 1)) ** 0.5
    layers = []
    ki = 1
    for _ in range(cfg.layers):
        layers.append(
            {
                "Wq": jax.random.normal(keys[ki], (cfg.d, r), dtype=jnp.float64) * scale,
                "Wk": jax.random.normal(keys[ki + 1], (cfg.d, r), dtype=jnp.float64) * scale,
                "Wv": jax.random.normal(keys[ki + 2], (cfg.d, r), dtype=jnp.float64) * scale,
                "Wo": jax.random.normal(keys[ki + 3], (r, cfg.d), dtype=jnp.float64) * scale,
                "W1": jax.random.normal(keys[ki + 4], (cfg.d, r), dtype=jnp.float64) * scale,
                "W2": jax.random.normal(keys[ki + 5], (r, cfg.d), dtype=jnp.float64) * scale,
            }
        )
        ki += 6
    return {"embedding": emb, "layers": layers}, r


def _causal_softmax_attn(X, Wq, Wk, Wv, Wo, logit_scale: float):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = (Q @ K.T) * (logit_scale / jnp.sqrt(jnp.maximum(float(Q.shape[-1]), 1.0)))
    T = X.shape[0]
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.float64))
    scores = jnp.where(mask > 0, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return (weights @ V) @ Wo


def nlcsa_logits_sentence(params, token_ids, pe, cfg: BaselineConfig):
    emb = _apply_nl_embedding(params["embedding"], cfg)
    X = _normalize_rows(emb[token_ids] + pe)
    h = X
    for layer in params["layers"]:
        attn = _causal_softmax_attn(
            h, layer["Wq"], layer["Wk"], layer["Wv"], layer["Wo"], cfg.nl_logit_scale
        )
        h = _normalize_rows(h + attn)
        ff = jax.nn.gelu(h @ layer["W1"]) @ layer["W2"]
        h = _normalize_rows(h + ff)
    return (h @ emb.T) * cfg.nl_logit_scale


def cross_entropy_loss_from_logits(logits, token_ids, eps: float = 1e-12):
    targets = token_ids[1:]
    logits_t = logits[:-1]
    log_probs = jax.nn.log_softmax(logits_t, axis=-1)
    nll = -log_probs[jnp.arange(targets.shape[0]), targets]
    return jnp.mean(nll)


def renyi_loss_from_logits(logits, token_ids, eps=1e-12):
    targets = token_ids[1:]
    logits_t = logits[:-1]
    probs = jax.nn.softmax(logits_t, axis=-1)
    p_true = probs[jnp.arange(targets.shape[0]), targets]
    f_bar = jnp.mean(jnp.sqrt(jnp.maximum(p_true, eps)))
    ppl = jnp.maximum(f_bar, eps) ** (-2)
    return jnp.log(jnp.maximum(ppl, eps)), ppl


def nlcsa_loss_sentence(params, token_ids, pe, cfg: BaselineConfig):
    logits = nlcsa_logits_sentence(params, token_ids, pe, cfg)
    if cfg.nl_loss_mode == "ce":
        return cross_entropy_loss_from_logits(logits, token_ids, cfg.epsilon)
    loss, _ = renyi_loss_from_logits(logits, token_ids, cfg.epsilon)
    return loss


def nlcsa_val_ce_sentence(params, token_ids, pe, cfg: BaselineConfig):
    """Always standard CE for cross-architecture comparison."""
    logits = nlcsa_logits_sentence(params, token_ids, pe, cfg)
    return cross_entropy_loss_from_logits(logits, token_ids, cfg.epsilon)


# --------------------------------------------------------------------------- #
#  I/O helpers
# --------------------------------------------------------------------------- #
def _run_dir(cfg: BaselineConfig, tag: str) -> Path:
    if not cfg.output_dir:
        raise ValueError("BaselineConfig.output_dir is required for save/resume")
    return Path(cfg.output_dir) / tag


def _is_run_complete(out: Path, expected_epochs: int) -> bool:
    metrics = out / "metrics.json"
    params = out / "params_final.npz"
    if not (metrics.exists() and params.exists()):
        return False
    try:
        data = json.loads(metrics.read_text(encoding="utf-8"))
        if data.get("stopped_early"):
            return True
        hist = data.get("loss_history", [])
        return len(hist) >= expected_epochs
    except Exception:
        return False


def _load_completed_result(out: Path) -> dict:
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


def _save_checkpoint(
    out: Path,
    params,
    losses: List[float],
    wall_times: List[float],
    epoch: int,
    rng: np.random.Generator,
    extra: Optional[dict] = None,
) -> None:
    ckpt_dir = out / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    flat = {}

    def _walk(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                _walk(v, f"{prefix}/{k}" if prefix else str(k))
        elif isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                _walk(v, f"{prefix}/{i}")
        else:
            flat[prefix.replace("/", "__")] = np.asarray(tree)

    _walk(params)
    np.savez_compressed(ckpt_dir / "params.npz", **flat)
    payload = {
        "epoch": epoch,
        "loss_history": losses,
        "wall_time_history": wall_times,
        "rng_state": rng.bit_generator.state,
        "extra": extra or {},
    }
    (ckpt_dir / "state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _try_load_checkpoint(out: Path, template_params):
    ckpt_dir = out / "checkpoint"
    state_path = ckpt_dir / "state.json"
    params_path = ckpt_dir / "params.npz"
    if not (state_path.exists() and params_path.exists()):
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))

    def _fill(tree, prefix=""):
        if isinstance(tree, dict):
            return {k: _fill(v, f"{prefix}/{k}" if prefix else str(k)) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            seq = [_fill(v, f"{prefix}/{i}") for i, v in enumerate(tree)]
            return type(tree)(seq)
        key = prefix.replace("/", "__")
        return jnp.asarray(np.load(params_path)[key])

    params = _fill(template_params)
    rng = np.random.default_rng()
    try:
        rng.bit_generator.state = state["rng_state"]
    except Exception:
        pass
    return {
        "params": params,
        "epoch": int(state["epoch"]),
        "loss_history": list(state.get("loss_history", [])),
        "wall_time_history": list(state.get("wall_time_history", [])),
        "rng": rng,
        "extra": state.get("extra", {}),
    }


def _relative_improvement(losses: List[float], window: int = 5) -> float:
    """Relative improvement of train loss over a trailing window.

    Definition (positive = loss still decreasing):
        old = mean(loss[t-window : t-1])   # previous `window` epochs
        new = loss[t]                      # current epoch
        rel = (old - new) / max(|old|, eps)

    Used as a plateau indicator (together with early-stopping on
    consecutive |Δloss|/|loss| < loss_rel_tol).  Not comparable across
    different loss families (−log μ vs CE vs Rényi).
    """
    if len(losses) < window + 1:
        return float("nan")
    old = float(np.mean(losses[-(window + 1) : -1]))
    new = float(losses[-1])
    return (old - new) / max(abs(old), 1e-12)


def _grad_norm(grads) -> float:
    leaves = jax.tree_util.tree_leaves(grads)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves)))


def _train_loop(
    name: str,
    cfg: BaselineConfig,
    bundle: dict,
    params,
    loss_fn,
    val_eval_fn,
    run_tag: str,
    log: logging.Logger,
    lr: Optional[float] = None,
    extra_ckpt: Optional[dict] = None,
) -> dict:
    out = _run_dir(cfg, run_tag)
    max_ep = int(cfg.max_epochs or cfg.epochs)

    if cfg.resume and _is_run_complete(out, max_ep):
        result = _load_completed_result(out)
        log.info(f"[{name}] resume: already complete -> {out}")
        return result

    token_batch = bundle["token_batch"]
    train_idx = bundle["train_idx"]
    val_idx = bundle["val_idx"]
    pe = bundle["pe"]
    train_batch = token_batch[train_idx]
    val_batch = token_batch[val_idx]
    n = int(train_batch.shape[0])
    bs = int(cfg.batch_size) if cfg.batch_size else n
    bs = max(1, min(bs, n))
    step_lr = lr if lr is not None else cfg.learning_rate
    rng = np.random.default_rng(cfg.model_seed + 17)

    def loss_batch(p, batch):
        return jnp.mean(jax.vmap(lambda ids: loss_fn(p, ids, pe, cfg))(batch))

    @jax.jit
    def step(p, batch):
        (loss, grads) = jax.value_and_grad(loss_batch)(p, batch)
        p = jax.tree_util.tree_map(lambda x, g: x - step_lr * g, p, grads)
        return p, loss, grads

    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
    losses: List[float] = []
    val_ce_history: List[float] = []
    val_ppl_history: List[float] = []
    grad_norm_history: List[float] = []
    rel_improve_history: List[float] = []
    wall_times: List[float] = []
    start_ep = 1
    elapsed0 = 0.0
    stagnant = 0
    stopped_early = False
    stop_reason = ""

    ckpt = _try_load_checkpoint(out, params) if cfg.resume else None
    if ckpt is not None:
        params = ckpt["params"]
        start_ep = int(ckpt["epoch"]) + 1
        losses = list(ckpt["loss_history"])
        wall_times = list(ckpt["wall_time_history"])
        rng = ckpt["rng"]
        extra = ckpt.get("extra") or {}
        val_ce_history = list(extra.get("val_ce_history", []))
        val_ppl_history = list(extra.get("val_ppl_history", []))
        grad_norm_history = list(extra.get("grad_norm_history", []))
        rel_improve_history = list(extra.get("rel_improve_history", []))
        elapsed0 = float(wall_times[-1]) if wall_times else 0.0
        log.info(f"[{name}] resume from epoch {ckpt['epoch']} -> {out}")

    t0 = time.perf_counter()
    for ep in range(start_ep, max_ep + 1):
        if bs < n:
            idx = rng.choice(n, size=bs, replace=False)
            batch = train_batch[idx]
        else:
            batch = train_batch
        params, loss, grads = step(params, batch)
        gn = _grad_norm(grads) if cfg.track_grad_norm else 0.0
        full_loss = float(loss_batch(params, train_batch))
        # val_eval_fn is model-specific: −log μ for mu-models, true CE for nl-CSA.
        # Do NOT treat exp(val) as a shared "perplexity" across architectures.
        val_metric = float(
            jnp.mean(jax.vmap(lambda ids: val_eval_fn(params, ids, pe, cfg))(val_batch))
        )
        losses.append(full_loss)
        val_ce_history.append(val_metric)  # name kept for resume compat; see val_metric_kind
        val_ppl_history.append(float(np.exp(val_metric)))
        if cfg.track_grad_norm:
            grad_norm_history.append(float(gn))
        rel_improve_history.append(_relative_improvement(losses, window=5))
        wall_times.append(elapsed0 + (time.perf_counter() - t0))

        if ep == 1 or ep % max(1, max_ep // 10) == 0 or ep == max_ep:
            log.info(
                f"[{name}] ep={ep:03d}/{max_ep} train_loss={full_loss:.6f} "
                f"val_metric={val_metric:.6f} exp(val)={val_ppl_history[-1]:.4f} "
                f"grad_norm={grad_norm_history[-1] if grad_norm_history else 0:.3e} "
                f"rel_impr5={rel_improve_history[-1]:.3e} t={wall_times[-1]:.1f}s"
            )

        ckpt_extra = {
            "val_ce_history": val_ce_history,
            "val_ppl_history": val_ppl_history,
            "grad_norm_history": grad_norm_history,
            "rel_improve_history": rel_improve_history,
            **(extra_ckpt or {}),
        }
        if cfg.checkpoint_every > 0 and (ep % cfg.checkpoint_every == 0 or ep == max_ep):
            _save_checkpoint(out, params, losses, wall_times, ep, rng, extra=ckpt_extra)

        if cfg.early_stop and ep > 1:
            rel_change = abs(losses[-2] - losses[-1]) / max(abs(losses[-2]), 1e-12)
            if rel_change < cfg.loss_rel_tol:
                stagnant += 1
            else:
                stagnant = 0
            if stagnant >= cfg.convergence_patience:
                stopped_early = True
                stop_reason = (
                    f"plateau: rel_change<{cfg.loss_rel_tol} "
                    f"for {cfg.convergence_patience} epochs"
                )
                log.info(f"[{name}] early stop at ep={ep}: {stop_reason}")
                break

    return {
        "params": params,
        "losses": losses,
        "val_ce_history": val_ce_history,
        "val_ppl_history": val_ppl_history,
        "grad_norm_history": grad_norm_history,
        "rel_improve_history": rel_improve_history,
        "wall_times": wall_times,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "epochs_run": len(losses),
        "out": out,
    }


def _finalize_mu_result(
    name: str,
    cfg: BaselineConfig,
    bundle: dict,
    train_out: dict,
    params,
    angle_n: int,
    matrix_n: int,
    note: str,
    run_name: str,
    loss_fn,
) -> dict:
    emb_n = int(np.asarray(params["embedding"]).size)
    result = {
        "model": name,
        "config": asdict(cfg),
        "data_seed": cfg.data_seed,
        "model_seed": cfg.model_seed,
        "seed": cfg.seed,
        "loss_history": train_out["losses"],
        "val_ce_history": train_out["val_ce_history"],
        "val_ppl_history": train_out["val_ppl_history"],
        "grad_norm_history": train_out["grad_norm_history"],
        "rel_improve_history": train_out["rel_improve_history"],
        "wall_time_history": train_out["wall_times"],
        "final_loss": train_out["losses"][-1],
        "final_val_ce": train_out["val_ce_history"][-1],
        "final_val_ppl": train_out["val_ppl_history"][-1],
        "train_metric": "L_B" if cfg.loss_objective == "L_B" else "neg_log_mu",
        "val_metric_kind": "L_B" if cfg.loss_objective == "L_B" else "neg_log_mu",
        "val_metric_common": None,
        "metric_warning": (
            "exp(val) = 1/μ is NOT language-model perplexity and must NOT be "
            "compared to nl-CSA next-token CE perplexity."
        ),
        "ppl_mu": float(np.exp(train_out["losses"][-1])),
        "n_params_angles": angle_n,
        "n_params_matrices": matrix_n,
        "n_params_embedding": emb_n,
        "n_params_total": angle_n + matrix_n + emb_n,
        "parametrization": cfg.parametrization.get(name, ""),
        "elapsed_seconds": train_out["wall_times"][-1] if train_out["wall_times"] else 0.0,
        "epochs_run": train_out["epochs_run"],
        "stopped_early": train_out["stopped_early"],
        "stop_reason": train_out["stop_reason"],
        "num_sentences": int(bundle["token_batch"].shape[0]),
        "num_train": int(bundle["train_idx"].size),
        "num_val": int(bundle["val_idx"].size),
        "batch_size": int(cfg.batch_size) if cfg.batch_size else int(bundle["train_idx"].size),
        "vocab_size": int(bundle["encoding"].vocabSize),
        "note": note,
    }
    train_batch = bundle["token_batch"][bundle["train_idx"]]
    lhalf_fn = _l_half_uniform_for_model(name)
    result["final_L_half_uniform"] = eval_batch_mean_loss(
        params, train_batch, bundle["pe"], cfg, lhalf_fn
    )
    if cfg.loss_objective == "L_B":
        result["final_L_B"] = float(result["final_loss"])
    if "test_token_batch" in bundle:
        test_batch = bundle["test_token_batch"]
        result["final_test_loss"] = eval_batch_mean_loss(
            params, test_batch, bundle["pe"], cfg, loss_fn
        )
        result["final_test_L_half_uniform"] = eval_batch_mean_loss(
            params, test_batch, bundle["pe"], cfg, lhalf_fn
        )
        if cfg.loss_objective == "L_B":
            result["final_test_L_B"] = float(result["final_test_loss"])
        result["num_test"] = int(bundle.get("num_test", test_batch.shape[0]))
        result["test_data_seed"] = int(cfg.test_data_seed) if cfg.test_data_seed is not None else None
    _save_run(result, params, bundle["sentences"], cfg, run_name)
    return result


def train_kqsa(cfg: BaselineConfig, bundle: dict, logger: Optional[logging.Logger] = None) -> dict:
    log = logger or logging.getLogger("baselines")
    run_name = cfg.run_label or f"kqsa_seed{cfg.model_seed}"
    out = _run_dir(cfg, run_name)
    max_ep = int(cfg.max_epochs or cfg.epochs)
    if cfg.resume and _is_run_complete(out, max_ep):
        result = _load_completed_result(out)
        log.info(f"[k-QSA] resume: already complete -> {out}")
        return result
    key = jax.random.PRNGKey(cfg.model_seed)
    params = init_qsa_params(bundle["encoding"].vocabSize, bundle["qcfg"], key)
    angle_n = qsa_angle_param_count(cfg.d, cfg.layers)
    loss_fn = _kqsa_loss_for_cfg(cfg)
    train_out = _train_loop(
        "k-QSA",
        cfg,
        bundle,
        params,
        loss_fn,
        loss_fn,
        run_name,
        log,
    )
    # Already-complete resume returns metrics.json (no "params" key).
    if "params" not in train_out:
        return train_out
    return _finalize_mu_result(
        "k-QSA",
        cfg,
        bundle,
        train_out,
        train_out["params"],
        angle_n=angle_n,
        matrix_n=0,
        note=(
            "Train loss = L_B (objective B, real ortho ansatz) on train split; "
            "val = same on val split. Quantum-angle ortho W,V; model_seed controls init."
            if cfg.loss_objective == "L_B"
            else (
                "Train loss = -log(mu) on train split; val_ce = same on val split. "
                "Quantum-angle ortho W,V; model_seed controls init. "
                "Local training does not execute the quantum circuit."
            )
        ),
        run_name=run_name,
        loss_fn=loss_fn,
    )
 
def train_kcsa(
    cfg: BaselineConfig, bundle: dict, logger: Optional[logging.Logger] = None
) -> dict:
    log = logger or logging.getLogger("baselines")
    run_name = cfg.run_label or f"kcsa_seed{cfg.model_seed}"
    out = _run_dir(cfg, run_name)
    max_ep = int(cfg.max_epochs or cfg.epochs)
    if cfg.resume and _is_run_complete(out, max_ep):
        result = _load_completed_result(out)
        log.info(f"[k-CSA] resume: already complete -> {out}")
        return result
    key = jax.random.PRNGKey(cfg.model_seed + 31)
    params = init_kcsa_params(bundle["encoding"].vocabSize, cfg, key)
    matrix_n = kcsa_matrix_param_count(cfg.d)
    loss_fn = _kcsa_loss_for_cfg(cfg)
    train_out = _train_loop(
        "k-CSA",
        cfg,
        bundle,
        params,
        loss_fn,
        loss_fn,
        run_name,
        log,
    )
    # Already-complete resume returns metrics.json (no "params" key).
    if "params" not in train_out:
        return train_out
    return _finalize_mu_result(
        "k-CSA",
        cfg,
        bundle,
        train_out,
        train_out["params"],
        angle_n=0,
        matrix_n=matrix_n,
        note=(
            "Independent classical baseline: QR-orthogonal W,V. "
            "Train loss = L_B on train split."
            if cfg.loss_objective == "L_B"
            else (
                "Independent classical baseline: QR-orthogonal W,V from learnable raw matrices "
                "(no quantum RY angles). Same data_seed / train conditions as k-QSA; "
                "separate model_seed. Train loss = -log(mu); val_ce = -log(mu) on val split."
            )
        ),
        run_name=run_name,
        loss_fn=loss_fn,
    )


def train_nlcsa(
    cfg: BaselineConfig,
    bundle: dict,
    logger: Optional[logging.Logger] = None,
) -> dict:
    log = logger or logging.getLogger("baselines")
    ablation_tag = f"{cfg.nl_embedding_mode}_{cfg.nl_loss_mode}"
    run_name = cfg.run_label or f"nlcsa_{ablation_tag}_seed{cfg.model_seed}"
    out = _run_dir(cfg, run_name)
    max_ep = int(cfg.max_epochs or cfg.epochs)

    if cfg.resume and _is_run_complete(out, max_ep):
        result = _load_completed_result(out)
        log.info(f"[nl-CSA] resume: already complete -> {out}")
        return result

    encoding = bundle["encoding"]
    key = jax.random.PRNGKey(cfg.model_seed + 13)
    params, r = init_nlcsa_params(encoding.vocabSize, cfg, key)
    lr = cfg.nl_learning_rate if cfg.nl_learning_rate is not None else max(cfg.learning_rate, 5e-3)

    train_out = _train_loop(
        f"nl-CSA({ablation_tag})",
        cfg,
        bundle,
        params,
        nlcsa_loss_sentence,
        nlcsa_val_ce_sentence,
        run_name,
        log,
        lr=lr,
        extra_ckpt={"nl_rank": r, "ablation": ablation_tag},
    )
    params = train_out["params"]
    n_total = count_tree_params(params)
    emb_n = int(np.asarray(params["embedding"]).size)
    n_model = n_total - emb_n
    result = {
        "model": "nl-CSA",
        "ablation": ablation_tag,
        "nl_embedding_mode": cfg.nl_embedding_mode,
        "nl_loss_mode": cfg.nl_loss_mode,
        "config": asdict(cfg),
        "data_seed": cfg.data_seed,
        "model_seed": cfg.model_seed,
        "seed": cfg.seed,
        "nl_rank": r,
        "nl_learning_rate": lr,
        "loss_history": train_out["losses"],
        "val_ce_history": train_out["val_ce_history"],
        "val_ppl_history": train_out["val_ppl_history"],
        "grad_norm_history": train_out["grad_norm_history"],
        "rel_improve_history": train_out["rel_improve_history"],
        "wall_time_history": train_out["wall_times"],
        "final_loss": train_out["losses"][-1],
        "final_val_ce": train_out["val_ce_history"][-1],
        "final_val_ppl": train_out["val_ppl_history"][-1],
        "train_metric": cfg.nl_loss_mode,
        "val_metric_kind": "next_token_cross_entropy",
        "val_metric_common": "cross_entropy",
        "metric_warning": (
            "final_val_ppl = exp(CE) is true next-token LM perplexity "
            f"(vocab≈{encoding.vocabSize}; uniform baseline ≈ vocab). "
            "Not comparable to k-QSA/k-CSA exp(−log μ)=1/μ."
        ),
        "n_params_model": n_model,
        "n_params_embedding": emb_n,
        "n_params_total": n_total,
        "parametrization": cfg.parametrization["nl-CSA"],
        "architecture": (
            "embedding → causal softmax attention → residual/normalization → "
            "FFN GELU → output logits (tied to embedding)"
        ),
        "k_independent": True,
        "elapsed_seconds": train_out["wall_times"][-1] if train_out["wall_times"] else 0.0,
        "epochs_run": train_out["epochs_run"],
        "stopped_early": train_out["stopped_early"],
        "stop_reason": train_out["stop_reason"],
        "num_sentences": int(bundle["token_batch"].shape[0]),
        "num_train": int(bundle["train_idx"].size),
        "num_val": int(bundle["val_idx"].size),
        "batch_size": int(cfg.batch_size) if cfg.batch_size else int(bundle["train_idx"].size),
        "vocab_size": int(encoding.vocabSize),
        "note": (
            f"Train loss = {cfg.nl_loss_mode}; common val metric = CE/perplexity. "
            "Architecture does not depend on k."
        ),
    }
    if "test_token_batch" in bundle:
        test_batch = bundle["test_token_batch"]
        result["final_test_loss"] = eval_batch_mean_loss(
            params, test_batch, bundle["pe"], cfg, nlcsa_loss_sentence
        )
        result["num_test"] = int(bundle.get("num_test", test_batch.shape[0]))
        result["test_data_seed"] = int(cfg.test_data_seed) if cfg.test_data_seed is not None else None
    _save_run(result, params, bundle["sentences"], cfg, run_name)
    return result


def _save_run(result: dict, params, sentences, cfg: BaselineConfig, tag: str) -> None:
    if not cfg.output_dir:
        return
    out = Path(cfg.output_dir) / tag
    params_dir = out / "params"
    out.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    losses = result.get("loss_history", [])
    times = result.get("wall_time_history", [])
    (out / "loss_history.txt").write_text(
        "\n".join(f"{v:.8f}" for v in losses), encoding="utf-8"
    )
    if times:
        (out / "wall_time_history.txt").write_text(
            "\n".join(f"{v:.6f}" for v in times), encoding="utf-8"
        )
        lines = ["epoch,wall_time_s,train_loss,val_ce,val_ppl,grad_norm,rel_improve_5"]
        for i, t in enumerate(times, start=1):
            val_ce = result.get("val_ce_history", [float("nan")] * len(times))[i - 1]
            val_ppl = result.get("val_ppl_history", [float("nan")] * len(times))[i - 1]
            gn = result.get("grad_norm_history", [float("nan")] * len(times))[i - 1]
            ri = result.get("rel_improve_history", [float("nan")] * len(times))[i - 1]
            lines.append(
                f"{i},{t:.6f},{losses[i-1]:.8f},{val_ce:.8f},{val_ppl:.6f},{gn:.6e},{ri:.6e}"
            )
        (out / "diagnostics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    flat = {}

    def _store(path: str, x):
        arr = np.asarray(x)
        flat[path] = arr
        safe = path.replace("/", "__")
        np.save(params_dir / f"{safe}.npy", arr)

    def _walk(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                _walk(v, f"{prefix}/{k}" if prefix else str(k))
        elif isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                _walk(v, f"{prefix}/{i}")
        else:
            _store(prefix, tree)

    _walk(params)
    np.savez_compressed(out / "params_final.npz", **{k.replace("/", "__"): v for k, v in flat.items()})

    meta = {
        "timestamp": datetime.now().isoformat(),
        "sentences": sentences,
        "tag": tag,
        "param_files": sorted(p.name for p in params_dir.glob("*.npy")),
    }
    np.save(out / "meta.npy", meta, allow_pickle=True)


def aggregate_seed_results(runs: List[dict]) -> dict:
    assert runs, "empty runs"
    name = runs[0]["model"]
    hist = np.array([r["loss_history"] for r in runs], dtype=float)
    mean = hist.mean(axis=0)
    std = hist.std(axis=0)
    out = {
        "model": name,
        "seeds": [int(r.get("model_seed", r.get("seed", -1))) for r in runs],
        "n_seeds": len(runs),
        "loss_history": mean.tolist(),
        "loss_std": std.tolist(),
        "loss_min": hist.min(axis=0).tolist(),
        "loss_max": hist.max(axis=0).tolist(),
        "seed_histories": hist.tolist(),
        "final_loss_mean": float(mean[-1]),
        "final_loss_std": float(std[-1]),
        "n_params_angles": runs[0].get("n_params_angles"),
        "n_params_matrices": runs[0].get("n_params_matrices"),
        "n_params_model": runs[0].get("n_params_model"),
        "nl_rank": runs[0].get("nl_rank"),
        "ablation": runs[0].get("ablation"),
        "batch_size": runs[0].get("batch_size"),
    }
    if all("val_ppl_history" in r for r in runs):
        vp = np.array([r["val_ppl_history"] for r in runs], dtype=float)
        out["val_ppl_history"] = vp.mean(axis=0).tolist()
        out["val_ppl_std"] = vp.std(axis=0).tolist()
        out["final_val_ppl_mean"] = float(vp[:, -1].mean())
        out["final_val_ppl_std"] = float(vp[:, -1].std())
    if all("wall_time_history" in r for r in runs):
        times = np.array([r["wall_time_history"] for r in runs], dtype=float)
        out["wall_time_history"] = times.mean(axis=0).tolist()
        out["wall_time_std"] = times.std(axis=0).tolist()
        out["seed_wall_times"] = times.tolist()
    return out


def plot_training_curves(
    results: List[dict],
    out_path: Path,
    show_seed_traces: bool = True,
) -> None:
    """Panels: train loss (model-specific) and common val perplexity."""
    import matplotlib.pyplot as plt

    mu_res = [r for r in results if r["model"] in ("k-QSA", "k-CSA")]
    nl_res = [r for r in results if str(r["model"]).startswith("nl-CSA")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def _plot_panel(ax, series, x_mode, ykey, ylabel, title, std_key=None):
        for i, res in enumerate(series):
            ys = np.asarray(res.get(ykey, res["loss_history"]), dtype=float)
            color = f"C{i}"
            n_seeds = int(res.get("n_seeds", 1))
            label = res["model"]
            if res.get("ablation"):
                label += f" [{res['ablation']}]"
            if x_mode == "epoch":
                xs = np.arange(1, len(ys) + 1)
                if show_seed_traces and ykey == "loss_history" and "seed_histories" in res:
                    for h in res["seed_histories"]:
                        ax.plot(np.arange(1, len(h) + 1), h, color=color, alpha=0.25, linewidth=1)
            else:
                if "wall_time_history" not in res:
                    continue
                xs = np.asarray(res["wall_time_history"], dtype=float)
            ax.plot(xs, ys, color=color, linewidth=2.2, label=label)
            sk = std_key or (ykey.replace("history", "_std") if ykey.endswith("_history") else None)
            if sk and sk in res and n_seeds > 1:
                std = np.asarray(res[sk], dtype=float)
                ax.fill_between(xs, ys - std, ys + std, color=color, alpha=0.2, linewidth=0)
        ax.set_xlabel("epoch" if x_mode == "epoch" else "wall time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    _plot_panel(axes[0, 0], mu_res, "epoch", "loss_history", r"train $-\log\mu$", "mu-models train loss")
    _plot_panel(axes[0, 1], mu_res, "epoch", "val_ppl_history", "val perplexity (common)", "mu-models val ppl")
    _plot_panel(axes[1, 0], nl_res, "epoch", "loss_history", "train loss (CE or Renyi)", "nl-CSA train loss")
    _plot_panel(axes[1, 1], nl_res, "epoch", "val_ppl_history", "val CE perplexity", "nl-CSA val ppl", "val_ppl_std")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_convergence_diagnostics(results: List[dict], out_path: Path) -> None:
    """Gradient norm and relative improvement for convergence checks."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i, res in enumerate(results):
        color = f"C{i}"
        label = res["model"]
        if res.get("ablation"):
            label += f" [{res['ablation']}]"
        ep = np.arange(1, len(res.get("loss_history", [])) + 1)
        if res.get("grad_norm_history"):
            axes[0].plot(ep, res["grad_norm_history"], color=color, label=label)
        if res.get("rel_improve_history"):
            axes[1].plot(ep, res["rel_improve_history"], color=color, label=label)
    axes[0].set_ylabel("grad norm")
    axes[0].set_yscale("log")
    axes[0].set_title("Gradient norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("rel. improvement (5-epoch window)")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Loss plateau indicator")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_final_loss_vs_k(
    points: List[dict],
    out_path: Path,
    title: str = "Train loss vs k",
    ykey: str = "final_loss_mean",
    ylabel: str = r"train loss ($-\log\mu$ for k-QSA/k-CSA)",
    nl_ref: Optional[dict] = None,
    nl_ref_ykey: str = "final_loss",
    nl_ref_label: str = "nl-CSA isometric+Rényi (k-indep.; different loss)",
    annotate_params: bool = True,
) -> None:
    """
    Line plot with error bars for mu-baselines across k.

    nl-CSA (k-independent) can be shown as a horizontal reference.  If that
    reference is a Rényi/CE value, the caption must state it is NOT the same
    metric as −log μ (different absolute scale; trends only if declared).
    """
    import matplotlib.pyplot as plt

    by_model: dict[str, list[dict]] = {}
    for p in points:
        by_model.setdefault(str(p["model"]), []).append(p)

    styles = {
        "k-QSA": dict(color="#0072B2", marker="o", linestyle="-"),
        "k-CSA": dict(color="#D55E00", marker="s", linestyle="--"),
        "nl-CSA": dict(color="#009E73", marker="D", linestyle="-."),
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model, rows in by_model.items():
        rows = sorted(rows, key=lambda r: int(r["k"]))
        ks = [int(r["k"]) for r in rows]
        means = [float(r.get(ykey, r.get("final_loss", r.get("final_loss_mean", 0)))) for r in rows]
        std_key = ykey.replace("_mean", "_std") if ykey.endswith("_mean") else "final_loss_std"
        stds = [float(r.get(std_key, r.get("final_loss_std", 0.0))) for r in rows]
        st = styles.get(model, dict(color="0.3", marker="o", linestyle="-"))
        n_params = rows[0].get("n_params_angles") or rows[0].get("n_params_matrices")
        label = model
        if annotate_params and n_params is not None:
            kind = "angles" if rows[0].get("n_params_angles") else "matrices"
            label = f"{model} ({kind}={int(n_params)})"
        ax.errorbar(
            ks, means, yerr=stds, color=st["color"], marker=st["marker"],
            linestyle=st["linestyle"], linewidth=2.2, markersize=8, capsize=4, label=label,
        )

    if nl_ref is not None:
        mean = float(nl_ref.get(nl_ref_ykey, nl_ref.get("final_loss", 0)))
        std = float(nl_ref.get("final_loss_std", nl_ref.get("final_val_ppl_std", 0)) or 0)
        n_nl = nl_ref.get("n_params_model") or nl_ref.get("n_params_total")
        lbl = nl_ref_label
        if annotate_params and n_nl is not None:
            lbl = f"{nl_ref_label} (params≈{int(n_nl)})"
        ax.axhline(mean, color="#009E73", linestyle="-.", linewidth=2, label=lbl)
        if std > 0:
            ax.axhspan(mean - std, mean + std, color="#009E73", alpha=0.15)

    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if points:
        ax.set_xticks(sorted({int(p["k"]) for p in points}))
    ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


NL_ABLATIONS = (
    ("isometric", "ce"),
    ("isometric", "renyi"),
    ("general", "ce"),
    ("general", "renyi"),
)
