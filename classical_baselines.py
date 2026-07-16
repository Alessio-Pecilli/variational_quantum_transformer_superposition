"""
Classical baselines for comparison with k-QSA (Section 2).

- k-CSA: same polynomial kernel objective sum_{i<=j} a_ij s_ij^k with W,V
  parametrized by real_ortho_block (O(L log d) angles), trained classically.
- nl-CSA: softmax self-attention + small FFN with a matched trainable budget
  (low-rank projections), Renyi-1/2 next-token loss, isometric embedding.

Shared data path with qsa_training (Encoding + PTB + PE).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from math import log2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from config import DATASET_CONFIG, get_training_sentences
from encoding import Encoding
from pennylane_jax_vqt import isometrize_matrix
from qsa_training import (
    QSATrainConfig,
    _feature_qubits,
    _normalize_rows,
    classical_mu_jax,
    init_qsa_params,
    ortho_matrix_jax,
    sequence_xy_from_tokens,
)

jax.config.update("jax_enable_x64", True)


def count_tree_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.asarray(x).size for x in leaves))


def qsa_angle_param_count(d: int, layers: int) -> int:
    """Trainable angles for W and V (embedding counted separately)."""
    n = _feature_qubits(d)
    return 2 * layers * n


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
    train_embedding: bool = True
    epsilon: float = 1e-12
    run_label: Optional[str] = None
    output_dir: Optional[str] = None
    # nl-CSA low-rank width (chosen to keep attn+ffn near O(L log d))
    nl_rank: Optional[int] = None


def baseline_config_from_qsa(cfg: QSATrainConfig, nl_rank: Optional[int] = None) -> BaselineConfig:
    return BaselineConfig(
        T=cfg.T,
        d=cfg.d,
        k=cfg.k,
        layers=cfg.layers,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        max_sentences=cfg.max_sentences,
        seed=cfg.seed,
        train_embedding=cfg.train_embedding,
        epsilon=cfg.epsilon,
        run_label=cfg.run_label,
        output_dir=cfg.output_dir,
        nl_rank=nl_rank,
    )


def _prepare_dataset(cfg: BaselineConfig):
    sentences = get_training_sentences()
    if len(sentences) < cfg.max_sentences:
        expanded = list(sentences)
        while len(expanded) < cfg.max_sentences:
            expanded.extend(sentences)
        sentences = expanded[: cfg.max_sentences]
    encoding = Encoding(sentences, embeddingDim=cfg.d, embeddingSeed=cfg.seed)
    token_batch = jnp.stack([encoding.encode_tokens(s) for s in sentences])
    pe = encoding._positionalEncoding(cfg.T)
    return encoding, sentences, token_batch, pe


# --------------------------------------------------------------------------- #
#  k-CSA: classical polynomial-kernel model (same mu loss as k-QSA train path)
# --------------------------------------------------------------------------- #
def kcsa_loss_sentence(params, token_ids, pe, cfg: BaselineConfig):
    emb_raw = params["embedding"]
    if not cfg.train_embedding:
        emb_raw = jax.lax.stop_gradient(emb_raw)
    emb = isometrize_matrix(emb_raw)
    X, Y = sequence_xy_from_tokens(token_ids, emb, pe)
    W = ortho_matrix_jax(params["weights_w"], cfg.d)
    V = ortho_matrix_jax(params["weights_v"], cfg.d)
    mu = classical_mu_jax(X, Y, W, V, cfg.k)
    return -jnp.log(jnp.maximum(mu, cfg.epsilon))


def train_kcsa(cfg: BaselineConfig, logger: Optional[logging.Logger] = None) -> dict:
    """k-CSA: classical sum a s^k with O(L log d) ortho angles."""
    log = logger or logging.getLogger("baselines")
    encoding, sentences, token_batch, pe = _prepare_dataset(cfg)
    key = jax.random.PRNGKey(cfg.seed + 7)
    qcfg = QSATrainConfig(
        T=cfg.T,
        d=cfg.d,
        k=cfg.k,
        layers=cfg.layers,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        max_sentences=cfg.max_sentences,
        seed=cfg.seed + 7,
        train_embedding=cfg.train_embedding,
        epsilon=cfg.epsilon,
    )
    params = init_qsa_params(encoding.vocabSize, qcfg, key)

    def loss_batch(p, batch):
        return jnp.mean(jax.vmap(lambda ids: kcsa_loss_sentence(p, ids, pe, cfg))(batch))

    @jax.jit
    def step(p, batch):
        loss, grads = jax.value_and_grad(loss_batch)(p, batch)
        p = jax.tree_util.tree_map(lambda x, g: x - cfg.learning_rate * g, p, grads)
        return p, loss

    losses: List[float] = []
    t0 = time.perf_counter()
    for ep in range(1, cfg.epochs + 1):
        params, loss = step(params, token_batch)
        losses.append(float(loss))
        if ep == 1 or ep % max(1, cfg.epochs // 10) == 0 or ep == cfg.epochs:
            log.info(f"[k-CSA] epoch={ep:03d}/{cfg.epochs} loss={float(loss):.6f}")

    angle_n = qsa_angle_param_count(cfg.d, cfg.layers)
    emb_n = int(np.asarray(params["embedding"]).size)
    result = {
        "model": "k-CSA",
        "config": asdict(cfg),
        "loss_history": losses,
        "final_loss": losses[-1],
        "ppl_mu": float(np.exp(losses[-1])),
        "n_params_angles": angle_n,
        "n_params_embedding": emb_n,
        "n_params_total": angle_n + emb_n,
        "elapsed_seconds": time.perf_counter() - t0,
        "num_sentences": int(token_batch.shape[0]),
        "vocab_size": int(encoding.vocabSize),
    }
    _save_run(result, params, sentences, cfg, "kcsa")
    return result


# --------------------------------------------------------------------------- #
#  nl-CSA: softmax + FFN, Renyi next-token loss, matched-ish param budget
# --------------------------------------------------------------------------- #
def _default_nl_rank(d: int, layers: int) -> int:
    """Choose rank so attn+ffn params ~ O(L log d)."""
    target = max(4, qsa_angle_param_count(d, layers))
    # per layer: 3*(d*r) + (d*r) + (r*d)  ~ 5 d r  => r ~ target / (5 d L)
    r = max(1, int(round(target / (5.0 * d * max(layers, 1)))))
    return min(r, d)


def init_nlcsa_params(vocab_size: int, cfg: BaselineConfig, key: jax.Array) -> Tuple[dict, int]:
    r = cfg.nl_rank if cfg.nl_rank is not None else _default_nl_rank(cfg.d, cfg.layers)
    keys = jax.random.split(key, 1 + 5 * cfg.layers)
    emb = isometrize_matrix(jax.random.normal(keys[0], (vocab_size, cfg.d), dtype=jnp.float64))
    layers = []
    ki = 1
    for _ in range(cfg.layers):
        layers.append(
            {
                "Wq": jax.random.normal(keys[ki], (cfg.d, r), dtype=jnp.float64) * 0.1,
                "Wk": jax.random.normal(keys[ki + 1], (cfg.d, r), dtype=jnp.float64) * 0.1,
                "Wv": jax.random.normal(keys[ki + 2], (cfg.d, r), dtype=jnp.float64) * 0.1,
                "W1": jax.random.normal(keys[ki + 3], (cfg.d, r), dtype=jnp.float64) * 0.1,
                "W2": jax.random.normal(keys[ki + 4], (r, cfg.d), dtype=jnp.float64) * 0.1,
            }
        )
        ki += 5
    return {"embedding": emb, "layers": layers}, r


def _causal_softmax_attn(X, Wq, Wk, Wv, eps=1e-12):
    # X: (T, d)
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = Q @ K.T / jnp.sqrt(jnp.maximum(Q.shape[-1], 1.0))
    T = X.shape[0]
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.float64))
    scores = jnp.where(mask > 0, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ V


def nlcsa_logits_sentence(params, token_ids, pe, cfg: BaselineConfig):
    emb_raw = params["embedding"]
    if not cfg.train_embedding:
        emb_raw = jax.lax.stop_gradient(emb_raw)
    emb = isometrize_matrix(emb_raw)
    X = _normalize_rows(emb[token_ids] + pe)
    h = X
    for layer in params["layers"]:
        attn = _causal_softmax_attn(h, layer["Wq"], layer["Wk"], layer["Wv"])
        h = _normalize_rows(h + attn)
        ff = jax.nn.gelu(h @ layer["W1"]) @ layer["W2"]
        h = _normalize_rows(h + ff)
    # next-token logits via isometric embedding (tied)
    return h @ emb.T  # (T, vocab)


def renyi_loss_from_logits(logits, token_ids, eps=1e-12):
    """Renyi-1/2 style: targets are next tokens (shift), last position dropped."""
    # predict token_ids[t+1] from position t
    targets = token_ids[1:]
    logits_t = logits[:-1]
    probs = jax.nn.softmax(logits_t, axis=-1)
    p_true = probs[jnp.arange(targets.shape[0]), targets]
    f_bar = jnp.mean(jnp.sqrt(jnp.maximum(p_true, eps)))
    ppl = jnp.maximum(f_bar, eps) ** (-2)
    return jnp.log(jnp.maximum(ppl, eps)), ppl


def nlcsa_loss_sentence(params, token_ids, pe, cfg: BaselineConfig):
    logits = nlcsa_logits_sentence(params, token_ids, pe, cfg)
    loss, _ = renyi_loss_from_logits(logits, token_ids, cfg.epsilon)
    return loss


def train_nlcsa(cfg: BaselineConfig, logger: Optional[logging.Logger] = None) -> dict:
    log = logger or logging.getLogger("baselines")
    encoding, sentences, token_batch, pe = _prepare_dataset(cfg)
    key = jax.random.PRNGKey(cfg.seed + 13)
    params, r = init_nlcsa_params(encoding.vocabSize, cfg, key)

    def loss_batch(p, batch):
        return jnp.mean(jax.vmap(lambda ids: nlcsa_loss_sentence(p, ids, pe, cfg))(batch))

    @jax.jit
    def step(p, batch):
        loss, grads = jax.value_and_grad(loss_batch)(p, batch)
        p = jax.tree_util.tree_map(lambda x, g: x - cfg.learning_rate * g, p, grads)
        return p, loss

    losses: List[float] = []
    ppls: List[float] = []
    t0 = time.perf_counter()
    for ep in range(1, cfg.epochs + 1):
        params, loss = step(params, token_batch)
        losses.append(float(loss))
        ppls.append(float(np.exp(float(loss))))  # Renyi PPL ~ exp(loss) under this def
        if ep == 1 or ep % max(1, cfg.epochs // 10) == 0 or ep == cfg.epochs:
            log.info(f"[nl-CSA] epoch={ep:03d}/{cfg.epochs} loss={float(loss):.6f} ppl~{ppls[-1]:.4f}")

    n_total = count_tree_params(params)
    emb_n = int(np.asarray(params["embedding"]).size)
    n_model = n_total - emb_n
    result = {
        "model": "nl-CSA",
        "config": asdict(cfg),
        "nl_rank": r,
        "loss_history": losses,
        "ppl_history": ppls,
        "final_loss": losses[-1],
        "final_ppl": ppls[-1],
        "n_params_model": n_model,
        "n_params_embedding": emb_n,
        "n_params_total": n_total,
        "target_angle_budget": qsa_angle_param_count(cfg.d, cfg.layers),
        "elapsed_seconds": time.perf_counter() - t0,
        "num_sentences": int(token_batch.shape[0]),
        "vocab_size": int(encoding.vocabSize),
    }
    _save_run(result, params, sentences, cfg, "nlcsa")
    return result


# --------------------------------------------------------------------------- #
#  k-QSA thin wrapper (reuse qsa_training) for side-by-side curves
# --------------------------------------------------------------------------- #
def train_kqsa(cfg: BaselineConfig, logger: Optional[logging.Logger] = None) -> dict:
    from qsa_training import train_qsa

    log = logger or logging.getLogger("baselines")
    qcfg = QSATrainConfig(
        T=cfg.T,
        d=cfg.d,
        k=cfg.k,
        layers=cfg.layers,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        max_sentences=cfg.max_sentences,
        seed=cfg.seed,
        train_embedding=cfg.train_embedding,
        epsilon=cfg.epsilon,
        run_label=(cfg.run_label or "kqsa") + "_qsa",
        output_dir=str(Path(cfg.output_dir) / "kqsa") if cfg.output_dir else None,
        local_max_qubits=32,
    )
    DATASET_CONFIG["sentence_length"] = cfg.T
    DATASET_CONFIG["max_sentences"] = cfg.max_sentences
    out = train_qsa(qcfg, output_dir=Path(qcfg.output_dir) if qcfg.output_dir else None, logger=log)
    angle_n = qsa_angle_param_count(cfg.d, cfg.layers)
    result = {
        "model": "k-QSA",
        "config": asdict(cfg),
        "loss_history": list(out["loss_history"]),
        "final_loss": float(out["final_loss"]),
        "ppl_mu": float(np.exp(float(out["final_loss"]))),
        "mean_O_ij": float(out["mean_O_ij"]),
        "obar": float(out["obar"]),
        "n_params_angles": angle_n,
        "n_params_total_angles_only": angle_n,
        "elapsed_seconds": float(out["elapsed_seconds"]),
        "num_sentences": int(out["num_sentences"]),
    }
    return result


def _save_run(result: dict, params, sentences, cfg: BaselineConfig, tag: str) -> None:
    if not cfg.output_dir:
        return
    out = Path(cfg.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "loss_history.txt").write_text(
        "\n".join(f"{v:.8f}" for v in result["loss_history"]), encoding="utf-8"
    )
    meta = {"timestamp": datetime.now().isoformat(), "sentences": sentences, "tag": tag}
    np.save(out / "meta.npy", meta, allow_pickle=True)


def plot_training_curves(results: List[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results:
        ys = res["loss_history"]
        xs = np.arange(1, len(ys) + 1)
        if res["model"] == "nl-CSA":
            label = f"{res['model']} (Renyi; model_params={res.get('n_params_model')})"
        else:
            label = f"{res['model']} (-log mu; angles={res.get('n_params_angles')})"
        ax.plot(xs, ys, linewidth=2, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Training curves: k-QSA vs k-CSA vs nl-CSA")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
