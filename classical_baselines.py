"""
Classical baselines for comparison with k-QSA (Section 2).

Important (referee-facing):
- Local k-QSA training already optimizes the classical mu = |sum a s^k|^2 path.
  Therefore k-QSA and k-CSA MUST share data + init and produce matching curves.
  The quantum/classical distinction is how mu is *evaluated* (circuit vs classical sum),
  not a different training objective in this setup.
- nl-CSA: softmax + FFN, Renyi next-token loss, isometric embedding.
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
    nl_rank: Optional[int] = None
    nl_learning_rate: Optional[float] = None
    nl_logit_scale: float = 8.0


def _load_sentences(cfg: BaselineConfig) -> List[str]:
    """Seeded sentence sample so all models see the exact same data."""
    rng = random.Random(cfg.seed)
    local = "ptb_sentences.txt"
    # Load all valid, then seeded sample (avoid unseeded random.sample in config.py)
    with open(local, "r", encoding="utf-8") as f:
        valid = [line.strip() for line in f if line.strip() and len(line.split()) == cfg.T]
    if len(valid) <= cfg.max_sentences:
        sentences = valid
    else:
        sentences = rng.sample(valid, cfg.max_sentences)
    if len(sentences) < cfg.max_sentences:
        expanded = list(sentences)
        while len(expanded) < cfg.max_sentences:
            expanded.extend(sentences)
        sentences = expanded[: cfg.max_sentences]
    return sentences


def prepare_shared_bundle(cfg: BaselineConfig):
    """One dataset + one QSA/CSA param init shared by k-QSA and k-CSA."""
    DATASET_CONFIG["sentence_length"] = cfg.T
    DATASET_CONFIG["max_sentences"] = cfg.max_sentences
    sentences = _load_sentences(cfg)
    encoding = Encoding(sentences, embeddingDim=cfg.d, embeddingSeed=cfg.seed)
    token_batch = jnp.stack([encoding.encode_tokens(s) for s in sentences])
    pe = encoding._positionalEncoding(cfg.T)
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
        local_max_qubits=32,
    )
    key = jax.random.PRNGKey(cfg.seed)
    params0 = init_qsa_params(encoding.vocabSize, qcfg, key)
    return {
        "sentences": sentences,
        "encoding": encoding,
        "token_batch": token_batch,
        "pe": pe,
        "qcfg": qcfg,
        "params0": params0,
    }


def mu_loss_sentence(params, token_ids, pe, cfg: BaselineConfig):
    emb_raw = params["embedding"]
    if not cfg.train_embedding:
        emb_raw = jax.lax.stop_gradient(emb_raw)
    emb = isometrize_matrix(emb_raw)
    X, Y = sequence_xy_from_tokens(token_ids, emb, pe)
    W = ortho_matrix_jax(params["weights_w"], cfg.d)
    V = ortho_matrix_jax(params["weights_v"], cfg.d)
    mu = classical_mu_jax(X, Y, W, V, cfg.k)
    return -jnp.log(jnp.maximum(mu, cfg.epsilon))


def _train_mu_model(
    name: str,
    cfg: BaselineConfig,
    bundle: dict,
    params,
    log: logging.Logger,
) -> dict:
    token_batch, pe = bundle["token_batch"], bundle["pe"]

    def loss_batch(p, batch):
        return jnp.mean(jax.vmap(lambda ids: mu_loss_sentence(p, ids, pe, cfg))(batch))

    @jax.jit
    def step(p, batch):
        loss, grads = jax.value_and_grad(loss_batch)(p, batch)
        p = jax.tree_util.tree_map(lambda x, g: x - cfg.learning_rate * g, p, grads)
        return p, loss

    # clone params
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
    losses: List[float] = []
    t0 = time.perf_counter()
    for ep in range(1, cfg.epochs + 1):
        params, loss = step(params, token_batch)
        losses.append(float(loss))
        if ep == 1 or ep % max(1, cfg.epochs // 10) == 0 or ep == cfg.epochs:
            log.info(f"[{name}] epoch={ep:03d}/{cfg.epochs} loss={float(loss):.6f}")

    angle_n = qsa_angle_param_count(cfg.d, cfg.layers)
    emb_n = int(np.asarray(params["embedding"]).size)
    result = {
        "model": name,
        "config": asdict(cfg),
        "loss_history": losses,
        "final_loss": losses[-1],
        "ppl_mu": float(np.exp(losses[-1])),
        "n_params_angles": angle_n,
        "n_params_embedding": emb_n,
        "n_params_total": angle_n + emb_n,
        "elapsed_seconds": time.perf_counter() - t0,
        "num_sentences": int(token_batch.shape[0]),
        "vocab_size": int(bundle["encoding"].vocabSize),
        "note": (
            "Same classical -log(mu) objective and shared init/data as the other mu-model. "
            "Local k-QSA training does not use the quantum circuit."
        ),
    }
    tag = "kqsa" if name == "k-QSA" else "kcsa"
    _save_run(result, params, bundle["sentences"], cfg, tag)
    return result


def train_kqsa(cfg: BaselineConfig, bundle: dict, logger: Optional[logging.Logger] = None) -> dict:
    log = logger or logging.getLogger("baselines")
    return _train_mu_model("k-QSA", cfg, bundle, bundle["params0"], log)


def train_kcsa(cfg: BaselineConfig, bundle: dict, logger: Optional[logging.Logger] = None) -> dict:
    log = logger or logging.getLogger("baselines")
    return _train_mu_model("k-CSA", cfg, bundle, bundle["params0"], log)


# --------------------------------------------------------------------------- #
#  nl-CSA
# --------------------------------------------------------------------------- #
def _default_nl_rank(d: int, layers: int) -> int:
    """Pick rank with usable capacity; report params honestly vs angle budget."""
    # r=1 kills Q/K grads; use at least 2, prefer ~sqrt(d)
    return int(min(d, max(2, int(round(d ** 0.5)))))


def init_nlcsa_params(vocab_size: int, cfg: BaselineConfig, key: jax.Array) -> Tuple[dict, int]:
    r = cfg.nl_rank if cfg.nl_rank is not None else _default_nl_rank(cfg.d, cfg.layers)
    keys = jax.random.split(key, 1 + 6 * cfg.layers)
    emb = isometrize_matrix(jax.random.normal(keys[0], (vocab_size, cfg.d), dtype=jnp.float64))
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
    emb_raw = params["embedding"]
    if not cfg.train_embedding:
        emb_raw = jax.lax.stop_gradient(emb_raw)
    emb = isometrize_matrix(emb_raw)
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
    loss, _ = renyi_loss_from_logits(logits, token_ids, cfg.epsilon)
    return loss


def train_nlcsa(
    cfg: BaselineConfig,
    bundle: dict,
    logger: Optional[logging.Logger] = None,
) -> dict:
    log = logger or logging.getLogger("baselines")
    token_batch, pe = bundle["token_batch"], bundle["pe"]
    encoding = bundle["encoding"]
    key = jax.random.PRNGKey(cfg.seed + 13)
    params, r = init_nlcsa_params(encoding.vocabSize, cfg, key)
    lr = cfg.nl_learning_rate if cfg.nl_learning_rate is not None else max(cfg.learning_rate, 5e-3)

    def loss_batch(p, batch):
        return jnp.mean(jax.vmap(lambda ids: nlcsa_loss_sentence(p, ids, pe, cfg))(batch))

    @jax.jit
    def step(p, batch):
        loss, grads = jax.value_and_grad(loss_batch)(p, batch)
        p = jax.tree_util.tree_map(lambda x, g: x - lr * g, p, grads)
        return p, loss

    losses: List[float] = []
    ppls: List[float] = []
    t0 = time.perf_counter()
    for ep in range(1, cfg.epochs + 1):
        params, loss = step(params, token_batch)
        losses.append(float(loss))
        ppls.append(float(np.exp(float(loss))))
        if ep == 1 or ep % max(1, cfg.epochs // 10) == 0 or ep == cfg.epochs:
            log.info(f"[nl-CSA] epoch={ep:03d}/{cfg.epochs} loss={float(loss):.6f} ppl~{ppls[-1]:.4f}")

    n_total = count_tree_params(params)
    emb_n = int(np.asarray(params["embedding"]).size)
    n_model = n_total - emb_n
    result = {
        "model": "nl-CSA",
        "config": asdict(cfg),
        "nl_rank": r,
        "nl_learning_rate": lr,
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
        "note": "Renyi next-token loss (not -log mu); plotted separately from mu-models when needed.",
    }
    _save_run(result, params, bundle["sentences"], cfg, "nlcsa")
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
    """Two panels: mu-models together; nl-CSA Renyi on its own axis/panel."""
    import matplotlib.pyplot as plt

    mu_res = [r for r in results if r["model"] in ("k-QSA", "k-CSA")]
    nl_res = [r for r in results if r["model"] == "nl-CSA"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for res in mu_res:
        ys = res["loss_history"]
        xs = np.arange(1, len(ys) + 1)
        ax.plot(xs, ys, linewidth=2, label=f"{res['model']} (angles={res.get('n_params_angles')})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"training loss $-\log\mu$")
    ax.set_title("k-QSA vs k-CSA (same objective, shared init)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for res in nl_res:
        ys = res["loss_history"]
        xs = np.arange(1, len(ys) + 1)
        ax.plot(
            xs,
            ys,
            linewidth=2,
            color="C2",
            label=f"nl-CSA Renyi (model_params={res.get('n_params_model')}, rank={res.get('nl_rank')})",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("Renyi training loss")
    ax.set_title("nl-CSA (softmax+FFN)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
