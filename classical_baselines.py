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
    # None / 0 => full-batch (smooth curves); >0 => SGD minibatch noise
    batch_size: Optional[int] = None
    checkpoint_every: int = 20
    resume: bool = True


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
    raw = np.load(params_path)
    # rebuild tree from template structure
    def _fill(tree, prefix=""):
        if isinstance(tree, dict):
            return {k: _fill(v, f"{prefix}/{k}" if prefix else str(k)) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            seq = [_fill(v, f"{prefix}/{i}") for i, v in enumerate(tree)]
            return type(tree)(seq)
        key = prefix.replace("/", "__")
        return jnp.asarray(raw[key])

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


def _train_mu_model(
    name: str,
    cfg: BaselineConfig,
    bundle: dict,
    params,
    log: logging.Logger,
) -> dict:
    tag = "kqsa" if name == "k-QSA" else "kcsa"
    run_name = f"{tag}_seed{cfg.seed}"
    out = _run_dir(cfg, run_name)

    if cfg.resume and _is_run_complete(out, cfg.epochs):
        result = _load_completed_result(out)
        log.info(f"[{name}] resume: already complete -> {out}")
        return result

    token_batch, pe = bundle["token_batch"], bundle["pe"]
    n = int(token_batch.shape[0])
    bs = int(cfg.batch_size) if cfg.batch_size else n
    bs = max(1, min(bs, n))
    rng = np.random.default_rng(cfg.seed)  # same minibatch schedule for QSA/CSA

    def loss_batch(p, batch):
        return jnp.mean(jax.vmap(lambda ids: mu_loss_sentence(p, ids, pe, cfg))(batch))

    @jax.jit
    def step(p, batch):
        loss, grads = jax.value_and_grad(loss_batch)(p, batch)
        p = jax.tree_util.tree_map(lambda x, g: x - cfg.learning_rate * g, p, grads)
        return p, loss

    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
    losses: List[float] = []
    wall_times: List[float] = []
    start_ep = 1
    elapsed0 = 0.0

    ckpt = _try_load_checkpoint(out, params) if cfg.resume else None
    if ckpt is not None:
        params = ckpt["params"]
        start_ep = int(ckpt["epoch"]) + 1
        losses = list(ckpt["loss_history"])
        wall_times = list(ckpt["wall_time_history"])
        rng = ckpt["rng"]
        elapsed0 = float(wall_times[-1]) if wall_times else 0.0
        log.info(f"[{name}] resume from epoch {ckpt['epoch']} -> {out}")

    t0 = time.perf_counter()
    for ep in range(start_ep, cfg.epochs + 1):
        if bs < n:
            idx = rng.choice(n, size=bs, replace=False)
            batch = token_batch[idx]
        else:
            batch = token_batch
        params, loss = step(params, batch)
        # report full-batch loss for comparable curves across seeds
        full_loss = float(loss_batch(params, token_batch))
        losses.append(full_loss)
        wall_times.append(elapsed0 + (time.perf_counter() - t0))
        if ep == 1 or ep % max(1, cfg.epochs // 10) == 0 or ep == cfg.epochs:
            log.info(
                f"[{name}] epoch={ep:03d}/{cfg.epochs} "
                f"step_loss={float(loss):.6f} full_loss={full_loss:.6f} "
                f"t={wall_times[-1]:.1f}s"
            )
        if cfg.checkpoint_every > 0 and (ep % cfg.checkpoint_every == 0 or ep == cfg.epochs):
            _save_checkpoint(out, params, losses, wall_times, ep, rng)

    angle_n = qsa_angle_param_count(cfg.d, cfg.layers)
    emb_n = int(np.asarray(params["embedding"]).size)
    result = {
        "model": name,
        "config": asdict(cfg),
        "seed": cfg.seed,
        "loss_history": losses,
        "wall_time_history": wall_times,
        "final_loss": losses[-1],
        "ppl_mu": float(np.exp(losses[-1])),
        "n_params_angles": angle_n,
        "n_params_embedding": emb_n,
        "n_params_total": angle_n + emb_n,
        "elapsed_seconds": wall_times[-1] if wall_times else 0.0,
        "num_sentences": n,
        "batch_size": bs,
        "vocab_size": int(bundle["encoding"].vocabSize),
        "note": (
            "Same classical -log(mu) objective and shared init/data as the other mu-model. "
            "Local k-QSA training does not use the quantum circuit. "
            "Logged loss is full-batch evaluation each epoch; wall_time_history is cumulative seconds."
        ),
    }
    _save_run(result, params, bundle["sentences"], cfg, run_name)
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
    run_name = f"nlcsa_seed{cfg.seed}"
    out = _run_dir(cfg, run_name)

    if cfg.resume and _is_run_complete(out, cfg.epochs):
        result = _load_completed_result(out)
        log.info(f"[nl-CSA] resume: already complete -> {out}")
        return result

    token_batch, pe = bundle["token_batch"], bundle["pe"]
    encoding = bundle["encoding"]
    key = jax.random.PRNGKey(cfg.seed + 13)
    params, r = init_nlcsa_params(encoding.vocabSize, cfg, key)
    lr = cfg.nl_learning_rate if cfg.nl_learning_rate is not None else max(cfg.learning_rate, 5e-3)
    n = int(token_batch.shape[0])
    bs = int(cfg.batch_size) if cfg.batch_size else n
    bs = max(1, min(bs, n))
    rng = np.random.default_rng(cfg.seed + 99)

    def loss_batch(p, batch):
        return jnp.mean(jax.vmap(lambda ids: nlcsa_loss_sentence(p, ids, pe, cfg))(batch))

    @jax.jit
    def step(p, batch):
        loss, grads = jax.value_and_grad(loss_batch)(p, batch)
        p = jax.tree_util.tree_map(lambda x, g: x - lr * g, p, grads)
        return p, loss

    losses: List[float] = []
    ppls: List[float] = []
    wall_times: List[float] = []
    start_ep = 1
    elapsed0 = 0.0

    ckpt = _try_load_checkpoint(out, params) if cfg.resume else None
    if ckpt is not None:
        params = ckpt["params"]
        start_ep = int(ckpt["epoch"]) + 1
        losses = list(ckpt["loss_history"])
        wall_times = list(ckpt["wall_time_history"])
        rng = ckpt["rng"]
        extra = ckpt.get("extra") or {}
        ppls = list(extra.get("ppl_history", [float(np.exp(x)) for x in losses]))
        r = int(extra.get("nl_rank", r))
        elapsed0 = float(wall_times[-1]) if wall_times else 0.0
        log.info(f"[nl-CSA] resume from epoch {ckpt['epoch']} -> {out}")

    t0 = time.perf_counter()
    for ep in range(start_ep, cfg.epochs + 1):
        if bs < n:
            idx = rng.choice(n, size=bs, replace=False)
            batch = token_batch[idx]
        else:
            batch = token_batch
        params, loss = step(params, batch)
        full_loss = float(loss_batch(params, token_batch))
        losses.append(full_loss)
        ppls.append(float(np.exp(full_loss)))
        wall_times.append(elapsed0 + (time.perf_counter() - t0))
        if ep == 1 or ep % max(1, cfg.epochs // 10) == 0 or ep == cfg.epochs:
            log.info(
                f"[nl-CSA] epoch={ep:03d}/{cfg.epochs} "
                f"step_loss={float(loss):.6f} full_loss={full_loss:.6f} "
                f"ppl~{ppls[-1]:.4f} t={wall_times[-1]:.1f}s"
            )
        if cfg.checkpoint_every > 0 and (ep % cfg.checkpoint_every == 0 or ep == cfg.epochs):
            _save_checkpoint(
                out,
                params,
                losses,
                wall_times,
                ep,
                rng,
                extra={"ppl_history": ppls, "nl_rank": r},
            )

    n_total = count_tree_params(params)
    emb_n = int(np.asarray(params["embedding"]).size)
    n_model = n_total - emb_n
    result = {
        "model": "nl-CSA",
        "config": asdict(cfg),
        "seed": cfg.seed,
        "nl_rank": r,
        "nl_learning_rate": lr,
        "loss_history": losses,
        "ppl_history": ppls,
        "wall_time_history": wall_times,
        "final_loss": losses[-1],
        "final_ppl": ppls[-1],
        "n_params_model": n_model,
        "n_params_embedding": emb_n,
        "n_params_total": n_total,
        "target_angle_budget": qsa_angle_param_count(cfg.d, cfg.layers),
        "elapsed_seconds": wall_times[-1] if wall_times else 0.0,
        "num_sentences": n,
        "batch_size": bs,
        "vocab_size": int(encoding.vocabSize),
        "note": "Renyi next-token loss (not -log mu); full-batch eval each epoch; wall_time cumulative.",
    }
    _save_run(result, params, bundle["sentences"], cfg, run_name)
    return result


def _save_run(result: dict, params, sentences, cfg: BaselineConfig, tag: str) -> None:
    if not cfg.output_dir:
        return
    out = Path(cfg.output_dir) / tag
    params_dir = out / "params"
    out.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    # metrics without huge arrays already only scalars/lists
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
        lines = ["epoch,wall_time_s,loss"]
        for i, (t, loss) in enumerate(zip(times, losses), start=1):
            lines.append(f"{i},{t:.6f},{loss:.8f}")
        (out / "loss_vs_time.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # final parameters (all leaves)
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
    """Mean ± std envelope over seeds for one model name."""
    assert runs, "empty runs"
    name = runs[0]["model"]
    hist = np.array([r["loss_history"] for r in runs], dtype=float)
    mean = hist.mean(axis=0)
    std = hist.std(axis=0)
    out = {
        "model": name,
        "seeds": [int(r.get("seed", -1)) for r in runs],
        "n_seeds": len(runs),
        "loss_history": mean.tolist(),
        "loss_std": std.tolist(),
        "loss_min": hist.min(axis=0).tolist(),
        "loss_max": hist.max(axis=0).tolist(),
        "seed_histories": hist.tolist(),
        "final_loss_mean": float(mean[-1]),
        "final_loss_std": float(std[-1]),
        "n_params_angles": runs[0].get("n_params_angles"),
        "n_params_model": runs[0].get("n_params_model"),
        "nl_rank": runs[0].get("nl_rank"),
        "batch_size": runs[0].get("batch_size"),
    }
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
    """Panels: loss vs epoch and loss vs wall-time, with mean ± std."""
    import matplotlib.pyplot as plt

    mu_res = [r for r in results if r["model"] in ("k-QSA", "k-CSA")]
    nl_res = [r for r in results if r["model"] == "nl-CSA"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def _plot_panel(ax, series: List[dict], x_mode: str, ylabel: str, title: str):
        for i, res in enumerate(series):
            ys = np.asarray(res["loss_history"], dtype=float)
            color = f"C{i}"
            n_seeds = int(res.get("n_seeds", 1))
            label = res["model"]
            if res.get("n_params_angles") is not None:
                label += f" (angles={res['n_params_angles']}"
            elif res.get("n_params_model") is not None:
                label += f" (model={res['n_params_model']}"
            else:
                label += " ("
            if n_seeds > 1:
                label += f", n_seeds={n_seeds}"
            label += ")"

            if x_mode == "epoch":
                xs = np.arange(1, len(ys) + 1)
                if show_seed_traces and "seed_histories" in res:
                    for h in res["seed_histories"]:
                        ax.plot(np.arange(1, len(h) + 1), h, color=color, alpha=0.25, linewidth=1)
            else:
                if "wall_time_history" not in res:
                    continue
                xs = np.asarray(res["wall_time_history"], dtype=float)
                if show_seed_traces and "seed_wall_times" in res and "seed_histories" in res:
                    for t_h, y_h in zip(res["seed_wall_times"], res["seed_histories"]):
                        ax.plot(t_h, y_h, color=color, alpha=0.25, linewidth=1)

            ax.plot(xs, ys, color=color, linewidth=2.2, label=label)
            if "loss_std" in res and n_seeds > 1:
                std = np.asarray(res["loss_std"], dtype=float)
                ax.fill_between(xs, ys - std, ys + std, color=color, alpha=0.2, linewidth=0)

        ax.set_xlabel("epoch" if x_mode == "epoch" else "wall time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    _plot_panel(axes[0, 0], mu_res, "epoch", r"$-\log\mu$", "k-QSA/k-CSA vs epoch")
    _plot_panel(axes[0, 1], mu_res, "time", r"$-\log\mu$", "k-QSA/k-CSA vs time")
    _plot_panel(axes[1, 0], nl_res, "epoch", "Renyi loss", "nl-CSA vs epoch")
    _plot_panel(axes[1, 1], nl_res, "time", "Renyi loss", "nl-CSA vs time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
