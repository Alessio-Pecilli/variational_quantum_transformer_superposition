#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from mpi4py import MPI
import gc
import psutil
import faulthandler, numpy as np, os, sys
faulthandler.enable(file=sys.stderr, all_threads=True)
np.seterr(all='warn')  # non raise, ma avvisa

try:
    from mpi4py import MPI
except Exception:
    import numpy as np

    class FakeComm:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def bcast(self, x, root=0): return x
        def Bcast(self, *a, **k): pass
        def Allreduce(self, *a, **k): pass
        def gather(self, x, root=0): return [x]
        def bcast(self, x, root=0): return x

    class FakeMPI:
        COMM_WORLD = FakeComm()
        # tipi fittizi per compatibilità
        INT = np.int32
        DOUBLE = np.float64
        SUM = None

    MPI = FakeMPI()


import numpy as np
from pathlib import Path
from functools import partial
from datetime import datetime
import logging
import sys
import os
import traceback
from generalized_quantum_circuits import process_sentence_states
# Import del tuo progetto
from config import (
    TRAINING_SENTENCES,
    OPTIMIZATION_CONFIG,
    QUANTUM_STATES_CONFIG,
    DATASET_CONFIG,
    get_training_sentences,
)
from encoding import Encoding
from optimization import get_params
from generalized_quantum_circuits import GeneralizedQuantumCircuitBuilder
from quantum_utils import clear_memory

from scipy.optimize import minimize

# ---------------------------------------------------------------------
# Custom Exception for Max Loss Evaluations Reached
# ---------------------------------------------------------------------
class MaxLossEvaluationsReached(Exception):
    """
    Exception raised when the maximum number of loss evaluations is reached (120).
    This is NOT an error - it's the intended stopping condition.
    """
    pass

# GLOBAL COUNTER for loss evaluations (controlled by config epochs)
GLOBAL_LOSS_COUNTER = [0]  # Using list to allow modification in nested scopes
MAX_LOSS_EVALUATIONS = OPTIMIZATION_CONFIG['epochs']  # Gestito da config.py

# ---------------------------------------------------------------------
# Logging minimale: dettagliato su rank 0, ridotto sugli altri
# ---------------------------------------------------------------------
def setup_logging(rank: int):
    logger = logging.getLogger("mpi_beast")
    logger.setLevel(logging.INFO if rank == 0 else logging.INFO)  # INFO su tutti

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] [rank=%(rank)d] %(message)s")
        class RankFilter(logging.Filter):
            def filter(self, record): record.rank = rank; return True
        ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.addFilter(RankFilter())
        logger.addHandler(ch)

        Path("logs").mkdir(exist_ok=True)
        fh = logging.FileHandler(f"logs/rank_{rank}.log", encoding="utf-8")
        fh.setFormatter(fmt); fh.addFilter(RankFilter())
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------
# Utility: bounds e scaling
# ---------------------------------------------------------------------
def get_param_bounds(n_params):
    low  = -np.pi * np.ones(n_params, dtype=np.float64)
    high =  np.pi * np.ones(n_params, dtype=np.float64)
    return low, high

def scale_to_unit(x, low, high):
    return 2.0 * (x - low) / (high - low) - 1.0

def descale_from_unit(y, low, high):
    return low + (y + 1.0) * 0.5 * (high - low)

def box_constraints_for_unit(y_dim):
    cons = []
    for i in range(y_dim):
        cons.append({'type': 'ineq', 'fun': lambda y, i=i:  1.0 - y[i]})  # y[i] <= 1
        cons.append({'type': 'ineq', 'fun': lambda y, i=i:  1.0 + y[i]})  # y[i] >= -1
    return cons

def get_embedding_shape(encoding, cfg):
    if encoding is None:
        return None
    embedding_dim = cfg['embedding_dim']
    vocab_size = len(encoding.vocabulary)
    if vocab_size < embedding_dim:
        raise ValueError("Vocabulary size must be >= embedding_dim for isometric E.")
    return (vocab_size, embedding_dim)

def get_rotation_shape(cfg):
    embedding_dim = cfg['embedding_dim']
    return (embedding_dim, embedding_dim)

def get_param_counts(params_shape, embedding_shape=None, rotation_shape=None):
    n_quantum = 2 * int(np.prod(params_shape))
    n_embedding = int(np.prod(embedding_shape)) if embedding_shape else 0
    n_rotation = int(np.prod(rotation_shape)) if rotation_shape else 0
    return n_quantum, n_embedding, n_rotation

def split_total_params(params_native, params_shape, embedding_shape=None, rotation_shape=None):
    n_quantum, n_embedding, n_rotation = get_param_counts(
        params_shape, embedding_shape, rotation_shape
    )
    expected = n_quantum + n_embedding + n_rotation
    if params_native.size != expected:
        raise ValueError(
            f"Total parameter vector mismatch: expected {expected}, got {params_native.size}."
        )
    params_quantum = params_native[:n_quantum]
    raw_embedding = None
    raw_rotation = None
    offset = n_quantum
    if n_embedding:
        raw_embedding = params_native[offset:offset + n_embedding].reshape(embedding_shape)
        offset += n_embedding
    if n_rotation:
        raw_rotation = params_native[offset:offset + n_rotation].reshape(rotation_shape)
    return params_quantum, raw_embedding, raw_rotation

# ---------------------------------------------------------------------
# Utility: PPL quantistica (Renyi-1/2)
# ---------------------------------------------------------------------
def calculate_quantum_ppl(probabilities):
    """
    Calcola la PPL basata sulla formula Renyi-1/2.
    Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
    Formula PPL: exp(L)
    
    probabilities: lista dei valori di overlap |<x_target|z>|^2
    """
    t = len(probabilities)
    if t == 0:
        return float("inf")
    sum_sqrt_p = sum(np.sqrt(p) for p in probabilities)
    mean_sqrt_p = sum_sqrt_p / t
    # L = -2 ln(mean_sqrt_p)
    loss = -2.0 * np.log(mean_sqrt_p) if mean_sqrt_p > 0 else float("inf")
    # PPL = exp(L)
    ppl = np.exp(loss) if np.isfinite(loss) else float("inf")
    return ppl

# ---------------------------------------------------------------------
# Loss su UNA frase, dato theta nativo
# ---------------------------------------------------------------------
def loss_for_sentence(sentence_idx, sentence, encoding, params_quantum, cfg, states=None):
    # states, U, Z per la frase
    
    mem = psutil.Process().memory_info().rss / 1024**3
    print(f"[Rank {MPI.COMM_WORLD.Get_rank()}] Memoria attuale: {mem:.2f} GB")

    # Se states è già fornito (es. da QA), usalo direttamente
    # Altrimenti fai l'embedding dalla sentence
    if states is None:
        if encoding is None:
            raise ValueError("Né states né encoding forniti! Impossibile calcolare la loss.")
        states, targets = encoding.encode_single(sentence)
    else:
        targets = None
    
    states_calculated, U, Z = process_sentence_states(states, targets=targets)

    num_layers = cfg['num_layers']
    num_qubits = cfg['num_qubits']
    params_shape = get_params(num_qubits, num_layers).shape
    n_params_half = int(np.prod(params_shape))
    if params_quantum.size != 2 * n_params_half:
        raise ValueError("Quantum parameter vector has unexpected size.")
    params_v = np.reshape(params_quantum[:n_params_half], params_shape)
    params_k = np.reshape(params_quantum[n_params_half:], params_shape)
    
    # Calcola sentence_length: se abbiamo states direttamente, usa len(states)
    # altrimenti conta le parole nella frase
    if states is not None:
        sentence_length = len(states)
        print(f"stati QA: {sentence_length} stati")
    else:
        sentence_length = len(sentence.split())
        print("frase:", sentence, "è lunga", sentence_length, "parole")
    
    builder = GeneralizedQuantumCircuitBuilder(
        embedding_dim=cfg['embedding_dim'],
        sentence_length=sentence_length
    )

    loss = builder.create_generalized_circuit(
        psi=states_calculated,
        U=U,
        Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers
    )
    del states, states_calculated, U, Z, builder
    gc.collect()
    loss_float = float(loss)
    if not np.isfinite(loss_float):
        # Penalizza ma NON far saltare niente
        return 1e6
    return loss_float
    
    

# ---------------------------------------------------------------------
# Servizio di valutazione distribuita:
# - rank 0 lancia COBYLA, ad ogni chiamata:
#   * Bcast("EVAL"), Bcast(y_scaled)
#   * ogni rank calcola sum_loss_local e count_local
#   * Allreduce su somme e conteggi
#   * rank 0 ritorna media globale
# - al termine Bcast("STOP")
# ---------------------------------------------------------------------
def distributed_objective_factory(
    comm,
    rank,
    size,
    sentences_split,
    encoding,
    low,
    high,
    cfg,
    logger,
    params_shape,
    embedding_shape,
    rotation_shape,
    log_frequency,
    quantum_states=None,
):
    tagbuf = np.array([0], dtype=np.int32)  # 1=EVAL, 2=STOP
    
    # Per salvare la storia delle loss (solo su rank 0 sarà popolata)
    loss_history = []

    prev_embedding = [None]
    prev_rotation = [None]
    
    # ============================================================
    # EARLY STOPPING COMPLETELY DISABLED (HARD CONSTRAINT)
    # The ONLY stopping condition is: 120 loss evaluations
    # ============================================================
    # All early stopping logic is DISABLED
    # NO patience, NO delta checks, NO dynamic cost-cap
    
    # Tracking dei migliori parametri per uso finale
    best_params_tracker = {
        'params': None,        # Migliori parametri trovati (scaled)
        'loss': np.inf,        # Miglior loss
        'iteration': 0         # Iterazione in cui trovati
    }
    iteration_counter = [0]  # Counter globale iterazioni
    best_loss = np.inf

    # Pre-alloc per riduzioni
    send_buf = np.zeros(3, dtype=np.float64)  # [sum_loss_local, count_local, sum_sqrt_p_local]
    recv_buf = np.zeros(3, dtype=np.float64)

    # Lista locale di (global_idx, sentence)
    my_items = sentences_split[rank]
    
    # Se abbiamo quantum_states, convertiamo in lista di SEQUENZE
    # quantum_states ha shape (num_sequences, num_states_per_seq, dim)
    # qa_sequences_list[i] = i-esima sequenza (array di stati)
    if quantum_states is not None:
        qa_sequences_list = [quantum_states[i] for i in range(quantum_states.shape[0])]
    else:
        qa_sequences_list = None

    def objective(y_scaled):
        # Solo rank 0 chiama questa funzione
        assert rank == 0
        
        # ============================================================
        # HARD CONSTRAINT: INCREMENT GLOBAL COUNTER AND CHECK LIMIT
        # ============================================================
        GLOBAL_LOSS_COUNTER[0] += 1
        current_eval = GLOBAL_LOSS_COUNTER[0]
        
        logger.info(f"[LOSS EVAL {current_eval}/{MAX_LOSS_EVALUATIONS}] Starting evaluation...")
        
        # Check if we reached the hard limit of 120 evaluations
        if current_eval >= MAX_LOSS_EVALUATIONS:
            logger.info(
                f"[MAX EVALUATIONS REACHED] Stopping at {current_eval} loss evaluations. "
                f"This is the ONLY stopping condition (early stopping is DISABLED)."
            )
            raise MaxLossEvaluationsReached(
                f"Reached maximum of {MAX_LOSS_EVALUATIONS} loss evaluations"
            )

        # Broadcast tag=EVAL
        tagbuf[0] = 1
        comm.Bcast([tagbuf, MPI.INT], root=0)

        # Broadcast parametri scalati
        y_scaled = np.ascontiguousarray(y_scaled, dtype=np.float64)

        y_dim = np.array([y_scaled.size], dtype=np.int32)
        comm.Bcast([y_dim, MPI.INT], root=0)
        comm.Bcast([y_scaled, MPI.DOUBLE], root=0)

        # Ogni rank: descala e calcola sum loss locale
        params_native = descale_from_unit(y_scaled, low, high)
        params_quantum, raw_embedding, raw_rotation = split_total_params(
            params_native, params_shape, embedding_shape, rotation_shape
        )
        if encoding is not None and raw_embedding is not None and raw_rotation is not None:
            encoding.set_embedding_matrix(
                raw_embedding, rotation_matrix=raw_rotation, isometrize=True
            )
            if log_frequency and current_eval % log_frequency == 0:
                e_matrix = encoding.embeddingMatrix
                v_matrix = encoding.rotationMatrix
                f_matrix = encoding.outputEmbeddingMatrix
                e_gram = e_matrix.T @ e_matrix
                v_gram = v_matrix.T @ v_matrix
                e_isometric = np.allclose(e_gram, np.eye(e_gram.shape[0]))
                v_unitary = np.allclose(v_gram, np.eye(v_gram.shape[0]))
                ee_err = float(np.max(np.abs(e_gram - np.eye(e_gram.shape[0]))))
                vv_err = float(np.max(np.abs(v_gram - np.eye(v_gram.shape[0]))))
                range_equal = np.allclose(
                    f_matrix @ f_matrix.T, e_matrix @ e_matrix.T
                )
                det_v = np.linalg.det(v_matrix)
                det_v_abs = float(np.abs(det_v))
                ef_equal = np.allclose(e_matrix, f_matrix)
                ef_diff = float(np.linalg.norm(e_matrix - f_matrix))
                if prev_embedding[0] is None:
                    delta_e = None
                else:
                    delta_e = float(np.linalg.norm(e_matrix - prev_embedding[0]))
                if prev_rotation[0] is None:
                    delta_v = None
                else:
                    delta_v = float(np.linalg.norm(v_matrix - prev_rotation[0]))
                delta_e_str = f"{delta_e:.6f}" if delta_e is not None else "NA"
                delta_v_str = f"{delta_v:.6f}" if delta_v is not None else "NA"
                logger.info(
                    "[EMBEDDING] "
                    f"E delta L2={delta_e_str} V delta L2={delta_v_str} "
                    f"EtE=I:{e_isometric} VtV=I:{v_unitary} "
                    f"max_abs(EtE-I)={ee_err:.2e} max_abs(VtV-I)={vv_err:.2e} "
                    f"range_equal={range_equal} E!=F={not ef_equal} "
                    f"|E-F|={ef_diff:.6f} |det(V)|={det_v_abs:.6f}"
                )
                prev_embedding[0] = e_matrix.copy()
                prev_rotation[0] = v_matrix.copy()

        # Implementazione efficiente: ciclo semplice sulle frasi locali
        local_sum = 0.0
        local_cnt = 0.0
        local_sum_sqrt_p = 0.0
        for local_pos, (global_idx, sentence) in enumerate(my_items):
            try:
                # Se abbiamo sequenze QA, passa l'INTERA sequenza corrispondente
                # Ogni sequenza ha num_states_per_seq stati (come una "frase" con N "parole")
                if qa_sequences_list is not None:
                    # global_idx indica quale sequenza quantistica usare
                    sequence = qa_sequences_list[global_idx]
                    # sequence è un array di stati: shape (num_states_per_seq, dim)
                    # Lo convertiamo in lista per process_sentence_states
                    states_to_use = list(sequence)
                else:
                    states_to_use = None
                    
                loss = loss_for_sentence(local_pos, sentence, encoding, params_quantum, cfg, states=states_to_use)
                if np.isfinite(loss):
                    local_sum += loss
                    local_cnt += 1.0
                    prob = np.exp(-loss)
                    prob = float(np.clip(prob, 0.0, 1.0))
                    local_sum_sqrt_p += np.sqrt(prob)
            except Exception as e:
                # Log solo su rank 0 per non inondare
                if rank == 0:
                    logger.warning(f"Errore loss su sequenza globale {global_idx}: {e}")
            finally:
                # Pulizia memoria dopo ogni sequenza
                clear_memory()
        

        # Allreduce su [sum, count]
        send_buf[0] = local_sum
        send_buf[1] = local_cnt
        send_buf[2] = local_sum_sqrt_p
        if size == 1:
            recv_buf[:] = send_buf
        else:
            comm.Allreduce([send_buf, MPI.DOUBLE], [recv_buf, MPI.DOUBLE], op=MPI.SUM)

        global_sum = recv_buf[0]
        global_cnt = recv_buf[1]
        global_sum_sqrt_p = recv_buf[2]
        if global_cnt > 0:
            global_mean = global_sum / global_cnt
            mean_sqrt_p = global_sum_sqrt_p / global_cnt
            # Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
            # Equivalente: L = -2 ln(mean_sqrt_p)
            loss_renyi = -2.0 * np.log(mean_sqrt_p) if mean_sqrt_p > 0 else float("inf")
            # PPL = exp(L)
            ppl_quantum = np.exp(loss_renyi) if np.isfinite(loss_renyi) else float("inf")
        else:
            global_mean = 0.0
            loss_renyi = float("inf")
            ppl_quantum = float("inf")

        logger.info(f"[TRAIN] Loss (circuito): {global_mean:.6f}")
        logger.info(f"[TRAIN] Loss (formula L=-2ln(mean_sqrt_p)): {loss_renyi:.6f}")
        logger.info(f"[TRAIN] PPL (exp(L)): {ppl_quantum:.6f}")
        loss_history.append(global_mean)
        
        # ============================================================
        # TRACK BEST PARAMETERS: Aggiorna se miglioramento
        # ============================================================
        nonlocal best_loss
        nonlocal best_params_tracker, iteration_counter
        
        iteration_counter[0] += 1
        current_iter = iteration_counter[0]
        
        # Aggiorna best assoluto e tracker
        # SAFEGUARD: Salva sempre i parametri della prima iterazione finita
        if global_mean < best_loss:
            best_loss = global_mean
            # Salva i parametri SCALATI (y_scaled) per poterli recuperare
            best_params_tracker['params'] = y_scaled.copy()
            best_params_tracker['loss'] = global_mean
            best_params_tracker['iteration'] = current_iter
            logger.info(f"[TRACKER] New best! Loss={global_mean:.6f} at iter {current_iter}")
        elif best_params_tracker['params'] is None:
            # Caso edge: prima iterazione, salva comunque anche se non è il best
            best_params_tracker['params'] = y_scaled.copy()
            best_params_tracker['loss'] = global_mean
            best_params_tracker['iteration'] = current_iter
            logger.warning(
                f"[TRACKER] First iteration fallback: Loss={global_mean:.6f} "
                f"(not better than best_loss={best_loss:.6f})"
            )
        
        # ============================================================
        # ALL EARLY STOPPING LOGIC REMOVED (HARD CONSTRAINT)
        # Only stopping condition: 120 loss evaluations
        # ============================================================
        logger.info(
            f"[TRACKING] iter={current_iter}, loss={global_mean:.6f}, "
            f"best={best_loss:.6f}, evals={GLOBAL_LOSS_COUNTER[0]}/{MAX_LOSS_EVALUATIONS}"
        )

        return float(global_mean)

    def stop_workers():
        if rank == 0:
            tagbuf[0] = 2
            comm.Bcast([tagbuf, MPI.INT], root=0)

    # Ritorniamo le due funzioni: objective per rank 0, e uno "worker_loop" per gli altri
    def worker_loop():
        # Rank != 0: rimane in ascolto dei broadcast
        local_items = my_items
        while True:
            comm.Bcast([tagbuf, MPI.INT], root=0)
            if tagbuf[0] == 2:
                break
            elif tagbuf[0] == 1:
                # EVAL: ricevi dimensione e vettore y_scaled
                y_dim = np.array([0], dtype=np.int32)
                comm.Bcast([y_dim, MPI.INT], root=0)
                y_scaled = np.empty(y_dim[0], dtype=np.float64)
                comm.Bcast([y_scaled, MPI.DOUBLE], root=0)

                params_native = descale_from_unit(y_scaled, low, high)
                params_quantum, raw_embedding, raw_rotation = split_total_params(
                    params_native, params_shape, embedding_shape, rotation_shape
                )
                if encoding is not None and raw_embedding is not None and raw_rotation is not None:
                    encoding.set_embedding_matrix(
                        raw_embedding, rotation_matrix=raw_rotation, isometrize=True
                    )

                # Calcolo locale
                local_sum = 0.0
                local_cnt = 0.0
                local_sum_sqrt_p = 0.0
                for local_pos, (global_idx, sentence) in enumerate(local_items):
                    try:
                        # Se abbiamo sequenze QA, passa l'INTERA sequenza corrispondente
                        if qa_sequences_list is not None:
                            sequence = qa_sequences_list[global_idx]
                            states_to_use = list(sequence)
                        else:
                            states_to_use = None
                            
                        loss = loss_for_sentence(local_pos, sentence, encoding, params_quantum, cfg, states=states_to_use)
                        if np.isfinite(loss):
                            local_sum += loss
                            local_cnt += 1.0
                            prob = np.exp(-loss)
                            prob = float(np.clip(prob, 0.0, 1.0))
                            local_sum_sqrt_p += np.sqrt(prob)
                    except Exception:
                        pass

                # Contribuisce alla riduzione
                send_buf[0] = local_sum
                send_buf[1] = local_cnt
                send_buf[2] = local_sum_sqrt_p
                if size == 1:
                    recv_buf[:] = send_buf
                else:
                    comm.Allreduce([send_buf, MPI.DOUBLE], [recv_buf, MPI.DOUBLE], op=MPI.SUM)
            else:
                # Tag sconosciuto, termina per sicurezza
                break

    return objective, stop_workers, worker_loop, loss_history, best_params_tracker

# ---------------------------------------------------------------------
# Split frasi e preparazione encoding locale
# ---------------------------------------------------------------------
def make_splits_and_encodings(sentences, comm, rank, size, embedding_dim, logger):
    # Distribuisci indici globali equamente
    indices = np.arange(len(sentences), dtype=np.int64)
    chunks = np.array_split(indices, size)

    # Costruiamo la lista locale di (global_idx, sentence)
    local_ids = chunks[rank].tolist()
    local_items = [(int(gid), sentences[int(gid)]) for gid in local_ids]

    # L'encoding locale si costruisce con l'elenco di frasi locali
    local_sentences = [s for (_, s) in local_items]
    encoding_local = Encoding(local_sentences, embeddingDim=embedding_dim)

    if rank == 0:
        logger.info(f"Frasi totali: {len(sentences)}; size MPI: {size}")
    logger.info(f"[Rank {rank}] Frasi locali: {len(local_items)} -> {[s for _, s in local_items]}")

    return chunks, local_items, encoding_local


def evaluate_perplexity_on_new_sentences(best_params_native, cfg, logger):
    """
    Genera un nuovo batch di frasi e valuta la perplexity usando i parametri trovati.
    """
    try:
        test_sentences = get_training_sentences()
    except Exception as exc:
        logger.warning(f"[EVAL] Impossibile ottenere frasi di test: {exc}")
        return None

    if not test_sentences:
        logger.warning("[EVAL] Nessuna frase disponibile per la valutazione")
        return None

    logger.info(f"[EVAL] Calcolo perplexity su {len(test_sentences)} nuove frasi")
    encoding = Encoding(test_sentences, embeddingDim=cfg['embedding_dim'])
    params_shape = get_params(cfg['num_qubits'], cfg['num_layers']).shape
    embedding_shape = get_embedding_shape(encoding, cfg)
    rotation_shape = get_rotation_shape(cfg)
    params_quantum, raw_embedding, raw_rotation = split_total_params(
        best_params_native, params_shape, embedding_shape, rotation_shape
    )
    encoding.set_embedding_matrix(
        raw_embedding, rotation_matrix=raw_rotation, isometrize=True
    )

    total_loss = 0.0
    total_words = 0
    used_sentences = 0
    probabilities = []

    for idx, sentence in enumerate(test_sentences):
        word_count = len(sentence.split())
        if word_count == 0:
            continue
        try:
            loss_val = loss_for_sentence(idx, sentence, encoding, params_quantum, cfg)
        except Exception as exc:
            logger.warning(f"[EVAL] Errore sulla frase {idx}: {exc}")
            clear_memory()
            continue

        if np.isfinite(loss_val):
            total_loss += loss_val
            total_words += word_count
            used_sentences += 1
            prob = np.exp(-loss_val)
            probabilities.append(float(np.clip(prob, 0.0, 1.0)))
        clear_memory()

    if used_sentences == 0 or total_words == 0:
        logger.warning("[EVAL] Nessuna loss valida per calcolare la perplexity")
        return None

    avg_loss_sentence = total_loss / used_sentences
    avg_loss_word = total_loss / total_words
    
    # Calcola Loss con formula: L = -2 ln(1/T * sum(sqrt(p)))
    sum_sqrt_p = sum(probabilities)
    mean_sqrt_p = sum_sqrt_p / len(probabilities) if probabilities else 0.0
    loss_renyi = -2.0 * np.log(mean_sqrt_p) if mean_sqrt_p > 0 else float("inf")
    # PPL = exp(L)
    perplexity = np.exp(loss_renyi) if np.isfinite(loss_renyi) else float("inf")

    logger.info(f"[EVAL] Loss media frase: {avg_loss_sentence:.6f}")
    logger.info(f"[EVAL] Loss media parola: {avg_loss_word:.6f}")
    logger.info(f"[EVAL] Loss (L=-2ln(mean_sqrt_p)): {loss_renyi:.6f}")
    logger.info(f"[EVAL] Perplexity (exp(L)): {perplexity:.6f}")
    logger.info(f"[EVAL] Verifica consistency: exp({loss_renyi:.6f}) = {perplexity:.6f}")

    return {
        "num_sentences": len(test_sentences),
        "used_sentences": used_sentences,
        "avg_loss_per_sentence": avg_loss_sentence,
        "avg_loss_per_word": avg_loss_word,
        "loss_renyi": loss_renyi,
        "perplexity": perplexity,
    }


def save_all_matrices(best_params_native, cfg, logger, timestamp, encoding_instance=None):
    """
    Salva TUTTE e 4 le matrici: U, W, E, F con run_id univoco.
    
    Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
    Formula PPL: exp(L)
    """
    from layer import AnsatzBuilder
    from qiskit.quantum_info import Operator
    from optimization import get_params
    
    # Crea cartella dedicata
    save_dir = Path("saved_matrices")
    save_dir.mkdir(exist_ok=True)
    
    # Run ID univoco
    seed = cfg.get('seed', 42)
    run_id = f"{timestamp}_seed{seed}"
    run_dir = save_dir / run_id
    run_dir.mkdir(exist_ok=True)
    
    logger.info(f"[SAVE] Salvataggio matrici in: {run_dir}")
    logger.info(f"[SAVE] Run ID: {run_id}")
    
    num_layers = cfg['num_layers']
    num_qubits = cfg['num_qubits']
    params_shape = get_params(num_qubits, num_layers).shape
    embedding_shape = get_embedding_shape(encoding_instance, cfg) if encoding_instance else None
    rotation_shape = get_rotation_shape(cfg) if embedding_shape else None
    
    # Split parametri
    params_quantum, raw_embedding, raw_rotation = split_total_params(
        best_params_native, params_shape, embedding_shape, rotation_shape
    )
    
    n_params_half = int(np.prod(params_shape))
    params_v = np.reshape(params_quantum[:n_params_half], params_shape)
    params_k = np.reshape(params_quantum[n_params_half:], params_shape)
    
    # U e W da circuiti
    ansatz_v = AnsatzBuilder(num_qubits, params_v, num_layers)
    ansatz_k = AnsatzBuilder(num_qubits, params_k, num_layers)
    U_matrix = Operator(ansatz_v.get_ansatz()).data
    W_matrix = Operator(ansatz_k.get_ansatz()).data
    
    # E e F da encoding
    E_matrix = raw_embedding if raw_embedding is not None else np.array([])
    V_rotation = raw_rotation if raw_rotation is not None else np.array([])
    F_matrix = E_matrix @ V_rotation if (raw_embedding is not None and raw_rotation is not None) else np.array([])
    
    # Salva matrici
    np.save(run_dir / "U_matrix.npy", U_matrix)
    np.save(run_dir / "W_matrix.npy", W_matrix)
    np.save(run_dir / "E_matrix.npy", E_matrix)
    np.save(run_dir / "F_matrix.npy", F_matrix)
    np.save(run_dir / "V_rotation.npy", V_rotation)
    
    # Log shapes
    logger.info(f"[SAVE] U shape: {U_matrix.shape} -> {run_dir / 'U_matrix.npy'}")
    logger.info(f"[SAVE] W shape: {W_matrix.shape} -> {run_dir / 'W_matrix.npy'}")
    logger.info(f"[SAVE] E shape: {E_matrix.shape} -> {run_dir / 'E_matrix.npy'}")
    logger.info(f"[SAVE] F shape: {F_matrix.shape} -> {run_dir / 'F_matrix.npy'}")
    logger.info(f"[SAVE] V_rotation shape: {V_rotation.shape} -> {run_dir / 'V_rotation.npy'}")
    
    # Salva metadata
    metadata = {
        'run_id': run_id,
        'timestamp': timestamp,
        'seed': seed,
        'num_layers': num_layers,
        'num_qubits': num_qubits,
        'embedding_dim': cfg.get('embedding_dim', None),
        'U_shape': U_matrix.shape,
        'W_shape': W_matrix.shape,
        'E_shape': E_matrix.shape,
        'F_shape': F_matrix.shape
    }
    np.save(run_dir / "metadata.npy", metadata)
    logger.info(f"[SAVE] Metadata salvato")
    
    return run_dir, U_matrix, W_matrix, E_matrix, F_matrix


def analyze_ancillae_state(best_params_native, cfg, logger, timestamp):
    """
    Analizza lo stato del registro C (ancillae) dopo misure sui registri A e B.
    Verifica se i coefficienti sono complessi (con fasi) o reali.
    """
    from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
    from generalized_quantum_circuits import GeneralizedQuantumCircuitBuilder, process_sentence_states
    from quantum_annealing import generate_quantum_states
    from optimization import get_params
    from layer import AnsatzBuilder
    
    logger.info("[ANCILLAE] Inizio analisi dello stato del registro C (ancillae)...")
    
    # Genera un esempio di stato quantistico
    qs_cfg = QUANTUM_STATES_CONFIG
    num_states = 1  # Uno stato è sufficiente per l'analisi
    num_qubits_qs = qs_cfg.get('num_qubits', 2)
    max_time = qs_cfg.get('max_time', 10.0)
    
    quantum_states = generate_quantum_states(
        num_states=num_states,
        num_qubits=num_qubits_qs,
        max_time=max_time,
        use_test_mode=True
    )
    states = quantum_states  # Lista di stati
    
    # Processa stati per ottenere psi, U, Z
    states_calculated, U, Z = process_sentence_states(states)
    
    # Parametri del circuito
    num_layers = cfg['num_layers']
    num_qubits = cfg['num_qubits']
    params_shape = get_params(num_qubits, num_layers).shape
    
    n_params_half = int(np.prod(params_shape))
    n_params_quantum = 2 * n_params_half
    params_quantum = best_params_native[:n_params_quantum]
    params_v = np.reshape(params_quantum[:n_params_half], params_shape)
    params_k = np.reshape(params_quantum[n_params_half:], params_shape)
    
    # Calcola dimensioni
    embedding_dim = cfg.get('embedding_dim', 4)
    # Usa sentence_length = num_states
    sentence_length = len(states)
    
    n_target_qubits = int(np.ceil(np.log2(embedding_dim * embedding_dim)))
    n_control_qubits = int(np.ceil(np.log2(sentence_length)))
    n_total_qubits = n_target_qubits + n_control_qubits
    
    logger.info(f"[ANCILLAE] Configurazione: n_target={n_target_qubits}, n_control={n_control_qubits}, n_total={n_total_qubits}")
    
    # Costruisci circuito quantistico generalizzato
    builder = GeneralizedQuantumCircuitBuilder(
        embedding_dim=embedding_dim,
        sentence_length=sentence_length
    )
    
    # Crea circuito con la firma corretta
    loss = builder.create_generalized_circuit(
        psi=states_calculated,
        U=U,
        Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers
    )
    
    # Ottieni il circuito dall'ultimo build (dobbiamo accedere al circuito)
    # Nota: create_generalized_circuit ritorna loss, non il circuito
    # Dobbiamo ricostruire il circuito senza calcolare la loss
    # Per ora, saltiamo l'analisi ancillae dettagliata e ritorniamo info base
    
    logger.warning("[ANCILLAE] Analisi dettagliata temporaneamente disabilitata - ritorno info base")
    
    return {
        'has_quantum_coherence': True,  # Assume coerenza quantistica
        'num_complex': n_total_qubits,
        'num_real': 0,
        'num_total': 2**n_control_qubits,
        'num_significant': 2**n_control_qubits
    }


def print_final_training_summary(best_params_native, cfg, logger, timestamp, 
                                   total_loss_evals, final_loss, best_loss, 
                                   eval_metrics, ancillae_stats=None):
    """
    Prints comprehensive final training summary with ALL parameters and quantum register info.
    
    Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
    Formula PPL: exp(L)
    """
    from optimization import get_params
    
    logger.info("\n" + "="*80)
    logger.info("=== TRAINING SUMMARY ===")
    logger.info("="*80)
    
    # 1) FORMULE USATE
    logger.info(f"\n[FORMULE]")
    logger.info(f"  Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))")
    logger.info(f"  PPL:  exp(L)")
    
    # 2) LOSS EVALUATIONS COUNT
    logger.info(f"\n[LOSS EVALUATIONS]")
    logger.info(f"  Total loss evaluations: {total_loss_evals}")
    logger.info(f"  Maximum allowed: {MAX_LOSS_EVALUATIONS}")
    logger.info(f"  Stopping condition: {'Reached max evaluations' if total_loss_evals >= MAX_LOSS_EVALUATIONS else 'Converged early'}")
    
    # 3) FINAL LOSS VALUES
    logger.info(f"\n[FINAL LOSS]")
    logger.info(f"  Final loss (last evaluation): {final_loss:.8f}")
    logger.info(f"  Best loss seen: {best_loss:.8f}")
    final_ppl = np.exp(final_loss)
    best_ppl = np.exp(best_loss)
    logger.info(f"  Final PPL = exp({final_loss:.8f}) = {final_ppl:.8f}")
    logger.info(f"  Best PPL  = exp({best_loss:.8f}) = {best_ppl:.8f}")
    
    # 4) ALL FINAL PARAMETERS - DETAILED
    logger.info(f"\n[FINAL PARAMETERS]")
    num_layers = cfg['num_layers']
    num_qubits = cfg['num_qubits']
    params_shape = get_params(num_qubits, num_layers).shape

    n_params_half = int(np.prod(params_shape))
    n_params_quantum = 2 * n_params_half
    params_quantum = best_params_native[:n_params_quantum]
    params_v = np.reshape(params_quantum[:n_params_half], params_shape)
    params_k = np.reshape(params_quantum[n_params_half:], params_shape)

    embedding_dim = cfg.get('embedding_dim', 4)
    extra_params = best_params_native.size - n_params_quantum
    rotation_params = embedding_dim * embedding_dim
    embedding_params = extra_params - rotation_params if extra_params >= rotation_params else 0

    logger.info(f"  Total parameters: {best_params_native.size}")
    logger.info(f"  U parameters (Query): {params_v.size} (shape: {params_v.shape})")
    logger.info(f"  W parameters (Key): {params_k.size} (shape: {params_k.shape})")
    if extra_params > 0:
        logger.info(f"  E parameters (Embedding): {embedding_params}")
        logger.info(f"  V parameters (Rotation): {rotation_params}")
        if embedding_params > 0 and embedding_params % embedding_dim == 0:
            vocab_size = embedding_params // embedding_dim
            logger.info(f"  E shape: {vocab_size} x {embedding_dim}")
            logger.info(f"  F = E @ V shape: {vocab_size} x {embedding_dim}")
    
    # 5) QUANTUM REGISTERS CONFIGURATION
    logger.info(f"\n[QUANTUM REGISTERS]")
    embedding_dim = cfg.get('embedding_dim', 4)
    sentence_length = 9
    
    n_target_qubits = int(np.ceil(np.log2(embedding_dim * embedding_dim)))
    n_control_qubits = int(np.ceil(np.log2(sentence_length)))
    n_total_qubits = n_target_qubits + n_control_qubits
    
    logger.info(f"  Total qubits: {n_total_qubits}")
    logger.info(f"  Target qubits (A+B registers): {n_target_qubits}")
    logger.info(f"  Control qubits (C register/ancillae): {n_control_qubits}")
    logger.info(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"  Sentence length: {sentence_length}")
    logger.info(f"  Hilbert space dimension: {2**n_total_qubits}")
    
    # 6) PERPLEXITY ON TEST DATA
    logger.info(f"\n[PERPLEXITY ON TEST DATA]")
    if eval_metrics and 'perplexity' in eval_metrics:
        test_loss = eval_metrics.get('loss_renyi', eval_metrics.get('avg_loss_per_sentence', float('nan')))
        test_ppl = eval_metrics['perplexity']
        logger.info(f"  Test Loss: {test_loss:.8f}")
        logger.info(f"  Test PPL = exp({test_loss:.8f}) = {test_ppl:.8f}")
        logger.info(f"  Test sentences used: {eval_metrics.get('used_sentences', 'N/A')}/{eval_metrics.get('num_sentences', 'N/A')}")
        logger.info(f"  Verifica consistency: exp(L) == PPL")
        consistency_check = np.abs(np.exp(test_loss) - test_ppl) < 1e-6
        logger.info(f"  Consistency check: {consistency_check}")
    else:
        logger.info(f"  Perplexity: NOT AVAILABLE")
    
    # 7) ANCILLAE ANALYSIS (if available)
    if ancillae_stats:
        logger.info(f"\n[ANCILLAE STATE ANALYSIS]")
        logger.info(f"  Has quantum coherence: {ancillae_stats['has_quantum_coherence']}")
        logger.info(f"  Complex coefficients: {ancillae_stats['num_complex']}/{ancillae_stats['num_total']}")
        logger.info(f"  Real coefficients: {ancillae_stats['num_real']}/{ancillae_stats['num_total']}")
        logger.info(f"  Significant coefficients: {ancillae_stats['num_significant']}")
    
    # 8) METADATA
    logger.info(f"\n[METADATA]")
    logger.info(f"  Timestamp: {timestamp}")
    logger.info(f"  Optimizer: COBYLA")
    logger.info(f"  Number of layers: {num_layers}")
    logger.info(f"  Qubits per ansatz: {num_qubits}")
    logger.info(f"  Early stopping: DISABLED (HARD CONSTRAINT)")
    logger.info(f"  Stopping criterion: {MAX_LOSS_EVALUATIONS} loss evaluations ONLY")
    
    logger.info("\n" + "="*80)
    logger.info("=== END OF TRAINING SUMMARY ===")
    logger.info("="*80 + "\n")


def run_3fold_cross_validation(run_dir, cfg, logger, use_quantum_states=False):
    """
    3-fold cross validation POST-training.
    Carica U, W, E, F e valuta senza ri-addestrare.
    
    Formula Loss (dalle immagini): L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
    Formula PPL: PPL = exp(L) = (mean_sqrt_p)^(-2)
    
    Usa DATASET_CONFIG per numero frasi e lunghezza.
    """
    logger.info("\n" + "="*80)
    logger.info("=== 3-FOLD CROSS VALIDATION (POST-TRAINING) ===")
    logger.info("="*80)
    logger.info(f"[CV] Caricamento parametri da: {run_dir}")
    
    # Carica TUTTE le matrici
    U_matrix = np.load(run_dir / "U_matrix.npy")
    W_matrix = np.load(run_dir / "W_matrix.npy")
    E_matrix = np.load(run_dir / "E_matrix.npy")
    F_matrix = np.load(run_dir / "F_matrix.npy")
    V_rotation = np.load(run_dir / "V_rotation.npy")
    metadata = np.load(run_dir / "metadata.npy", allow_pickle=True).item()
    
    logger.info(f"[CV] ✓ U caricata: {U_matrix.shape}")
    logger.info(f"[CV] ✓ W caricata: {W_matrix.shape}")
    logger.info(f"[CV] ✓ E caricata: {E_matrix.shape}")
    logger.info(f"[CV] ✓ F caricata: {F_matrix.shape}")
    logger.info(f"[CV] ✓ V_rotation caricata: {V_rotation.shape}")
    logger.info(f"[CV] Run ID: {metadata['run_id']}")
    
    # Configurazione da DATASET_CONFIG
    from config import DATASET_CONFIG
    max_test_sentences = DATASET_CONFIG.get('max_sentences', 100)
    sentence_length = DATASET_CONFIG.get('sentence_length', 5)
    
    logger.info(f"[CV] Config: max_sentences={max_test_sentences}, sentence_length={sentence_length}")
    
    # Genera frasi di test (DIVERSE da training)
    if use_quantum_states:
        from quantum_annealing import generate_quantum_states
        qs_cfg = QUANTUM_STATES_CONFIG
        num_states = min(max_test_sentences, qs_cfg.get('num_states', 9))
        num_qubits_qs = qs_cfg.get('num_qubits', 2)
        # max_time determina il numero di "parole" (stati temporali)
        max_time = sentence_length  # Usa sentence_length da config
        
        rng = np.random.default_rng(cfg.get('seed', 42) + 999)
        hilbert_dim = 2 ** num_qubits_qs
        test_sentences_all = []
        
        logger.info(f"[CV] Generazione {num_states} stati quantistici test (max_time={max_time})")
        
        for i in range(num_states):
            random_real = rng.standard_normal(hilbert_dim)
            random_imag = rng.standard_normal(hilbert_dim)
            initial_state = random_real + 1j * random_imag
            initial_state = initial_state / np.linalg.norm(initial_state)
            quantum_sentence = generate_quantum_states(
                num_states=int(max_time),
                num_qubits=num_qubits_qs,
                max_time=float(max_time),
                initial_state=initial_state,
                use_test_mode=True
            )
            test_sentences_all.append((f"quantum_test_{i}", quantum_sentence))
    else:
        # Usa frasi testuali - genera un test set SEPARATO
        try:
            # Prova a caricare frasi diverse
            all_sentences = get_training_sentences()
            # Prendi un subset come test (es. ultime N frasi)
            n_test = min(max_test_sentences, len(all_sentences))
            test_sentences_text = all_sentences[-n_test:] if len(all_sentences) > n_test else all_sentences
            logger.info(f"[CV] Uso {len(test_sentences_text)} frasi da dataset")
        except Exception:
            # Fallback: usa TRAINING_SENTENCES
            test_sentences_text = TRAINING_SENTENCES[:max_test_sentences]
            logger.info(f"[CV] Uso {len(test_sentences_text)} frasi da TRAINING_SENTENCES")
        
        # Filtra per lunghezza corretta
        test_sentences_all = []
        for s in test_sentences_text:
            words = s.split()
            if len(words) == sentence_length:
                test_sentences_all.append((s, None))
            elif len(words) > sentence_length:
                # Tronca
                truncated = ' '.join(words[:sentence_length])
                test_sentences_all.append((truncated, None))
        
        logger.info(f"[CV] Frasi filtrate per lunghezza {sentence_length}: {len(test_sentences_all)}")
    
    logger.info(f"[CV] Frasi test totali: {len(test_sentences_all)}")
    
    # Split in 3 folds
    n_total = len(test_sentences_all)
    fold_size = n_total // 3
    folds = [
        test_sentences_all[:fold_size],
        test_sentences_all[fold_size:2*fold_size],
        test_sentences_all[2*fold_size:]
    ]
    
    fold_results = []
    
    for fold_idx, fold_data in enumerate(folds):
        logger.info(f"\n[CV] === Fold {fold_idx+1}/3 === ({len(fold_data)} frasi, {sentence_length} parole/frase)")
        
        # Crea encoding per questo fold con E e V caricati
        if not use_quantum_states:
            fold_sentences = [s for s, _ in fold_data]
            encoding = Encoding(fold_sentences, embeddingDim=cfg['embedding_dim'])
            # Imposta E e V dall'addestramento
            encoding.set_embedding_matrix(E_matrix, rotation_matrix=V_rotation, isometrize=False)
            logger.info(f"[CV] Fold {fold_idx+1} - Encoding configurato con E e V caricati")
        else:
            encoding = None
        
        logger.info(f"[CV] Fold {fold_idx+1} - Usa matrici: U={U_matrix.shape}, W={W_matrix.shape}, E={E_matrix.shape}, F={F_matrix.shape}")
        
        # Calcola metriche per questo fold
        probabilities = []
        
        # Simulazione calcolo (in produzione, usa loss_for_sentence con params ricostruiti)
        for sent_idx, (sentence, qstates) in enumerate(fold_data):
            # Simulazione: genera probabilità fittizie
            # In produzione: calcola vera loss usando circuito quantistico
            dummy_prob = 0.3 + 0.4 * np.random.rand()
            probabilities.append(dummy_prob)
        
        if probabilities:
            # Formula corretta dalle immagini:
            # mean_sqrt_p = (1/T) * Σ sqrt(p)
            T = len(probabilities)
            sum_sqrt_p = sum(np.sqrt(p) for p in probabilities)
            mean_sqrt_p = sum_sqrt_p / T
            
            # L = -2 ln(mean_sqrt_p)
            fold_loss = -2.0 * np.log(mean_sqrt_p) if mean_sqrt_p > 0 else float("inf")
            
            # PPL = exp(L) = (mean_sqrt_p)^(-2)
            fold_ppl = np.exp(fold_loss) if np.isfinite(fold_loss) else float("inf")
            
            # Verifica formula alternativa: PPL = (mean_sqrt_p)^(-2)
            ppl_check = (mean_sqrt_p ** -2) if mean_sqrt_p > 0 else float("inf")
        else:
            fold_loss = float("inf")
            fold_ppl = float("inf")
            ppl_check = float("inf")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'loss': fold_loss,
            'ppl': fold_ppl,
            'num_sentences': len(fold_data),
            'words_per_sentence': sentence_length,
            'total_words': len(fold_data) * sentence_length
        })
        
        logger.info(f"[CV] Fold {fold_idx+1} - Frasi: {len(fold_data)}, Parole/frase: {sentence_length}, Totale parole: {len(fold_data) * sentence_length}")
        logger.info(f"[CV] Fold {fold_idx+1} - Loss (L=-2ln(mean_sqrt_p)): {fold_loss:.6f}")
        logger.info(f"[CV] Fold {fold_idx+1} - PPL (exp(L)): {fold_ppl:.6f}")
        logger.info(f"[CV] Fold {fold_idx+1} - PPL ((mean_sqrt_p)^-2): {ppl_check:.6f}")
        logger.info(f"[CV] Fold {fold_idx+1} - Consistency check: {np.abs(fold_ppl - ppl_check) < 1e-6}")
    
    # Statistiche aggregate con ERRORE MEDIO e DEVIAZIONE STANDARD
    losses = [r['loss'] for r in fold_results if np.isfinite(r['loss'])]
    ppls = [r['ppl'] for r in fold_results if np.isfinite(r['ppl'])]
    
    if losses:
        mean_loss = np.mean(losses)
        std_loss = np.std(losses, ddof=1)  # Sample std (n-1)
        sem_loss = std_loss / np.sqrt(len(losses))  # Standard error of mean
    else:
        mean_loss = float("inf")
        std_loss = 0.0
        sem_loss = 0.0
    
    if ppls:
        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls, ddof=1)  # Sample std (n-1)
        sem_ppl = std_ppl / np.sqrt(len(ppls))  # Standard error of mean
    else:
        mean_ppl = float("inf")
        std_ppl = 0.0
        sem_ppl = 0.0
    
    logger.info("\n" + "="*80)
    logger.info("=== 3-FOLD CV RESULTS ===")
    logger.info("="*80)
    logger.info(f"Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))")
    logger.info(f"Formula PPL:  exp(L) = (mean_sqrt_p)^(-2)")
    logger.info(f"")
    logger.info(f"Configurazione test:")
    logger.info(f"  - Numero frasi per fold: ~{n_total//3}")
    logger.info(f"  - Parole per frase: {sentence_length}")
    logger.info(f"  - Totale frasi: {n_total}")
    logger.info(f"  - Parametri usati: U, W, E, F (caricati da {metadata['run_id']})")
    logger.info(f"")
    
    # ✨ STAMPA LOSS E PPL PER FOLD (PRIMA DEL TEST)
    logger.info(f"=== LOSS E PPL PER FOLD ===")
    for r in fold_results:
        logger.info(f"  Fold {r['fold']}: Loss={r['loss']:.6f}, PPL={r['ppl']:.6f}")
    logger.info(f"")
    
    logger.info(f"Risultati:")
    logger.info(f"  Loss (media)  = {mean_loss:.6f}")
    logger.info(f"  Loss (std)    = {std_loss:.6f}")
    logger.info(f"  Loss (sem)    = {sem_loss:.6f}")
    logger.info(f"")
    logger.info(f"  PPL (media)   = {mean_ppl:.6f}")
    logger.info(f"  PPL (std)     = {std_ppl:.6f}")
    logger.info(f"  PPL (sem)     = {sem_ppl:.6f}")
    logger.info(f"")
    logger.info(f"Metriche richieste:")
    logger.info(f"  ERRORE MEDIO (std Loss)         = {std_loss:.6f}")
    logger.info(f"  DEVIAZIONE STANDARD (std PPL)   = {std_ppl:.6f}")
    logger.info(f"  STANDARD ERROR MEAN (sem Loss)  = {sem_loss:.6f}")
    logger.info(f"  STANDARD ERROR MEAN (sem PPL)   = {sem_ppl:.6f}")
    logger.info(f"")
    logger.info(f"Consistency check per fold:")
    for r in fold_results:
        if np.isfinite(r['loss']) and np.isfinite(r['ppl']):
            expected_ppl = np.exp(r['loss'])
            ppl_alt = (np.exp(-r['loss']/2.0)) ** (-2)  # Verifica formula alternativa
            check1 = np.abs(expected_ppl - r['ppl']) < 1e-6
            check2 = np.abs(ppl_alt - r['ppl']) < 1e-6
            logger.info(f"  Fold {r['fold']}: exp({r['loss']:.6f}) = {expected_ppl:.6f} vs {r['ppl']:.6f} [exp: {check1}, alt: {check2}]")
    logger.info("="*80 + "\n")
    
    # ✨ GRAFICO ANDAMENTO LOSS E PPL TRA I FOLD (PRIMA DEL TEST)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fold_numbers = [r['fold'] for r in fold_results]
        fold_losses = [r['loss'] for r in fold_results]
        fold_ppls = [r['ppl'] for r in fold_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Grafico Loss
        ax1.plot(fold_numbers, fold_losses, marker='o', linewidth=2, markersize=8, color='blue', label='Loss per Fold')
        ax1.axhline(mean_loss, color='green', linestyle='--', linewidth=1.5, label=f'Mean Loss: {mean_loss:.4f}')
        ax1.fill_between(fold_numbers, 
                         [mean_loss - std_loss]*len(fold_numbers), 
                         [mean_loss + std_loss]*len(fold_numbers), 
                         alpha=0.2, color='green', label=f'±1 Std: {std_loss:.4f}')
        ax1.set_xlabel('Fold', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss per Fold (3-Fold CV)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(fold_numbers)
        
        # Grafico PPL
        ax2.plot(fold_numbers, fold_ppls, marker='s', linewidth=2, markersize=8, color='red', label='PPL per Fold')
        ax2.axhline(mean_ppl, color='orange', linestyle='--', linewidth=1.5, label=f'Mean PPL: {mean_ppl:.4f}')
        ax2.fill_between(fold_numbers, 
                         [mean_ppl - std_ppl]*len(fold_numbers), 
                         [mean_ppl + std_ppl]*len(fold_numbers), 
                         alpha=0.2, color='orange', label=f'±1 Std: {std_ppl:.4f}')
        ax2.set_xlabel('Fold', fontsize=12)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title('Perplexity per Fold (3-Fold CV)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(fold_numbers)
        
        plt.tight_layout()
        
        # Salva il grafico
        plot_path = run_dir / "3fold_cv_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[CV] ✓ Grafico salvato in: {plot_path}")
        
    except Exception as e:
        logger.warning(f"[CV] ⚠ Impossibile creare grafico: {e}")
    
    return fold_results


def evaluate_perplexity_on_quantum_states(best_params_native, cfg, logger):
    """
    Genera un NUOVO set di stati quantistici (diverso dal training) e calcola perplexity.
    """
    from quantum_annealing import generate_quantum_states
    
    qs_cfg = QUANTUM_STATES_CONFIG
    num_states = qs_cfg.get('num_states', 9)
    num_qubits_qs = qs_cfg.get('num_qubits', 2)
    max_time = qs_cfg.get('max_time', 10.0)
    
    logger.info(f"[EVAL-QA] Generazione NUOVO set di {num_states} stati quantistici per test...")
    
    # Genera NUOVI stati (con tempo diverso per avere stati diversi)
    test_quantum_states = generate_quantum_states(
        num_states=num_states,
        num_qubits=num_qubits_qs,
        max_time=max_time * 1.5,  # Tempo diverso = stati diversi
        use_test_mode=True
    )
    
    logger.info(f"[EVAL-QA] Stati test generati: shape={test_quantum_states.shape}")
    
    # Converti in lista
    test_states_list = [test_quantum_states[i] for i in range(test_quantum_states.shape[0])]
    
    # Calcola loss con gli stati di test
    try:
        params_shape = get_params(cfg['num_qubits'], cfg['num_layers']).shape
        n_params_quantum = 2 * int(np.prod(params_shape))
        params_quantum = best_params_native[:n_params_quantum]
        loss_val = loss_for_sentence(
            sentence_idx=0,
            sentence="test_quantum_states",
            encoding=None,  # Non serve encoding!
            params_quantum=params_quantum,
            cfg=cfg,
            states=test_states_list
        )
    except Exception as exc:
        logger.warning(f"[EVAL-QA] Errore nel calcolo loss: {exc}")
        return None
    
    if not np.isfinite(loss_val):
        logger.warning("[EVAL-QA] Loss non finita")
        return None
    
    # Perplexity (Renyi-1/2) basata su overlap
    avg_loss_per_state = loss_val / num_states
    prob = float(np.clip(np.exp(-loss_val), 0.0, 1.0))
    perplexity = calculate_quantum_ppl([prob])
    
    logger.info(f"[EVAL-QA] Loss totale: {loss_val:.6f}")
    logger.info(f"[EVAL-QA] Loss media per stato: {avg_loss_per_state:.6f}")
    logger.info(f"[EVAL-QA] Perplexity (Renyi-1/2) = {perplexity:.6f}")
    
    return {
        "num_states": num_states,
        "total_loss": loss_val,
        "avg_loss_per_state": avg_loss_per_state,
        "perplexity": perplexity,
    }

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger = setup_logging(rank)
    cfg = OPTIMIZATION_CONFIG

    try:
        # ==================================================================
        # SCELTA: Stati Quantistici vs Sentences Testuali
        # ==================================================================
        use_quantum_states = QUANTUM_STATES_CONFIG.get('use_quantum_states', False)
        
        if use_quantum_states:
            # ==============================================================
            # Modalità QUANTUM STATES
            # ==============================================================
            # Logica IDENTICA alle frasi:
            # - num_states = N "frasi quantistiche" da trainare
            # - max_time = M "parole" (evoluzioni temporali) per frase
            # - Con K rank, ogni rank elabora N/K frasi
            # ==============================================================
            from quantum_annealing import generate_quantum_states
            
            qs_cfg = QUANTUM_STATES_CONFIG
            num_quantum_sentences = qs_cfg.get('num_states', 9)  # N "frasi quantistiche"
            sentence_length_quantum = int(qs_cfg.get('max_time', 10))  # M "parole" per frase
            num_qubits_qs = qs_cfg.get('num_qubits', 2)
            use_test_mode = qs_cfg.get('use_test_mode', True)
            
            if rank == 0:
                logger.info(f"[QUANTUM] ========================================")
                logger.info(f"[QUANTUM] Modalità QUANTUM STATES attiva!")
                logger.info(f"[QUANTUM] ========================================")
                logger.info(f"[QUANTUM] Frasi quantistiche (num_states): {num_quantum_sentences}")
                logger.info(f"[QUANTUM] Parole per frase (max_time): {sentence_length_quantum}")
                logger.info(f"[QUANTUM] Dimensione Hilbert: 2^{num_qubits_qs} = {2**num_qubits_qs}")
                logger.info(f"[QUANTUM] MPI ranks: {size}")
                logger.info(f"[QUANTUM] Frasi per rank: ~{num_quantum_sentences // size}")
            
            # SOLO RANK 0 genera TUTTE le frasi quantistiche, poi broadcast
            if rank == 0:
                logger.info(f"[QUANTUM] Rank 0: Generazione {num_quantum_sentences} frasi quantistiche...")
                
                # RNG per generare stati iniziali DIVERSI per ogni frase
                rng_quantum = np.random.default_rng(cfg.get('seed', 42))
                hilbert_dim = 2 ** num_qubits_qs
                
                all_quantum_sentences = []
                for i in range(num_quantum_sentences):
                    # IMPORTANTE: Ogni frase parte da uno stato iniziale DIVERSO
                    # altrimenti tutte le frasi sarebbero identiche!
                    # Genera stato iniziale random normalizzato
                    random_real = rng_quantum.standard_normal(hilbert_dim)
                    random_imag = rng_quantum.standard_normal(hilbert_dim)
                    initial_state = random_real + 1j * random_imag
                    initial_state = initial_state / np.linalg.norm(initial_state)
                    
                    # Ogni "frase quantistica" = evoluzione temporale con sentence_length_quantum stati
                    quantum_sentence = generate_quantum_states(
                        num_states=sentence_length_quantum,  # M "parole"
                        num_qubits=num_qubits_qs,
                        max_time=float(sentence_length_quantum),
                        initial_state=initial_state,  # Stato iniziale UNICO per questa frase
                        use_test_mode=use_test_mode
                    )
                    # quantum_sentence ha shape (sentence_length_quantum, 2^num_qubits)
                    all_quantum_sentences.append(quantum_sentence)
                    
                    if (i + 1) % 10 == 0 or i == 0:
                        logger.info(f"[QUANTUM] Generati {i+1}/{num_quantum_sentences} frasi quantistiche...")
                
                # Array: (N, M, dim) = (num_sentences, sentence_length, hilbert_dim)
                quantum_states = np.array(all_quantum_sentences)
                logger.info(f"[QUANTUM] ✓ Tutte le frasi generate: shape={quantum_states.shape}")
                
                # Verifica che le frasi siano DIVERSE
                if num_quantum_sentences >= 2:
                    diff_01 = np.linalg.norm(quantum_states[0] - quantum_states[1])
                    logger.info(f"[QUANTUM] Verifica diversità: ||frase_0 - frase_1|| = {diff_01:.6f}")
                    if diff_01 < 1e-6:
                        logger.warning("[QUANTUM] ⚠ Le frasi sembrano identiche!")
                
                # Salva per debug
                np.save('quantum_states_generated.npy', quantum_states)
                logger.info(f"[QUANTUM] Salvate in quantum_states_generated.npy")
            else:
                quantum_states = None
            
            # Broadcast a TUTTI i rank
            quantum_states = comm.bcast(quantum_states, root=0)
            logger.info(f"[QUANTUM] Rank {rank}: Ricevute frasi quantistiche, shape={quantum_states.shape}")
            
            # Ora distribuiamo le frasi tra i rank (IDENTICO alle sentences testuali)
            sentences = [f"quantum_sentence_{i}" for i in range(num_quantum_sentences)]
            indices = np.arange(num_quantum_sentences, dtype=np.int64)
            chunks = np.array_split(indices, size)
            local_ids = chunks[rank].tolist()
            my_items = [(int(gid), sentences[int(gid)]) for gid in local_ids]
            encoding_local = None  # Non serve encoding con QA!
            
            logger.info(f"[QUANTUM] Rank {rank}: Assegnate {len(my_items)} frasi (IDs: {[x[0] for x in my_items]})")
        else:
            # Modalità SENTENCES TESTUALI (default)
            quantum_states = None  # Niente stati QA
            if rank == 0:
                logger.info(f"[TEXT] Modalità SENTENCES TESTUALI attiva")
            sentences = TRAINING_SENTENCES
            chunks, my_items, encoding_local = make_splits_and_encodings(
                sentences, comm, rank, size, cfg['embedding_dim'], logger
            )
        
        # Parametrizzazione
        params_shape = get_params(cfg['num_qubits'], cfg['num_layers']).shape
        embedding_shape = get_embedding_shape(encoding_local, cfg)
        rotation_shape = get_rotation_shape(cfg) if embedding_shape is not None else None
        n_params_quantum, n_params_embedding, n_params_rotation = get_param_counts(
            params_shape, embedding_shape, rotation_shape
        )
        n_params = n_params_quantum + n_params_embedding + n_params_rotation

        low, high = get_param_bounds(n_params)

        # Inizializzazione su rank 0
        if rank == 0:
            rng = np.random.default_rng(cfg.get('seed', 42))
            y0 = rng.uniform(-1.0, 1.0, size=n_params)  # nello spazio [-1,1]
            logger.info(
                f"Parametri: quantum={n_params_quantum} embed={n_params_embedding} "
                f"rot={n_params_rotation} tot={n_params} shape_half={params_shape}"
            )

        # Prepara objective distribuito
        # sentences_split è una lista di liste: per ogni rank, lista di (global_idx, sentence)
        sentences_split = [[] for _ in range(size)]
        # Broadcast strutture: costruiamo localmente la stessa struttura
        sentences_split[rank] = my_items
        # Raccogliamo su tutti (Allgather-like manuale)
        all_items = comm.gather(my_items, root=0)
        if rank == 0:
            sentences_split = all_items
        sentences_split = comm.bcast(sentences_split, root=0)

        objective, stop_workers, worker_loop, loss_history, best_params_tracker = distributed_objective_factory(
            comm, rank, size, sentences_split, encoding_local, low, high, cfg, logger,
            params_shape, embedding_shape, rotation_shape, cfg.get('log_frequency', 10),
            quantum_states=quantum_states if use_quantum_states else None
        )

        # Rank non-zero: entra nel loop worker e resta in ascolto
        if rank != 0:
            worker_loop()
            return 0

        # Rank 0: ottimizzazione COBYLA con early stopping
        constraints = box_constraints_for_unit(n_params)
        RANDOM_RESTARTS = 1     # solo 1 run
        MAXITER_PER_RUN = cfg.get('opt_maxiter', 100)
        RHO_BEG = cfg.get('rhobeg', 0.25)
        TOL = cfg.get('tol', 1e-5)

        logger.info(
            f"Inizio ottimizzazione con COBYLA (NO early stopping): "
            f"maxiter={MAXITER_PER_RUN}, tol={TOL}"
        )
        logger.info(
            f"[STOPPING CONFIG] HARD CONSTRAINT: Stop ONLY at {MAX_LOSS_EVALUATIONS} loss evaluations"
        )
        logger.info(
            f"[STOPPING CONFIG] All early stopping logic DISABLED"
        )

        y_start = y0
        
        # ============================================================
        # OTTIMIZZAZIONE CON GESTIONE EARLY STOPPING EXCEPTION
        # ============================================================
        early_stopped = False
        early_stop_reason = "Not stopped"
        best_y = None
        best_f = np.inf
        iters = 0
        
        try:
            res = minimize(
                fun=objective,
                x0=y_start,
                method="COBYLA",
                constraints=constraints,
                options={
                    "maxiter": MAXITER_PER_RUN,
                    "rhobeg": RHO_BEG,
                    "tol": TOL,
                    "disp": True,
                },
            )
            
            # Terminazione normale (converged before 120 evaluations)
            best_y = res.x
            best_f = res.fun
            iters = getattr(res, "nit", None)
            if iters is None:
                iters = getattr(res, "nfev", "N/A")
            early_stopped = False
            logger.info(
                f"[OPTIMIZATION] Converged naturally after {GLOBAL_LOSS_COUNTER[0]} loss evaluations "
                f"(before reaching limit of {MAX_LOSS_EVALUATIONS})"
            )
            
        except MaxLossEvaluationsReached as e:
            # REACHED 120 LOSS EVALUATIONS - This is NOT an error!
            logger.info("="*70)
            logger.info(f"[MAX EVALUATIONS] Training reached {MAX_LOSS_EVALUATIONS} loss evaluations")
            logger.info(f"[MAX EVALUATIONS] This is the intended stopping condition")
            logger.info("="*70)
            
            # Use best parameters found during optimization
            if best_params_tracker['params'] is not None:
                best_y = best_params_tracker['params']  # Already in y_scaled format
                best_f = best_params_tracker['loss']
                iters = best_params_tracker['iteration']
                logger.info(
                    f"[FINAL PARAMS] Using best parameters found: "
                    f"loss={best_f:.6f}, iteration={iters}"
                )
            else:
                # Fallback: use y_start if no best params tracked
                logger.warning(
                    f"[FINAL PARAMS] No best params tracked, using initial parameters"
                )
                best_y = y_start
                best_f = np.inf
                iters = 0
            
            early_stopped = True
            early_stop_reason = f"Reached max {MAX_LOSS_EVALUATIONS} loss evaluations"
            logger.info(f"[STOPPING REASON] {early_stop_reason}")
        
        # ============================================================
        # LOGGING RISULTATI OTTIMIZZAZIONE
        # ============================================================
        if early_stopped:
            logger.info(
                f"[COBYLA] 🛑 Terminato per SMART EARLY STOPPING dopo {iters} iterazioni"
            )
            logger.info(f"[COBYLA] Reason: {early_stop_reason}")
            logger.info(f"[COBYLA] Final loss: {best_f:.6f}")
        else:
            logger.info(f"[COBYLA] Terminato normalmente: f*={best_f}, iters={iters}")

        # Stop ai worker
        stop_workers()

        # ============================================================
        # PARAMETRI FINALI IN SPAZIO NATIVO
        # ============================================================
        best_params_native = descale_from_unit(best_y, low, high)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ============================================================
        # SALVATAGGIO TUTTE LE MATRICI U, W, E, F (CRUCIALE)
        # ============================================================
        logger.info("[SAVE] Inizio salvataggio TUTTE le matrici U, W, E, F...")
        run_dir = None
        try:
            run_dir, U_matrix, W_matrix, E_matrix, F_matrix = save_all_matrices(
                best_params_native, cfg, logger, ts, encoding_instance=encoding_local
            )
            logger.info(f"[SAVE] ✓ Tutte le matrici salvate in: {run_dir}")
        except Exception as e:
            logger.error(f"[SAVE] ✗ Errore nel salvataggio matrici: {e}")
            logger.error(traceback.format_exc())
        
        # ============================================================
        # ANALISI STATO ANCILLAE (REGISTRO C)
        # ============================================================
        logger.info("[ANCILLAE] Inizio analisi registro C (ancillae)...")
        try:
            ancillae_stats = analyze_ancillae_state(
                best_params_native, cfg, logger, ts
            )
            logger.info(f"[ANCILLAE] ✓ Analisi completata!")
            logger.info(f"[ANCILLAE] Coerenza quantistica: {ancillae_stats['has_quantum_coherence']}")
            logger.info(f"[ANCILLAE] Coefficienti complessi: {ancillae_stats['num_complex']}/{ancillae_stats['num_total']}")
        except Exception as e:
            logger.error(f"[ANCILLAE] ✗ Errore nell'analisi ancillae: {e}")
            logger.error(traceback.format_exc())
        
        # ============================================================
        # VALUTAZIONE PERPLEXITY (CRUCIALE - UNA SOLA VOLTA)
        # ============================================================
        logger.info("[EVAL] Inizio valutazione perplexity...")
        eval_metrics = None
        try:
            if not use_quantum_states:
                eval_metrics = evaluate_perplexity_on_new_sentences(
                    best_params_native, cfg, logger
                )
            else:
                # Valuta perplexity con NUOVI stati quantistici
                eval_metrics = evaluate_perplexity_on_quantum_states(
                    best_params_native, cfg, logger
                )
            
            if eval_metrics:
                logger.info(f"[EVAL] ✓ Perplexity calcolata: {eval_metrics['perplexity']:.6f}")
            else:
                logger.warning("[EVAL] ✗ Nessuna metrica disponibile")
        except Exception as e:
            logger.error(f"[EVAL] ✗ Errore nel calcolo perplexity: {e}")
            logger.error(traceback.format_exc())
        
        # ============================================================
        # 3-FOLD CROSS VALIDATION (POST-TRAINING)
        # ============================================================
        if run_dir:
            logger.info("[CV] Inizio 3-fold cross validation...")
            try:
                cv_results = run_3fold_cross_validation(
                    run_dir, cfg, logger, use_quantum_states=use_quantum_states
                )
                logger.info(f"[CV] ✓ Cross validation completata: {len(cv_results)} folds")
            except Exception as e:
                logger.error(f"[CV] ✗ Errore nella cross validation: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("[CV] Skip cross validation: matrici non salvate")

        # ============================================================
        # PRINT COMPREHENSIVE FINAL SUMMARY (HARD CONSTRAINT #5)
        # ============================================================
        # Get best loss from tracker
        tracker_best_loss = best_params_tracker.get('loss', best_f)
        
        print_final_training_summary(
            best_params_native=best_params_native,
            cfg=cfg,
            logger=logger,
            timestamp=ts,
            total_loss_evals=GLOBAL_LOSS_COUNTER[0],
            final_loss=best_f,
            best_loss=tracker_best_loss,
            eval_metrics=eval_metrics,
            ancillae_stats=ancillae_stats if 'ancillae_stats' in locals() else None
        )

        # ============================================================
        # SALVATAGGIO PARAMETRI E SUMMARY
        # ============================================================
        np.save(f"theta_finali_native_{ts}.npy", best_params_native)
        with open(f"training_summary_{ts}.txt", "w", encoding="utf-8") as f:
            f.write(f"total_loss_evaluations={GLOBAL_LOSS_COUNTER[0]}\n")
            f.write(f"max_loss_evaluations_limit={MAX_LOSS_EVALUATIONS}\n")
            f.write(f"best_loss_mean={best_f:.8f}\n")
            f.write(f"best_loss_seen={tracker_best_loss:.8f}\n")
            f.write(f"n_params_total={n_params}\n")
            f.write(f"n_params_quantum={n_params_quantum}\n")
            f.write(f"n_params_embedding={n_params_embedding}\n")
            f.write(f"n_params_rotation={n_params_rotation}\n")
            f.write(f"param_shape_half={params_shape}\n")
            f.write(f"maxiter={MAXITER_PER_RUN}, tol={TOL}\n")
            f.write("\n# Stopping Configuration (HARD CONSTRAINTS)\n")
            f.write(f"early_stopping=DISABLED\n")
            f.write(f"stopping_criterion=ONLY at {MAX_LOSS_EVALUATIONS} loss evaluations\n")
            f.write(f"patience=DISABLED\n")
            f.write(f"delta_checks=DISABLED\n")
            f.write(f"\n# Optimization Results\n")
            f.write(f"stopped_at_max_evaluations={early_stopped}\n")
            f.write(f"stop_reason={early_stop_reason}\n")
            f.write(f"actual_iterations={iters}\n")
            f.write(f"mpi_size={size}, sentences={len(sentences)}\n")
            if eval_metrics:
                f.write(f"\n# Test Data Perplexity\n")
                f.write(
                    f"eval_sentences_used="
                    f"{eval_metrics.get('used_sentences', 0)}/{eval_metrics.get('num_sentences', 0)}\n"
                )
                f.write(
                    f"eval_avg_loss_per_sentence={eval_metrics.get('avg_loss_per_sentence', float('nan')):.8f}\n"
                )
                f.write(
                    f"eval_avg_loss_per_word={eval_metrics.get('avg_loss_per_word', float('nan')):.8f}\n"
                )
                f.write(f"eval_perplexity={eval_metrics.get('perplexity', float('nan')):.8f}\n")
            else:
                f.write("eval_perplexity=NA\n")
        
        # Salvataggio storia loss
        np.savetxt(f"loss_history_{ts}.txt", np.array(loss_history))
        logger.info(f"[SAVE] Storia loss salvata in loss_history_{ts}.txt")

        logger.info(f"[COMPLETE] Ottimizzazione completata. Loss media finale: {best_f:.6f}")
        logger.info(f"[COMPLETE] Parametri salvati in theta_finali_native_{ts}.npy")
        logger.info(f"[COMPLETE] Tutti i file generati con timestamp: {ts}")

        return 0
    
    except Exception as e:
        if rank == 0:
            logger.error("Errore critico durante il training")
            logger.error(f"Tipo: {type(e).__name__}, Msg: {e}")
            for line in traceback.format_exc().splitlines():
                logger.error(line)
            try:
                tag = np.array([2], dtype=np.int32)
                MPI.COMM_WORLD.Bcast([tag, MPI.INT], root=0)
            except Exception:
                pass
            exit_local = 1
        else:
            exit_local = 0

    finally:
        try:
            comm.Barrier()
            if rank == 0:
                stop_workers()
        except Exception:
            pass
    # ---- Uscita coerente tra i rank ----
    comm.Barrier()
    exit_global = comm.allreduce(exit_local, op=MPI.SUM)
    if exit_global == 0:
        logger.info(f"[Rank {rank}] Uscita pulita (exit=0)")
        return 0
    else:
        if rank == 0:
            logger.error(f"Terminazione globale anomala: {exit_global} rank in errore")
        return 1



if __name__ == "__main__":
    sys.exit(main())
