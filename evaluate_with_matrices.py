#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per valutare perplexity e analizzare registri A, B, C
usando matrici unitarie U e W pre-calcolate.

PARALLELIZZAZIONE MPI:
- 100 frasi totali distribuite tra N ranks
- Ogni rank elabora 1 frase (100 ranks per 100 frasi)
- Aggregazione finale dei risultati su rank 0

Input:
- W_matrix.npy: matrice unitaria 16x16 (complex128) per ansatz V
- U_matrix.npy: matrice unitaria 16x16 (complex128) per ansatz K
- Frasi random da PTB (configurato in config.py)

Output:
- Perplexity sulle frasi di test
- Analisi stati dei registri A (target query), B (target key), C (ancillae)
"""

import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
from qiskit_aer import AerSimulator

# MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    # Fallback per esecuzione senza MPI
    class FakeComm:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def bcast(self, x, root=0): return x
        def gather(self, x, root=0): return [x]
        def Barrier(self): pass
    comm = FakeComm()
    rank = 0
    size = 1

# Import configurazioni e utilities
from config import (
    OPTIMIZATION_CONFIG,
    DATASET_CONFIG,
    QUANTUM_STATES_CONFIG,
    get_training_sentences
)
from encoding import Encoding
from generalized_quantum_circuits import (
    GeneralizedQuantumCircuitBuilder, 
    process_sentence_states
)
from layer import AnsatzBuilder
from quantum_annealing import generate_quantum_states


def setup_logging():
    """Setup logging per questo script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler (solo rank 0 per evitare duplicati)
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    # File handler (tutti i rank scrivono su file separati)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"evaluation_log_rank{rank}_{timestamp}.txt"
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f'[%(asctime)s] [RANK {rank}] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger, timestamp


def load_ansatz_parameters(w_path='W_matrix.npy', u_path='U_matrix.npy', logger=None):
    """
    Carica le matrici unitarie W e U da file .npy.
    Queste matrici rappresentano gli ansatz già costruiti con parametri ottimizzati.
    
    Args:
        w_path: percorso al file W_matrix.npy (ansatz V)
        u_path: percorso al file U_matrix.npy (ansatz K)
        logger: logger per messaggi
    
    Returns:
        tuple: (W_matrix, U_matrix) - le matrici unitarie degli ansatz
    """
    if logger:
        logger.info("[LOAD] Caricamento parametri ansatz (matrici unitarie)...")
    
    try:
        W_matrix = np.load(w_path)
        U_matrix = np.load(u_path)
        
        if logger:
            logger.info(f"[LOAD] ✓ W_matrix (ansatz V) caricata: shape={W_matrix.shape}, dtype={W_matrix.dtype}")
            logger.info(f"[LOAD] ✓ U_matrix (ansatz K) caricata: shape={U_matrix.shape}, dtype={U_matrix.dtype}")
            
        # Verifica che siano matrici 16x16 complesse
        assert W_matrix.shape == (16, 16), f"W_matrix deve essere 16x16, trovata {W_matrix.shape}"
        assert U_matrix.shape == (16, 16), f"U_matrix deve essere 16x16, trovata {U_matrix.shape}"
        assert np.iscomplexobj(W_matrix), "W_matrix deve essere complessa"
        assert np.iscomplexobj(U_matrix), "U_matrix deve essere complessa"
        
        # Verifica unitarietà
        W_dag_W = W_matrix.conj().T @ W_matrix
        U_dag_U = U_matrix.conj().T @ U_matrix
        I = np.eye(16)
        
        W_unitary = np.allclose(W_dag_W, I, atol=1e-8)
        U_unitary = np.allclose(U_dag_U, I, atol=1e-8)
        
        if logger:
            logger.info(f"[LOAD] W è unitaria: {W_unitary}")
            logger.info(f"[LOAD] U è unitaria: {U_unitary}")
            
            if not W_unitary:
                logger.warning(f"[LOAD] ⚠️ W non è perfettamente unitaria, errore max: {np.max(np.abs(W_dag_W - I)):.2e}")
            if not U_unitary:
                logger.warning(f"[LOAD] ⚠️ U non è perfettamente unitaria, errore max: {np.max(np.abs(U_dag_U - I)):.2e}")
        
        return W_matrix, U_matrix
        
    except FileNotFoundError as e:
        if logger:
            logger.error(f"[LOAD] ✗ File non trovato: {e}")
        raise
    except Exception as e:
        if logger:
            logger.error(f"[LOAD] ✗ Errore nel caricamento: {e}")
        raise


def evaluate_perplexity_with_ansatz_matrices(ansatz_V_matrix, ansatz_K_matrix, data_input, encoding, cfg, logger, use_quantum_states=False):
    """
    Calcola la perplexity usando le matrici unitarie degli ansatz V e K pre-calcolate.
    
    PARALLELIZZAZIONE MPI:
    - Ogni rank elabora SOLO i suoi dati assegnati
    - Rank 0 aggrega i risultati finali
    
    Args:
        ansatz_V_matrix: matrice unitaria 16x16 dell'ansatz V (da W_matrix.npy)
        ansatz_K_matrix: matrice unitaria 16x16 dell'ansatz K (da U_matrix.npy)
        data_input: lista di frasi (se use_quantum_states=False) o stati quantistici (se True)
        encoding: oggetto Encoding per codifica frasi (usato solo se use_quantum_states=False)
        cfg: configurazione
        logger: logger
        use_quantum_states: se True usa stati quantistici, altrimenti frasi testuali
    
    Returns:
        dict con metriche (perplexity, loss media, ecc.) - solo su rank 0
    """
    
    # ============================================================
    # DISTRIBUZIONE DATI TRA RANKS
    # ============================================================
    total_data = len(data_input)
    data_type = "stati quantistici" if use_quantum_states else "frasi"
    
    if rank == 0:
        logger.info(f"[PERPLEXITY] Totale {data_type} da elaborare: {total_data}")
        logger.info(f"[PERPLEXITY] Modalità: {'QUANTUM STATES' if use_quantum_states else 'SENTENCES'}")
        logger.info(f"[PERPLEXITY] Numero di ranks MPI: {size}")
        logger.info(f"[PERPLEXITY] Distribuzione: ~{total_data//size} {data_type} per rank")
    
    # Assegna dati ai ranks (round-robin)
    my_data = []
    for i, item in enumerate(data_input):
        if i % size == rank:
            my_data.append((i, item))  # (global_index, item)
    
    logger.info(f"[PERPLEXITY] Rank {rank} elaborerà {len(my_data)} {data_type}")
    
    # ============================================================
    # ELABORAZIONE LOCALE (solo i dati di questo rank)
    # ============================================================
    embedding_dim = cfg['embedding_dim']
    
    local_loss = 0.0
    local_words = 0
    item_losses = []
    
    for local_idx, (global_idx, item) in enumerate(my_data):
        logger.info(f"[PERPLEXITY] Rank {rank}: frase {local_idx+1}/{len(my_sentences)} (global #{global_idx}): '{sentence}'")
        
        # Codifica frase
        states, targets = encoding.encode_single(sentence)
        states_calculated, U_mats, Z_mats = process_sentence_states(states, targets=targets)
        
        sentence_length = len(states)
        local_words += sentence_length - 1
        
        # Crea builder
        builder = GeneralizedQuantumCircuitBuilder(
            embedding_dim=embedding_dim,
            sentence_length=sentence_length
        )
        
        # Costruisci circuito usando le matrici ansatz pre-calcolate
        loss = compute_loss_with_ansatz_matrices(
            builder, states_calculated, U_mats, Z_mats,
            ansatz_V_matrix, ansatz_K_matrix, logger
        )
        
        sentence_losses.append((global_idx, loss))
        local_loss += loss
        
        logger.info(f"[PERPLEXITY] Rank {rank}: frase #{global_idx} → loss={loss:.8f}")
    
    # ============================================================
    # AGGREGAZIONE RISULTATI (MPI gather su rank 0)
    # ============================================================
    logger.info(f"[PERPLEXITY] Rank {rank}: elaborazione locale completata, invio risultati...")
    
    # Gather: ogni rank invia (local_loss, local_words, sentence_losses)
    local_data = {
        'loss': local_loss,
        'words': local_words,
        'sentence_losses': sentence_losses,
        'num_sentences': len(my_sentences)
    }
    
    all_data = comm.gather(local_data, root=0)
    
    # Solo rank 0 calcola i risultati finali
    if rank == 0:
        logger.info(f"\n[PERPLEXITY] === AGGREGAZIONE RISULTATI ===")
        
        total_loss = sum(d['loss'] for d in all_data)
        total_words = sum(d['words'] for d in all_data)
        total_sentences_processed = sum(d['num_sentences'] for d in all_data)
        
        # Unisci tutte le sentence_losses
        all_sentence_losses = []
        for d in all_data:
            all_sentence_losses.extend(d['sentence_losses'])
        
        # Ordina per global_idx
        all_sentence_losses.sort(key=lambda x: x[0])
        sentence_losses_only = [loss for _, loss in all_sentence_losses]
        
        avg_loss = total_loss / total_sentences_processed if total_sentences_processed > 0 else float('inf')
        avg_loss_per_word = total_loss / total_words if total_words > 0 else float('inf')
        perplexity = np.exp(avg_loss_per_word)
        
        logger.info(f"[PERPLEXITY] Frasi valutate: {total_sentences_processed}/{total_sentences}")
        logger.info(f"[PERPLEXITY] Parole totali: {total_words}")
        logger.info(f"[PERPLEXITY] Loss totale: {total_loss:.8f}")
        logger.info(f"[PERPLEXITY] Loss media per frase: {avg_loss:.8f}")
        logger.info(f"[PERPLEXITY] Loss media per parola: {avg_loss_per_word:.8f}")
        logger.info(f"[PERPLEXITY] Perplexity: {perplexity:.8f}")
        
        return {
            'perplexity': perplexity,
            'total_loss': total_loss,
            'avg_loss_per_sentence': avg_loss,
            'avg_loss_per_word': avg_loss_per_word,
            'num_sentences': total_sentences_processed,
            'num_words': total_words,
            'sentence_losses': sentence_losses_only
        }
    else:
        # Altri ranks ritornano None
        return None


def compute_loss_with_ansatz_matrices(builder, psi, U, Z, ansatz_V_matrix, ansatz_K_matrix, logger):
    """
    Costruisce il circuito quantistico usando le matrici unitarie degli ansatz pre-calcolate.
    Questo è IDENTICO al circuito del main, ma invece di costruire gli ansatz dai parametri,
    usiamo direttamente le matrici unitarie salvate (che rappresentano gli ansatz ottimizzati).
    
    Args:
        builder: GeneralizedQuantumCircuitBuilder
        psi: stati iniziali processati
        U: matrici U per predizione
        Z: matrici Z per parola corrente  
        ansatz_V_matrix: matrice unitaria ansatz V (16x16)
        ansatz_K_matrix: matrice unitaria ansatz K (16x16)
        logger: logger
    
    Returns:
        float: loss calcolata
    """
    from qiskit.circuit.library import UnitaryGate
    from quantum_utils import calculate_loss_from_statevector
    
    # Crea circuito quantistico - IDENTICO al main
    qc = QuantumCircuit(builder.n_total_qubits, builder.n_total_qubits)
    
    # 1. Inizializza control qubits in superposition
    builder._initialize_control_superposition(qc)
    
    # 2. Applica controlled unitaries per stati iniziali
    builder._apply_controlled_initial_states(qc, psi)
    
    # 3. Applica gli ansatz V e K usando le matrici pre-calcolate
    #    Nel main usa: qc.compose(ansatz_v.get_unitary("V"), builder.target_indices, inplace=True)
    #    Qui usiamo direttamente le matrici salvate che SONO gli ansatz ottimizzati
    V_gate = UnitaryGate(ansatz_V_matrix, label='V')
    K_gate = UnitaryGate(ansatz_K_matrix, label='K')
    
    qc.compose(V_gate, builder.target_indices, inplace=True)
    qc.compose(K_gate, builder.target_indices, inplace=True)
    
    # 4. Applica controlled prediction unitaries
    builder._apply_controlled_predictions(qc, U, Z)
    
    # 5. Hadamard finali sui control qubits
    builder._apply_final_hadamards(qc)
    
    # 6. Calcola loss - IDENTICO al main
    #    Nel main usa: loss = calculate_loss_from_statevector(qc)
    loss = calculate_loss_from_statevector(qc)
    
    return loss


def analyze_quantum_registers(ansatz_V_matrix, ansatz_K_matrix, item, encoding, cfg, logger, timestamp, use_quantum_states=False):
    """
    Analizza gli stati dei registri A, B, C dopo l'esecuzione del circuito.
    Usa le matrici unitarie degli ansatz pre-calcolate.
    
    Args:
        ansatz_V_matrix: matrice unitaria ansatz V
        ansatz_K_matrix: matrice unitaria ansatz K
        item: frase testuale o stato quantistico da analizzare
        encoding: oggetto Encoding (None se use_quantum_states=True)
        cfg: configurazione
        logger: logger
        timestamp: timestamp per salvare file
        use_quantum_states: se True, item è uno stato quantistico
    
    Returns:
        dict con statistiche dei registri
    """
    if use_quantum_states:
        logger.info(f"[REGISTERS] Analisi registri quantistici per: STATO QUANTISTICO")
        states = [item]
        targets = None
        item_label = "Quantum State"
    else:
        logger.info(f"[REGISTERS] Analisi registri quantistici per: '{item}'")
        states, targets = encoding.encode_single(item)
        item_label = item
    
    embedding_dim = cfg['embedding_dim']
    
    # Processa stati
    states_calculated, U_mats, Z_mats = process_sentence_states(states, targets=targets)
    sentence_length = len(states)
    
    # Crea builder
    builder = GeneralizedQuantumCircuitBuilder(
        embedding_dim=embedding_dim,
        sentence_length=sentence_length
    )
    
    # Costruisci circuito con ansatz pre-calcolati
    qc = QuantumCircuit(builder.n_total_qubits, builder.n_total_qubits)
    
    from qiskit.circuit.library import UnitaryGate
    
    # 1. Inizializza control qubits
    builder._initialize_control_superposition(qc)
    
    # 2. Stati iniziali
    builder._apply_controlled_initial_states(qc, states_calculated)
    
    # 3. Applica ansatz V e K usando le matrici pre-calcolate
    V_gate = UnitaryGate(ansatz_V_matrix, label='V')
    K_gate = UnitaryGate(ansatz_K_matrix, label='K')
    
    qc.append(V_gate, builder.target_indices)
    qc.append(K_gate, builder.target_indices)
    
    # 4. Prediction unitaries
    builder._apply_controlled_predictions(qc, U_mats, Z_mats)
    
    # 5. Hadamard finali
    builder._apply_final_hadamards(qc)
    
    # NON aggiungiamo misure, vogliamo lo statevector completo
    
    # Simula
    backend = AerSimulator(method="statevector")
    job = backend.run(qc)
    result = job.result()
    statevector = result.get_statevector()
    
    logger.info(f"[REGISTERS] Statevector totale: dim={len(statevector)}")
    
    # Converti in density matrix
    density_matrix = DensityMatrix(statevector)
    
    # Calcola dimensioni
    n_target_qubits = builder.n_target_qubits
    n_control_qubits = builder.n_control_qubits
    
    # I target qubits sono divisi in A e B (metà e metà)
    n_qubits_A = n_target_qubits // 2
    n_qubits_B = n_target_qubits - n_qubits_A
    
    # Indici dei registri
    indices_A = list(range(0, n_qubits_A))
    indices_B = list(range(n_qubits_A, n_target_qubits))
    indices_C = builder.control_indices
    
    logger.info(f"[REGISTERS] Configurazione registri:")
    logger.info(f"[REGISTERS]   - Registro A (query): {n_qubits_A} qubits, indices={indices_A}")
    logger.info(f"[REGISTERS]   - Registro B (key): {n_qubits_B} qubits, indices={indices_B}")
    logger.info(f"[REGISTERS]   - Registro C (ancillae): {n_control_qubits} qubits, indices={indices_C}")
    
    # REGISTRO C (ancillae): trace out registri A e B (target)
    logger.info(f"\n[REGISTERS] === Analisi REGISTRO C (ancillae, {n_control_qubits} qubits) ===")
    ancillae_dm = partial_trace(density_matrix, builder.target_indices)
    ancillae_state = np.array(ancillae_dm.data.diagonal())
    stats_C = analyze_register_coefficients(ancillae_state, "C (ancillae)", logger)
    
    # REGISTRO A (query): trace out registri B e C
    logger.info(f"\n[REGISTERS] === Analisi REGISTRO A (query, {n_qubits_A} qubits) ===")
    indices_to_trace_out_for_A = indices_B + indices_C
    register_A_dm = partial_trace(density_matrix, indices_to_trace_out_for_A)
    register_A_state = np.array(register_A_dm.data.diagonal())
    stats_A = analyze_register_coefficients(register_A_state, "A (query)", logger)
    
    # REGISTRO B (key): trace out registri A e C
    logger.info(f"\n[REGISTERS] === Analisi REGISTRO B (key, {n_qubits_B} qubits) ===")
    indices_to_trace_out_for_B = indices_A + indices_C
    register_B_dm = partial_trace(density_matrix, indices_to_trace_out_for_B)
    register_B_state = np.array(register_B_dm.data.diagonal())
    stats_B = analyze_register_coefficients(register_B_state, "B (key)", logger)
    
    # REGISTRO A+B COMBINATO: trace out solo registro C
    logger.info(f"\n[REGISTERS] === Analisi REGISTRO A+B (target combinato, {n_qubits_A + n_qubits_B} qubits) ===")
    target_dm = partial_trace(density_matrix, builder.control_indices)
    target_state = np.array(target_dm.data.diagonal())
    stats_AB = analyze_register_coefficients(target_state, "A+B (target combinato)", logger)
    
    # Salva risultati
    output_file = f"register_analysis_{timestamp}.txt"
    save_register_analysis(output_file, stats_A, stats_B, stats_C, stats_AB, item_label, 
                          n_qubits_A, n_qubits_B, n_control_qubits, logger)
    
    # Salva array numpy
    np.save(f"register_A_state_{timestamp}.npy", register_A_state)
    np.save(f"register_B_state_{timestamp}.npy", register_B_state)
    np.save(f"register_C_state_{timestamp}.npy", ancillae_state)
    np.save(f"register_AB_state_{timestamp}.npy", target_state)
    
    logger.info(f"\n[REGISTERS] ✓ Stati salvati:")
    logger.info(f"[REGISTERS]   - register_A_state_{timestamp}.npy")
    logger.info(f"[REGISTERS]   - register_B_state_{timestamp}.npy")
    logger.info(f"[REGISTERS]   - register_C_state_{timestamp}.npy")
    logger.info(f"[REGISTERS]   - register_AB_state_{timestamp}.npy")
    
    return {
        'register_A': stats_A,
        'register_B': stats_B,
        'register_C': stats_C,
        'register_AB': stats_AB,
        'n_qubits_A': n_qubits_A,
        'n_qubits_B': n_qubits_B,
        'n_control_qubits': n_control_qubits,
        'n_target_qubits': n_target_qubits
    }


def analyze_register_coefficients(state, register_name, logger):
    """
    Analizza i coefficienti di uno stato quantistico.
    
    Returns:
        dict con statistiche
    """
    threshold = 1e-10
    
    magnitudes = np.abs(state)
    phases = np.angle(state)
    real_parts = state.real
    imag_parts = state.imag
    
    # Coefficienti reali vs complessi
    is_real = np.abs(imag_parts) < threshold
    num_real = np.sum(is_real)
    num_complex = len(state) - num_real
    
    # Coefficienti significativi
    significant = magnitudes > threshold
    num_significant = np.sum(significant)
    
    # Entropia di Von Neumann (approssimata dalla diagonale)
    probs = magnitudes**2
    probs = probs[probs > threshold]
    if len(probs) > 0:
        probs = probs / np.sum(probs)  # Normalizza
        entropy = -np.sum(probs * np.log2(probs + 1e-15))
    else:
        entropy = 0.0
    
    logger.info(f"[{register_name}] Coefficienti totali: {len(state)}")
    logger.info(f"[{register_name}] Coefficienti significativi: {num_significant}")
    logger.info(f"[{register_name}] Coefficienti REALI: {num_real} ({100*num_real/len(state):.1f}%)")
    logger.info(f"[{register_name}] Coefficienti COMPLESSI: {num_complex} ({100*num_complex/len(state):.1f}%)")
    logger.info(f"[{register_name}] Entropia (approx): {entropy:.4f} bits")
    logger.info(f"[{register_name}] Coerenza quantistica: {num_complex > 0}")
    
    return {
        'num_total': len(state),
        'num_significant': num_significant,
        'num_real': num_real,
        'num_complex': num_complex,
        'has_quantum_coherence': num_complex > 0,
        'entropy': entropy,
        'magnitudes': magnitudes,
        'phases': phases,
        'real_parts': real_parts,
        'imag_parts': imag_parts
    }


def save_register_analysis(filename, stats_A, stats_B, stats_C, stats_AB, sentence, 
                           n_qubits_A, n_qubits_B, n_control, logger):
    """Salva analisi dettagliata dei registri su file."""
    
    n_total = n_qubits_A + n_qubits_B + n_control
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ANALISI REGISTRI QUANTISTICI A, B, C\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Frase analizzata: '{sentence}'\n\n")
        
        f.write(f"Configurazione circuito:\n")
        f.write(f"  - Qubits totali: {n_total}\n")
        f.write(f"  - Registro A (query): {n_qubits_A} qubits → dim Hilbert = {2**n_qubits_A}\n")
        f.write(f"  - Registro B (key): {n_qubits_B} qubits → dim Hilbert = {2**n_qubits_B}\n")
        f.write(f"  - Registro C (ancillae): {n_control} qubits → dim Hilbert = {2**n_control}\n")
        f.write(f"  - A+B combinati: {n_qubits_A + n_qubits_B} qubits → dim Hilbert = {2**(n_qubits_A + n_qubits_B)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("REGISTRO A (QUERY)\n")
        f.write("=" * 80 + "\n\n")
        write_register_stats(f, stats_A)
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("REGISTRO B (KEY)\n")
        f.write("=" * 80 + "\n\n")
        write_register_stats(f, stats_B)
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("REGISTRO C (ANCILLAE)\n")
        f.write("=" * 80 + "\n\n")
        write_register_stats(f, stats_C)
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("REGISTRI A+B COMBINATI (TARGET)\n")
        f.write("=" * 80 + "\n\n")
        write_register_stats(f, stats_AB)
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONFRONTO TRA REGISTRI\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Coerenza quantistica:\n")
        f.write(f"  - Registro A: {'SÌ' if stats_A['has_quantum_coherence'] else 'NO'}\n")
        f.write(f"  - Registro B: {'SÌ' if stats_B['has_quantum_coherence'] else 'NO'}\n")
        f.write(f"  - Registro C: {'SÌ' if stats_C['has_quantum_coherence'] else 'NO'}\n")
        f.write(f"  - A+B: {'SÌ' if stats_AB['has_quantum_coherence'] else 'NO'}\n\n")
        
        f.write("Entropia (bits):\n")
        f.write(f"  - Registro A: {stats_A['entropy']:.4f}\n")
        f.write(f"  - Registro B: {stats_B['entropy']:.4f}\n")
        f.write(f"  - Registro C: {stats_C['entropy']:.4f}\n")
        f.write(f"  - A+B: {stats_AB['entropy']:.4f}\n\n")
        
        f.write("Percentuale coefficienti complessi:\n")
        f.write(f"  - Registro A: {100*stats_A['num_complex']/stats_A['num_total']:.1f}%\n")
        f.write(f"  - Registro B: {100*stats_B['num_complex']/stats_B['num_total']:.1f}%\n")
        f.write(f"  - Registro C: {100*stats_C['num_complex']/stats_C['num_total']:.1f}%\n")
        f.write(f"  - A+B: {100*stats_AB['num_complex']/stats_AB['num_total']:.1f}%\n")
    
    logger.info(f"[REGISTERS] Analisi completa salvata in: {filename}")


def write_register_stats(f, stats):
    """Scrive statistiche registro su file."""
    
    f.write(f"Coefficienti totali: {stats['num_total']}\n")
    f.write(f"Coefficienti significativi: {stats['num_significant']}\n")
    f.write(f"Coefficienti REALI: {stats['num_real']} ({100*stats['num_real']/stats['num_total']:.1f}%)\n")
    f.write(f"Coefficienti COMPLESSI: {stats['num_complex']} ({100*stats['num_complex']/stats['num_total']:.1f}%)\n")
    f.write(f"Entropia: {stats['entropy']:.4f} bits\n")
    f.write(f"Coerenza quantistica: {'SÌ' if stats['has_quantum_coherence'] else 'NO'}\n\n")
    
    # Top 10 coefficienti per magnitudine
    mags = stats['magnitudes']
    phases = stats['phases']
    real_parts = stats['real_parts']
    imag_parts = stats['imag_parts']
    
    sorted_indices = np.argsort(mags)[::-1]
    
    f.write("Top 10 coefficienti (per magnitudine):\n\n")
    f.write(f"{'Idx':<6} {'|c|':<12} {'Fase(rad)':<12} {'Fase(°)':<10} {'Re':<14} {'Im':<14}\n")
    f.write("-" * 80 + "\n")
    
    for idx in sorted_indices[:10]:
        mag = mags[idx]
        phase = phases[idx]
        re = real_parts[idx]
        im = imag_parts[idx]
        phase_deg = phase * 180 / np.pi
        
        f.write(f"{idx:<6} {mag:<12.6e} {phase:<12.6f} {phase_deg:<10.2f} {re:<14.6e} {im:<14.6e}\n")


def main():
    """Main function con parallelizzazione MPI."""
    logger, timestamp = setup_logging()
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("VALUTAZIONE CON MATRICI UNITARIE PRE-CALCOLATE (MPI)")
        logger.info("=" * 80)
        logger.info(f"[MPI] Numero di ranks: {size}")
    
    # Configura percorsi file
    w_file = 'W_matrix.npy'
    u_file = 'U_matrix.npy'
    
    # Tutti i ranks caricano le matrici (necessarie per elaborazione locale)
    try:
        ansatz_V_matrix, ansatz_K_matrix = load_ansatz_parameters(w_file, u_file, logger)
    except Exception as e:
        logger.error(f"Impossibile caricare le matrici: {e}")
        return
    
    # Carica configurazione (tutti i ranks)
    cfg = OPTIMIZATION_CONFIG
    dataset_cfg = DATASET_CONFIG
    
    if rank == 0:
        logger.info(f"\n[CONFIG] Configurazione:")
        logger.info(f"[CONFIG] Embedding dim: {cfg['embedding_dim']}")
        logger.info(f"[CONFIG] Num qubits: {cfg['num_qubits']}")
        logger.info(f"[CONFIG] Num layers: {cfg['num_layers']}")
        logger.info(f"[CONFIG] Max sentences: {dataset_cfg['max_sentences']}")
        logger.info(f"[CONFIG] Sentence length: {dataset_cfg['sentence_length']}")
    
    # Rank 0 carica le frasi, poi broadcast a tutti
    if rank == 0:
        logger.info(f"\n[DATA] Caricamento frasi da PTB...")
        sentences = get_training_sentences()
        logger.info(f"[DATA] ✓ Caricate {len(sentences)} frasi")
    else:
        sentences = None
    
    # Broadcast frasi a tutti i ranks
    sentences = comm.bcast(sentences, root=0)
    
    # Tutti i ranks inizializzano encoding (necessario per codifica locale)
    encoding = Encoding(sentences=sentences, embeddingDim=cfg['embedding_dim'])
    logger.info(f"[RANK {rank}] Encoding inizializzato")
    
    # Sincronizza prima di iniziare elaborazione
    comm.Barrier()
    
    # ============================================================
    # FASE 1: VALUTA PERPLEXITY (distribuito tra ranks)
    # ============================================================
    if rank == 0:
        logger.info(f"\n{'=' * 80}")
        logger.info("FASE 1: VALUTAZIONE PERPLEXITY (DISTRIBUITA)")
        logger.info(f"{'=' * 80}\n")
    
    perplexity_results = evaluate_perplexity_with_ansatz_matrices(
        ansatz_V_matrix, ansatz_K_matrix, sentences, encoding, cfg, logger
    )
    
    # Sincronizza dopo perplexity
    comm.Barrier()
    
    # ============================================================
    # FASE 2: ANALISI REGISTRI (solo rank 0)
    # ============================================================
    if rank == 0:
        logger.info(f"\n{'=' * 80}")
        logger.info("FASE 2: ANALISI REGISTRI QUANTISTICI (RANK 0)")
        logger.info(f"{'=' * 80}\n")
        
        # Usa la prima frase come esempio
        example_sentence = sentences[0]
        register_stats = analyze_quantum_registers(
            ansatz_V_matrix, ansatz_K_matrix, example_sentence, encoding, cfg, logger, timestamp
        )
    
        # ============================================================
        # FASE 3: SALVA SUMMARY FINALE (solo rank 0)
        # ============================================================
        logger.info(f"\n{'=' * 80}")
        logger.info("SUMMARY FINALE")
        logger.info(f"{'=' * 80}\n")
        
        summary_file = f"evaluation_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION SUMMARY (MPI)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"MPI ranks utilizzati: {size}\n")
            f.write(f"Matrici usate:\n")
            f.write(f"  - W_matrix: {w_file}\n")
            f.write(f"  - U_matrix: {u_file}\n\n")
            
            f.write("PERPLEXITY:\n")
            f.write(f"  - Perplexity: {perplexity_results['perplexity']:.8f}\n")
            f.write(f"  - Frasi valutate: {perplexity_results['num_sentences']}\n")
            f.write(f"  - Parole totali: {perplexity_results['num_words']}\n")
            f.write(f"  - Loss media/frase: {perplexity_results['avg_loss_per_sentence']:.8f}\n")
            f.write(f"  - Loss media/parola: {perplexity_results['avg_loss_per_word']:.8f}\n\n")
            
            f.write("REGISTRI QUANTISTICI:\n\n")
            
            f.write(f"  Registro A (query, {register_stats['n_qubits_A']} qubits):\n")
            f.write(f"    - Coerenza quantistica: {register_stats['register_A']['has_quantum_coherence']}\n")
            f.write(f"    - Coefficienti complessi: {register_stats['register_A']['num_complex']}/{register_stats['register_A']['num_total']}\n")
            f.write(f"    - Entropia: {register_stats['register_A']['entropy']:.4f} bits\n\n")
            
            f.write(f"  Registro B (key, {register_stats['n_qubits_B']} qubits):\n")
            f.write(f"    - Coerenza quantistica: {register_stats['register_B']['has_quantum_coherence']}\n")
            f.write(f"    - Coefficienti complessi: {register_stats['register_B']['num_complex']}/{register_stats['register_B']['num_total']}\n")
            f.write(f"    - Entropia: {register_stats['register_B']['entropy']:.4f} bits\n\n")
            
            f.write(f"  Registro C (ancillae, {register_stats['n_control_qubits']} qubits):\n")
            f.write(f"    - Coerenza quantistica: {register_stats['register_C']['has_quantum_coherence']}\n")
            f.write(f"    - Coefficienti complessi: {register_stats['register_C']['num_complex']}/{register_stats['register_C']['num_total']}\n")
            f.write(f"    - Entropia: {register_stats['register_C']['entropy']:.4f} bits\n\n")
            
            f.write(f"  Registri A+B combinati (target, {register_stats['n_target_qubits']} qubits):\n")
            f.write(f"    - Coerenza quantistica: {register_stats['register_AB']['has_quantum_coherence']}\n")
            f.write(f"    - Coefficienti complessi: {register_stats['register_AB']['num_complex']}/{register_stats['register_AB']['num_total']}\n")
            f.write(f"    - Entropia: {register_stats['register_AB']['entropy']:.4f} bits\n")
        
        logger.info(f"✓ Summary salvato in: {summary_file}")
        logger.info(f"\n{'=' * 80}")
        logger.info("VALUTAZIONE COMPLETATA!")
        logger.info(f"{'=' * 80}\n")
    
    # Sincronizzazione finale
    comm.Barrier()
    logger.info(f"[RANK {rank}] Terminato")
    
    # ==================================================================
    # SCELTA: Stati Quantistici vs Sentences Testuali
    # ==================================================================
    qs_cfg = QUANTUM_STATES_CONFIG
    use_quantum_states = qs_cfg.get('use_quantum_states', False)
    
    if use_quantum_states:
        # ============================================================
        # MODALITÀ QUANTUM STATES
        # ============================================================
        if rank == 0:
            logger.info(f"\n[DATA] Modalità: QUANTUM STATES")
            logger.info(f"[DATA] Generazione nuovi stati quantistici per test...")
        
        num_states = qs_cfg.get('num_states', 9)
        num_qubits_qs = qs_cfg.get('num_qubits', 2)
        max_time = qs_cfg.get('max_time', 10.0)
        use_test_mode = qs_cfg.get('use_test_mode', True)
        
        if rank == 0:
            logger.info(f"[DATA] Configurazione:")
            logger.info(f"[DATA]   - Num stati: {num_states}")
            logger.info(f"[DATA]   - Num qubits: {num_qubits_qs}")
            logger.info(f"[DATA]   - Max time: {max_time}")
            logger.info(f"[DATA]   - Test mode: {use_test_mode}")
        
        # Genera NUOVI stati quantistici (diversi dal training)
        quantum_states = generate_quantum_states(
            num_states=num_states,
            num_qubits=num_qubits_qs,
            max_time=max_time,
            use_test_mode=use_test_mode
        )
        
        if rank == 0:
            logger.info(f"[DATA] ✓ Generati {len(quantum_states)} stati quantistici")
        
        data_input = quantum_states
        encoding = None  # Non serve encoding per stati quantistici
        example_item = quantum_states[0]
        
    else:
        # ============================================================
        # MODALITÀ SENTENCES
        # ============================================================
        if rank == 0:
            logger.info(f"\n[DATA] Modalità: SENTENCES")
            logger.info(f"[DATA] Caricamento frasi da PTB...")
        
        sentences = get_training_sentences()
        
        if rank == 0:
            logger.info(f"[DATA] ✓ Caricate {len(sentences)} frasi")
        
        # Inizializza encoding
        encoding = Encoding(sentences=sentences, embeddingDim=cfg['embedding_dim'])
        data_input = sentences
        example_item = sentences[0]
    
    # 1. VALUTA PERPLEXITY
    if rank == 0:
        logger.info(f"\n{'=' * 80}")
        logger.info("FASE 1: VALUTAZIONE PERPLEXITY")
        logger.info(f"{'=' * 80}\n")
    
    perplexity_results = evaluate_perplexity_with_ansatz_matrices(
        ansatz_V_matrix, ansatz_K_matrix, data_input, encoding, cfg, logger, use_quantum_states
    )
    
    # 2. ANALISI REGISTRI (su un esempio)
    if rank == 0:
        logger.info(f"\n{'=' * 80}")
        logger.info("FASE 2: ANALISI REGISTRI QUANTISTICI")
        logger.info(f"{'=' * 80}\n")
    
    register_stats = analyze_quantum_registers(
        ansatz_V_matrix, ansatz_K_matrix, example_item, encoding, cfg, logger, timestamp, use_quantum_states
    )
    
    # 3. SALVA SUMMARY FINALE
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY FINALE")
    logger.info(f"{'=' * 80}\n")
    
    summary_file = f"evaluation_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Matrici usate:\n")
        f.write(f"  - W_matrix: {w_file}\n")
        f.write(f"  - U_matrix: {u_file}\n\n")
        
        f.write("PERPLEXITY:\n")
        f.write(f"  - Perplexity: {perplexity_results['perplexity']:.8f}\n")
        f.write(f"  - Frasi valutate: {perplexity_results['num_sentences']}\n")
        f.write(f"  - Parole totali: {perplexity_results['num_words']}\n")
        f.write(f"  - Loss media/frase: {perplexity_results['avg_loss_per_sentence']:.8f}\n")
        f.write(f"  - Loss media/parola: {perplexity_results['avg_loss_per_word']:.8f}\n\n")
        
        f.write("REGISTRI QUANTISTICI:\n\n")
        
        f.write(f"  Registro A (query, {register_stats['n_qubits_A']} qubits):\n")
        f.write(f"    - Coerenza quantistica: {register_stats['register_A']['has_quantum_coherence']}\n")
        f.write(f"    - Coefficienti complessi: {register_stats['register_A']['num_complex']}/{register_stats['register_A']['num_total']}\n")
        f.write(f"    - Entropia: {register_stats['register_A']['entropy']:.4f} bits\n\n")
        
        f.write(f"  Registro B (key, {register_stats['n_qubits_B']} qubits):\n")
        f.write(f"    - Coerenza quantistica: {register_stats['register_B']['has_quantum_coherence']}\n")
        f.write(f"    - Coefficienti complessi: {register_stats['register_B']['num_complex']}/{register_stats['register_B']['num_total']}\n")
        f.write(f"    - Entropia: {register_stats['register_B']['entropy']:.4f} bits\n\n")
        
        f.write(f"  Registro C (ancillae, {register_stats['n_control_qubits']} qubits):\n")
        f.write(f"    - Coerenza quantistica: {register_stats['register_C']['has_quantum_coherence']}\n")
        f.write(f"    - Coefficienti complessi: {register_stats['register_C']['num_complex']}/{register_stats['register_C']['num_total']}\n")
        f.write(f"    - Entropia: {register_stats['register_C']['entropy']:.4f} bits\n\n")
        
        f.write(f"  Registri A+B combinati (target, {register_stats['n_target_qubits']} qubits):\n")
        f.write(f"    - Coerenza quantistica: {register_stats['register_AB']['has_quantum_coherence']}\n")
        f.write(f"    - Coefficienti complessi: {register_stats['register_AB']['num_complex']}/{register_stats['register_AB']['num_total']}\n")
        f.write(f"    - Entropia: {register_stats['register_AB']['entropy']:.4f} bits\n")
    
    logger.info(f"✓ Summary salvato in: {summary_file}")
    logger.info(f"\n{'=' * 80}")
    logger.info("VALUTAZIONE COMPLETATA!")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
