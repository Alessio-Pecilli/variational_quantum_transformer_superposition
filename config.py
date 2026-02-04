"""
Configuration settings for quantum optimization experiments.
"""

from pathlib import Path

# Test-only mode settings
TEST_ONLY_CONFIG = {
    'skip_training': False,  # True = salta training e carica matrici pre-addestrate
    'matrices_dir': Path.cwd(),  # Directory con le matrici (default: cartella corrente)
}

# Optimization settings
OPTIMIZATION_CONFIG = {
    'num_iterations': 20,      # RIDOTTO per test veloce (era 150)
    'num_layers': 5,
    'max_hours': 1,            # RIDOTTO
    'embedding_dim': 4,
    'num_qubits': 4,
    'opt_maxiter': 150,        # AUMENTATO per ottimizzazione con embedding
    'opt_maxfev': 60,          # RIDOTTO
    'restarts': 1,             # NUOVO: solo 1 restart per fare presto
    'epochs': 100,
    'learning_rate':  0.001,
    'save_frequency': 10,
    'log_frequency': 10,
    'early_stop_threshold': 0.05,
    'numerical_epsilon': 1e-12
}

# Optimization algorithm settings
OPTIMIZER_CONFIG = {
    'powell': {
        'maxiter': 400,
        'maxfev': 60,
        'xtol': 1e-4,
        'ftol': 1e-4
    },
    'lbfgs': {
        'maxiter': 1000,
        'maxfun': 100,
        'ftol': 1e-10,
        'maxcor': 20
    },
    'experimental_f': {
        'maxiter': 30,
        'maxfev': 50
    }
}

# Circuit settings
CIRCUIT_CONFIG = {
    'shots': 1024 * 2,
    'max_supported_words': 16
}

# File settings
FILE_CONFIG = {
    'params_filename': 'params_best.json',
    'loss_values_filename': 'loss_results.txt',
    'loss_plot_base': 'loss_plot',
    'circuit_image': 'quantum_attention_circuit.png'
}

# Visualization settings
PLOT_CONFIG = {
    'figsize': (12, 6),
    'dpi': 300,
    'grid_alpha': 0.3,
    'colors': {
        'average': 'blue',
        'best': 'green',
        'worst': 'red'
    }
}

# Dataset settings
DATASET_CONFIG = {
    'default_split': 'train',
    'max_sentences': 100,           # quante frasi caricare
    'sentence_length': 5,           # lunghezza ESATTA in parole per ogni frase
    'dataset_name': 'ptb_text_only',
    'use_ptb': True,                # True = usa PTB dataset, False = usa frasi generate
    'random_sample': True,          # True = selezione random, False = prime N frasi
    'local_ptb_file': 'ptb_sentences.txt'  # file locale con frasi PTB pre-scaricate (una per riga)
}

# Quantum States settings (alternativa a sentences testuali)
QUANTUM_STATES_CONFIG = {
    'use_quantum_states': True,    # True = usa stati quantistici, False = usa sentences testuali
    'num_states': 100,              # N = numero di "frasi quantistiche" da scegliere dallo spazio 2^10
    'num_qubits': 4,                # Qubit del circuito QSA (escluse ancille) 
    'source_qubits': 10,             # D = 8 qubit sorgente per 2^8 = 256 dimensioni (LIMITE per kron)
    'target_qubits': 4,             # d = 4 qubit target per embedding_dim = 16
    'use_projection': True,         # NECESSARIO: proietta da 256 -> 16 dimensioni
    'use_Projector': True,          # USA la matrice P per ridurre dimensionalità  
    'max_time': 5,                 # M = 5 evoluzioni temporali per "frase"
    'use_test_mode': True,          # ABILITATO: stabilità numerica con B(t)=1, J=1
    'num_test_sets': 3,             # Numero di test set diversi da valutare
    'skip_register_analysis': True, # True = salta analisi registri (evita errore AerSimulator con gate custom)
    'batch_size': 10,              # NUOVO: processa stati in batch per ridurre memoria
    'use_sparse_matrices': True,   # NUOVO: usa matrici sparse quando possibile
    'memory_optimization': True    # NUOVO: abilita ottimizzazioni di memoria
}

# ML Classico Training Config (per MLClassicoSENTENCES e MLClassicoQUANTUMSTATES)
ML_TRAINING_CONFIG = {
    'epochs': 150,                  # 3 * 50 = 150 epoche
    'num_heads': 1,                 # Single-head attention
    'use_real_only': False,         # True = usa solo parte reale, False = Re+Im concatenati
    'gradient_clip_norm': 1.0,      # Max norm per gradient clipping
    'epsilon_sqrt': 1e-8,           # Epsilon per stabilizzare sqrt(p)
    'epsilon_log': 1e-10,           # Epsilon per stabilizzare log
    'print_frequency': 10           # Stampa loss ogni N epoche
}

# Default test sentences
DEFAULT_SENTENCES = [
    "The quick brown",
    "The quick brown",
    "The quick brown",
    "The quick brown"
    #"every day is great right",
    #"The quick brown fox jumps over the lazy dog",
    #"come and play with us today in the sunny beautiful garden now please lets go outside together", 
]

# Training sentences
TRAINING_SENTENCES = [
    "The quick brown fox jumps over the lazy dog and"# then runs away into the forest"
,
    "A bright sunny day makes everyone feel happy and"# joyful as they walk through the park"
,
"The bright morning sun shines through the open window"# warming the cold wooden floor of the room"
,
"Effective communication is the key to building strong relationships"# resolving conflicts and achieving mutual understanding in life"
]



import random

SENTENCE_LENGTH = DATASET_CONFIG['sentence_length']

SUBJECT_ADJS = [
    "quiet", "friendly", "curious", "calm"
]

SUBJECTS = [
    "girl", "scientist", "cat"
]

ADVERBS = [
    "happily", "quietly", "slowly"
]

VERBS = [
    "follows", "explores"
]

DETERMINERS2 = [
    "a", "the"
]

PLACE_ADJS = [
    "hidden"
]

PLACES = [
    "pathway"
]

# ⚠️ ENDINGS devono essere UNA PAROLA SOLA
ENDINGS = [
    "peacefully", "softly", "curiously", "calmly",
   
]


def generate_sentence():
    """Genera una frase ESATTAMENTE di 9 parole."""
    tokens = [
        "The",                                  # 1
        random.choice(SUBJECT_ADJS),            # 2
        random.choice(SUBJECTS),                # 3
        random.choice(ADVERBS),                 # 4
        random.choice(VERBS),                   # 5
    ]

    assert len(tokens) == SENTENCE_LENGTH
    sentence = " ".join(tokens)

    # Sicurezza: DOUBLE CHECK
    assert len(sentence.split()) == SENTENCE_LENGTH, sentence

    return sentence


def generate_sentences(n):
    """Genera n frasi uniche (tutte di 9 parole)."""
    sentences = set()
    while len(sentences) < n:
        s = generate_sentence()
        sentences.add(s)
    return list(sentences)


def get_ptb_sentences(split="train", max_sentences=100, sentence_length=9, random_sample=True, local_file=None):
    """
    Carica frasi dal dataset PTB (Penn Treebank).
    
    Args:
        split: 'train', 'validation', o 'test'
        max_sentences: numero massimo di frasi da caricare
        sentence_length: lunghezza ESATTA in parole per ogni frase
        random_sample: se True, seleziona frasi random; se False, prende le prime N
        local_file: percorso a file locale con frasi pre-scaricate (una per riga)
    
    Returns:
        list: Lista di frasi dal dataset PTB
    """
    import os
    
    # PRIMA prova a caricare da file locale (per HPC senza internet)
    if local_file and os.path.exists(local_file):
        print(f"[PTB] Caricamento da file locale: {local_file}")
        try:
            with open(local_file, 'r', encoding='utf-8') as f:
                all_sentences = [line.strip() for line in f if line.strip()]
            
            # Filtra per lunghezza ESATTA
            valid_sentences = [s for s in all_sentences if len(s.split()) == sentence_length]
            print(f"[PTB-LOCAL] Trovate {len(valid_sentences)} frasi con esattamente {sentence_length} parole")
            
            # Seleziona le frasi
            if random_sample and len(valid_sentences) > max_sentences:
                selected = random.sample(valid_sentences, max_sentences)
                print(f"[PTB-LOCAL] Selezionate {len(selected)} frasi RANDOM")
            else:
                selected = valid_sentences[:max_sentences]
                print(f"[PTB-LOCAL] Selezionate prime {len(selected)} frasi")
            
            return selected
        except Exception as e:
            print(f"[ERRORE] Impossibile caricare file locale: {e}")
    
    # ALTRIMENTI prova HuggingFace (richiede internet)
    try:
        from datasets import load_dataset
        print("[PTB] Tentativo download da HuggingFace...")
        dataset_dict = load_dataset("ptb_text_only", trust_remote_code=True)
        
        if split not in dataset_dict:
            available = list(dataset_dict.keys())
            raise ValueError(f"Split '{split}' non disponibile! Split validi: {available}")
        
        dataset = dataset_dict[split]
        
        # Raccoglie TUTTE le frasi con lunghezza ESATTA
        valid_sentences = []
        for entry in dataset:
            sentence = entry["sentence"]
            word_count = len(sentence.split())
            
            # Filtra per lunghezza ESATTA
            if word_count == sentence_length:
                valid_sentences.append(sentence)
        
        print(f"[PTB] Trovate {len(valid_sentences)} frasi con esattamente {sentence_length} parole")
        
        # Seleziona le frasi
        if random_sample and len(valid_sentences) > max_sentences:
            selected = random.sample(valid_sentences, max_sentences)
            print(f"[PTB] Selezionate {len(selected)} frasi RANDOM")
        else:
            selected = valid_sentences[:max_sentences]
            print(f"[PTB] Selezionate prime {len(selected)} frasi")
        
        return selected
        
    except ImportError:
        print("[ERRORE] Libreria 'datasets' non installata. Installa con: pip install datasets")
        return []
    except Exception as e:
        print(f"[ERRORE] Impossibile caricare PTB: {e}")
        return []


def get_training_sentences():
    """
    Restituisce le frasi di training in base alla configurazione.
    
    Se DATASET_CONFIG['use_ptb'] è True, carica da PTB.
    Altrimenti genera frasi sintetiche.
    """
    cfg = DATASET_CONFIG
    
    if cfg.get('use_ptb', False):
        sentences = get_ptb_sentences(
            split=cfg.get('default_split', 'train'),
            max_sentences=cfg.get('max_sentences', 100),
            sentence_length=cfg.get('sentence_length', 9),
            random_sample=cfg.get('random_sample', True),
            local_file=cfg.get('local_ptb_file', None)
        )
        if sentences:
            return sentences
        print("[WARNING] Fallback a frasi generate...")
    
    return generate_sentences(cfg.get('max_sentences', 100))


# Genera le frasi di training in base alla configurazione
TRAINING_SENTENCES = get_training_sentences()
#TRAINING_SENTENCES = generate_sentences(5)  # Test genera 5 frasi
