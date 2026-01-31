# Modalità Test-Only - Guida all'Uso

## Descrizione
La modalità **Test-Only** permette di saltare completamente la fase di training e utilizzare matrici pre-addestrate per eseguire direttamente la 3-fold cross-validation.

## Configurazione

### 1. Posiziona le Matrici
Assicurati che i seguenti file siano nella **cartella principale** del progetto (stessa directory di `main_hpc.py`):

```
best_params_native.npy
E_matrix.npy
F_matrix.npy
U_matrix.npy
V_rotation.npy
W_matrix.npy
metadata.npy
```

### 2. Attiva la Modalità Test-Only
Modifica il file `config.py`:

```python
# Test-only mode settings
TEST_ONLY_CONFIG = {
    'skip_training': True,  # ⬅️ Imposta a True
    'matrices_dir': Path.cwd(),  # Directory con le matrici (default: cartella corrente)
}
```

Se le matrici sono in una directory diversa, specifica il percorso:

```python
TEST_ONLY_CONFIG = {
    'skip_training': True,
    'matrices_dir': Path('/percorso/completo/alle/matrici'),
}
```

### 3. Esegui il Programma
```bash
# Modalità standard (singolo processo)
python main_hpc.py

# Modalità MPI (multi-processo) - solo rank 0 esegue il test
mpiexec -n 4 python main_hpc.py
```

## Cosa Succede

1. ✅ Il training viene **completamente saltato**
2. ✅ Le matrici U, W, E, F vengono caricate dalla directory specificata
3. ✅ Viene eseguita la **3-fold cross-validation** usando le matrici caricate
4. ✅ I risultati vengono salvati in `results/test_only_<timestamp>/`

## Output

Il programma stamperà:

```
================================================================================
=== TEST-ONLY MODE: SKIP TRAINING ===
================================================================================
[TEST-ONLY] Caricamento matrici da: /path/to/matrices
[TEST-ONLY] ✓ Matrici caricate:
[TEST-ONLY]   - best_params_native: (...)
[TEST-ONLY]   - U_matrix: (...)
[TEST-ONLY]   - W_matrix: (...)
[TEST-ONLY]   - E_matrix: (...)
[TEST-ONLY]   - F_matrix: (...)
[TEST-ONLY]   - V_rotation: (...)
[TEST-ONLY] ✓ Matrici copiate in: results/test_only_*/matrices
[TEST-ONLY] Inizio 3-fold cross-validation...
...
[CV] === 3-FOLD CV RESULTS ===
...
================================================================================
=== TEST-ONLY MODE COMPLETATO ===
================================================================================
```

## Verifica

Controlla che **tutte le 4 matrici (U, W, E, F)** vengano usate nel test:

```
[CV] Fold X - ✅ TUTTE le 4 matrici (U, W, E, F) verranno usate nel test
```

## Disattivazione

Per tornare alla modalità normale con training:

```python
TEST_ONLY_CONFIG = {
    'skip_training': False,  # ⬅️ Imposta a False
    'matrices_dir': Path.cwd(),
}
```

## Note Importanti

- ⚠️ **Solo frasi testuali**: La modalità test-only usa `use_quantum_states=False`
- ⚠️ **MPI**: Con MPI, solo rank 0 esegue il test, gli altri rank aspettano e terminano
- ⚠️ **File mancanti**: Se manca anche solo un file, il programma termina con errore
