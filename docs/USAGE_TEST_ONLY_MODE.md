# Test-Only Mode - Usage Guide

## Description
Test-only mode skips training and uses pre-trained matrices to run the 3-fold cross-validation directly.

## Configuration

### 1. Place the Matrices
Ensure the following files are in the project root (the directory you run the command from):

```
best_params_native.npy
E_matrix.npy
F_matrix.npy
U_matrix.npy
V_rotation.npy
W_matrix.npy
metadata.npy
```

### 2. Enable Test-Only Mode
Edit `vqt/config.py`:

```python
# Test-only mode settings
TEST_ONLY_CONFIG = {
    'skip_training': True,  # set to True
    'matrices_dir': Path.cwd(),  # directory with matrices (default: current working dir)
}
```

If the matrices are in a different directory:

```python
TEST_ONLY_CONFIG = {
    'skip_training': True,
    'matrices_dir': Path('/full/path/to/matrices'),
}
```

### 3. Run
```bash
# Standard mode (single process)
python -m vqt.scripts.main_hpc

# MPI mode (multi-process) - only rank 0 runs the test
mpiexec -n 4 python -m vqt.scripts.main_hpc
```

## What Happens

1. Training is skipped.
2. Matrices U, W, E, F are loaded from the configured directory.
3. 3-fold cross-validation runs using the loaded matrices.
4. Results are saved in `results/test_only_<timestamp>/`.

## Output

```
================================================================================
=== TEST-ONLY MODE: SKIP TRAINING ===
================================================================================
[TEST-ONLY] Loading matrices from: /path/to/matrices
[TEST-ONLY] Matrices loaded:
[TEST-ONLY]   - best_params_native: (...)
[TEST-ONLY]   - U_matrix: (...)
[TEST-ONLY]   - W_matrix: (...)
[TEST-ONLY]   - E_matrix: (...)
[TEST-ONLY]   - F_matrix: (...)
[TEST-ONLY]   - V_rotation: (...)
[TEST-ONLY] Matrices copied to: results/test_only_*/matrices
[TEST-ONLY] Starting 3-fold cross-validation...
...
[CV] === 3-FOLD CV RESULTS ===
...
================================================================================
=== TEST-ONLY MODE DONE ===
================================================================================
```

## Verify

Check that all four matrices (U, W, E, F) are used during the test:

```
[CV] Fold X - OK: all 4 matrices (U, W, E, F) are used
```

## Disable

Set `skip_training` back to False:

```python
TEST_ONLY_CONFIG = {
    'skip_training': False,
    'matrices_dir': Path.cwd(),
}
```

## Notes

- Test-only mode uses `use_quantum_states=False`.
- With MPI, only rank 0 runs the test; other ranks wait and exit.
- If any required file is missing, the program exits with an error.
