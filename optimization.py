"""
Optimization functions for quantum circuit parameters.
"""
import time
import numpy as np
from datetime import datetime
from itertools import zip_longest
from scipy.optimize import minimize
from generalized_quantum_circuits import AdaptiveQuantumCircuitFactory
from quantum_utils import get_params, wrap_angles
from quantum_circuits import get_circuit_function, create_experimental_circuit
from visualization import save_loss_plot, save_loss_values_to_file, save_parameters

def compute_loss_variant(args):
    """Valuta una variante del circuito con shift diverso."""
    params, shift_value, states_calculated, U, Z, num_layers, embedding_dim = args
    builder = AdaptiveQuantumCircuitFactory.create_circuit_builder(embedding_dim, len(states_calculated))
    
    return builder.create_generalized_circuit(
        psi=states_calculated, U=U, Z=Z,
        params_v=params, params_k=params,
        num_layers=num_layers
    )

def adam_update(params, gradients, m, v, lr, beta1, beta2, epsilon, t):
    """Step singolo di Adam."""
    m = beta1 * m + (1 - beta1) * gradients
    v = beta2 * v + (1 - beta2) * (gradients ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return np.mod(params + np.pi, 2 * np.pi) - np.pi, m, v


def aggregate_losses(loss_lists):
    """
    Aggregate losses across multiple experiments.
    
    Args:
        loss_lists (list): List of loss lists from different experiments
        
    Returns:
        tuple: (average_losses, best_losses, worst_losses)
    """
    cols = list(zip_longest(*loss_lists, fillvalue=None))
    media, best, worst = [], [], []
    for col in cols:
        vals = [v for v in col if v is not None]
        if vals:
            media.append(sum(vals) / len(vals))
            best.append(min(vals))
            worst.append(max(vals))
    return media, best, worst


def create_loss_function(circuit_function, psi, U, Z, num_layers, n_qubit, dim, param_shape, n_params):
    """
    Create a loss function for optimization.
    
    Args:
        circuit_function (function): Circuit creation function
        psi (list): Initial state unitaries
        U (list): Next word unitaries
        Z (list): Current word unitaries
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        param_shape (tuple): Shape of parameter array
        n_params (int): Number of parameters per ansatz
        
    Returns:
        function: Loss function for optimization
    """
    f_evals = [0]
    last_elapsed_log = [0.0]
    start_time = time.time()
    
    def loss_function(params_all):
        nonlocal f_evals, last_elapsed_log, start_time
        
        params_all = wrap_angles(params_all)
        f_evals[0] += 1
        
        elapsed = time.time() - start_time
        
        # Throttled logging: max once per minute
        if elapsed - last_elapsed_log[0] >= 60:
            print(f" {int(elapsed // 60)} min elapsed | f_evals={f_evals[0]} | {datetime.now().strftime('%H:%M:%S')}")
            last_elapsed_log[0] = elapsed
        
        # Extract V and K parameters
        pV = params_all[:n_params].reshape(param_shape)
        pK = params_all[n_params:2 * n_params].reshape(param_shape)
        
        # Calculate loss using the circuit function
        loss = circuit_function(psi, U, Z, pV, pK, num_layers, dim)
        
        return loss
    
    return loss_function


def optimize_parameters(max_hours, num_iterations, num_layers, psi, U, Z, n_qubit, best_params=None, dim=16, opt_maxiter=40, opt_maxfev=60):
    """
    Comprehensive parameter optimization using multiple algorithms.
    
    Args:
        max_hours (float): Maximum optimization time in hours
        num_iterations (int): Number of optimization iterations
        num_layers (int): Number of ansatz layers
        psi (list): Initial state unitaries
        U (list): Next word unitaries
        Z (list): Current word unitaries
        best_params (array): Initial parameters (optional)
        dim (int): Dimension parameter
        opt_maxiter (int): Maximum iterations for internal optimizer
        opt_maxfev (int): Maximum function evaluations for internal optimizer
        
    Returns:
        array: Optimized parameters
    """
    print("\nStarting parameter optimization...\n")
    
    # Determine circuit function based on number of words
    num_words = len(psi)
    print(f"Detected {num_words} words")
    
    try:
        circuit_function = get_circuit_function(num_words)
        print(f"Using circuit function for {num_words} words")
    except ValueError as e:
        print(f"Error: {e}")
        return best_params
    
    # Setup parameters
    print("sono in optimize_parameters, n_qubit =", n_qubit)
    param_shape = get_params(n_qubit, num_layers).shape
    n_params = int(np.prod(param_shape))
    
    start_time = time.time()
    timeout_seconds = max_hours * 3600
    
    # Storage for results
    losses_saved = []
    num_experiments = 1
    aborted_by_user = False
    
    # Create loss function
    loss_function = create_loss_function(circuit_function, psi, U, Z, num_layers, n_qubit, dim, param_shape, n_params)
    
    def save_periodically(params_flat, save_every=50):
        """Save parameters periodically during optimization."""
        try:
            save_parameters(params_flat)
        except Exception:
            pass
    
    try:
        for experiment in range(num_experiments):
            print(f"\n=== Experiment {experiment + 1}/{num_experiments} ===")
            losses_temp = []
            
            for iteration in range(num_iterations):
                print(f"\n-- External iteration {iteration + 1}/{num_iterations} --")
                
                # Initialize parameters
                if best_params is None:
                    print("sono in optimize_parameters, best_params è None e n_qubit =", n_qubit)
                    print(f"Random parameter initialization for iteration {iteration}...")
                    params_init = np.concatenate([
                        get_params(n_qubit, num_layers).flatten(),
                        get_params(n_qubit, num_layers).flatten(),
                    ])
                else:
                    print(f"Using best parameters found so far (iteration {iteration})...")
                    params_flat = np.array(best_params).flatten()
                    V0 = params_flat[:n_params].reshape(param_shape)
                    K0 = params_flat[n_params:2 * n_params].reshape(param_shape)
                    params_init = np.concatenate([V0.flatten(), K0.flatten()])
                
                try:
                    # Early stopping callback
                    def early_stop_callback(xk, *args):
                        if loss_function(xk) < 0.1:
                            print("Loss below 0.1, optimization terminated.")
                            raise StopIteration
                    
                    # Phase 1: Powell warm-up
                    try:
                        result_powell = minimize(
                            loss_function, params_init, method='Powell',
                            callback=early_stop_callback,
                            options={'maxiter': opt_maxiter, 'maxfev': opt_maxfev, 'xtol': 1e-4, 'ftol': 1e-4, 'disp': False}
                        )
                        warm_params = result_powell.x
                        print(f"Powell warm-up completed with loss: {result_powell.fun:.6f}")
                    except StopIteration:
                        warm_params = result_powell.x
                    
                    # Phase 2: L-BFGS-B refinement
                    try:
                        result_lbfgs = minimize(
                            loss_function, warm_params, method='L-BFGS-B', jac=None,
                            bounds=[(-np.pi, np.pi)] * len(warm_params),
                            callback=early_stop_callback,
                            options={'maxiter': opt_maxiter, 'maxfun': opt_maxfev, 'ftol': 1e-10, 'maxcor': 20, 'disp': False}
                        )
                        best_params = result_lbfgs.x
                        print(f"L-BFGS-B refinement completed with loss: {result_lbfgs.fun:.6f}")
                    except StopIteration:
                        best_params = result_lbfgs.x
                    
                    # Save best parameters
                    try:
                        save_parameters(best_params)
                    except Exception:
                        pass
                    
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        print("\nTimeout reached.")
                        break
                        
                except TimeoutError:
                    print("\nInterrupted: maximum time reached.")
                    break
            
            losses_saved.append(losses_temp)
    
    except KeyboardInterrupt:
        aborted_by_user = True
        print("\nInterrupted by user (Ctrl+C). Saving partial results...")
        try:
            if 'losses_temp' in locals() and len(losses_temp) > 0 and (
                    len(losses_saved) == 0 or losses_temp is not losses_saved[-1]):
                losses_saved.append(losses_temp)
        except Exception:
            pass
    
    finally:
        # Save results if any losses were recorded
        if len(losses_saved) > 0 and any(len(lst) > 0 for lst in losses_saved):
            avg_per_iteration, best_losses, worst_losses = aggregate_losses(losses_saved)
            try:
                save_loss_plot(avg_per_iteration, best_losses, worst_losses, num_layers)
                save_loss_values_to_file(avg_per_iteration, best_losses, worst_losses, "loss_results.txt")
            except Exception as e:
                print(f"Warning: error saving plots/values: {e}")
        
        # Save final parameters
        try:
            if best_params is not None:
                save_parameters(best_params)
                print("Final parameters saved to 'params_best.json'.")
        except Exception as e:
            print(f"Warning: error saving parameters: {e}")
        
        if aborted_by_user:
            print("Partial results saved. Clean exit.")
    
    # SEMPRE ritorna parametri validi, anche se casuali per modalità instant
    if best_params is None:
        print("⚡ No optimal parameters found - generating fallback parameters for instant mode")
        # Genera parametri piccoli e casuali come fallback
        print("sono in optimize_parameters, best_params è None e n_qubit =", n_qubit)
        param_shape = get_params(n_qubit, num_layers).shape
        n_params = int(np.prod(param_shape))
        fallback_params = np.concatenate([
            0.1 * np.random.randn(n_params),  # Parametri V piccoli
            0.1 * np.random.randn(n_params)   # Parametri K piccoli
        ])
        return fallback_params
    
    return best_params


def optimize_experimental_parameters(max_hours, num_iterations, num_layers, psi, U, Z, best_params=None, dim=16, opt_maxiter=40, opt_maxfev=60):
    """
    Optimization for experimental circuit with additional phase parameters.
    
    Args:
        max_hours (float): Maximum optimization time in hours
        num_iterations (int): Number of optimization iterations
        num_layers (int): Number of ansatz layers
        psi (list): Initial state unitaries
        U (list): Next word unitaries
        Z (list): Current word unitaries
        best_params (array): Initial parameters (optional)
        dim (int): Dimension parameter
        opt_maxiter (int): Maximum iterations for internal optimizer
        opt_maxfev (int): Maximum function evaluations for internal optimizer
        
    Returns:
        array: Optimized parameters
    """
    print("\nStarting experimental parameter optimization...\n")
    print("sono in optimize_experimental_parameters, n_qubit =", n_qubit)
    # Setup parameters
    n_qubit = 2
    param_shape = get_params(n_qubit, num_layers).shape
    n_params = int(np.prod(param_shape))
    
    start_time = time.time()
    timeout_seconds = max_hours * 3600
    
    # Reset best_params for experimental version
    best_params = None
    
    # Storage for results
    losses_saved = []
    num_experiments = 1
    aborted_by_user = False
    
    f_evals = [0]
    last_elapsed_log = [0.0]
    
    def save_periodically(params_flat):
        if f_evals[0] % 50 == 0:
            try:
                save_parameters(params_flat)
            except Exception:
                pass
    
    def loss_function_experimental(params_all):
        nonlocal f_evals, last_elapsed_log, start_time
        
        params_all = wrap_angles(params_all)
        f_evals[0] += 1
        
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError("Maximum optimization time reached.")
        
        # Throttled logging
        if elapsed - last_elapsed_log[0] >= 60:
            print(f" {int(elapsed // 60)} min elapsed | f_evals={f_evals[0]} | {datetime.now().strftime('%H:%M:%S')}")
            last_elapsed_log[0] = elapsed
        
        # Extract parameters
        pV = params_all[:n_params].reshape(param_shape)
        pK = params_all[n_params:2 * n_params].reshape(param_shape)
        pExtra = params_all[2 * n_params:]
        
        print("pExtra:", pExtra)
        
        # Calculate loss using experimental circuit
        loss = create_experimental_circuit(psi, U, Z, pV, pK, pExtra, num_layers, dim)
        
        save_periodically(params_all)
        return loss
    
    try:
        for experiment in range(num_experiments):
            print(f"\n=== Experiment {experiment + 1}/{num_experiments} ===")
            losses_temp = []
            
            for iteration in range(1, num_iterations):
                print(f"\n-- External iteration {iteration + 1}/{num_iterations} --")
                
                # Initialize parameters including extra phase parameters
                if best_params is None:
                    print(f"Random parameter initialization for iteration {iteration}...")
                    params_init = np.concatenate([
                        get_params(n_qubit, num_layers).flatten(),
                        get_params(n_qubit, num_layers).flatten(),
                        np.random.uniform(-np.pi, np.pi, size=2)  # 2 extra parameters
                    ])
                else:
                    print(f"Using best parameters found so far (iteration {iteration})...")
                    params_flat = np.array(best_params).flatten()
                    V0 = params_flat[:n_params].reshape(param_shape)
                    K0 = params_flat[n_params:2 * n_params].reshape(param_shape)
                    F0 = params_flat[2 * n_params:]  # Keep 1D, no reshape
                    params_init = np.concatenate([V0.flatten(), K0.flatten(), F0])
                
                try:
                    # Early stopping callback
                    def early_stop_callback(xk, *args):
                        if loss_function_experimental(xk) < 0.1:
                            print("Loss below 0.1, optimization terminated.")
                            raise StopIteration
                    
                    # Phase 1: Powell warm-up
                    try:
                        result_powell = minimize(
                            loss_function_experimental, params_init, method='Powell',
                            callback=early_stop_callback,
                            options={'maxiter': opt_maxiter, 'maxfev': opt_maxfev, 'xtol': 1e-4, 'ftol': 1e-4, 'disp': False}
                        )
                        warm_params = result_powell.x
                    except StopIteration:
                        warm_params = result_powell.x
                    
                    print("Powell warm-up from:", warm_params)
                    
                    # Phase 2: L-BFGS-B refinement
                    try:
                        result_lbfgs = minimize(
                            loss_function_experimental, warm_params, method='L-BFGS-B', jac=None,
                            bounds=[(-np.pi, np.pi)] * len(warm_params),
                            callback=early_stop_callback,
                            options={'maxiter': 100, 'maxfun': 100, 'ftol': 1e-10, 'maxcor': 20, 'disp': False}
                        )
                    except StopIteration:
                        pass
                    
                    # Phase 3: Optimize F parameters with V/K fixed
                    print("\n[PHASE 3] Optimizing F with V/K fixed")
                    x_vk = result_lbfgs.x.copy()
                    V_flat = x_vk[:n_params]
                    K_flat = x_vk[n_params:2 * n_params]
                    F_init = x_vk[2 * n_params:]
                    
                    def loss_only_f(theta):
                        full = np.concatenate([V_flat, K_flat, theta])
                        return loss_function_experimental(full)
                    
                    # Optimize F with Powell
                    result_f = minimize(
                        loss_only_f, F_init, method='Powell',
                        options={'maxiter': 30, 'maxfev': 50, 'disp': True}
                    )
                    
                    # Reconstruct total parameters
                    best_params = np.concatenate([V_flat, K_flat, result_f.x])
                    
                    print(f"F optimized: {result_f.x}, loss={result_f.fun}")
                    
                    # Save best parameters
                    try:
                        save_parameters(best_params)
                    except Exception:
                        pass
                        
                except TimeoutError:
                    print("\nInterrupted: maximum time reached.")
                    break
            
            losses_saved.append(losses_temp)
    
    except KeyboardInterrupt:
        aborted_by_user = True
        print("\nInterrupted by user (Ctrl+C). Saving partial results...")
        try:
            if 'losses_temp' in locals() and len(losses_temp) > 0 and (
                    len(losses_saved) == 0 or losses_temp is not losses_saved[-1]):
                losses_saved.append(losses_temp)
        except Exception:
            pass
    
    finally:
        # Save results
        if len(losses_saved) > 0 and any(len(lst) > 0 for lst in losses_saved):
            avg_per_iteration, best_losses, worst_losses = aggregate_losses(losses_saved)
            try:
                save_loss_plot(avg_per_iteration, best_losses, worst_losses, num_layers)
                save_loss_values_to_file(avg_per_iteration, best_losses, worst_losses, "loss_results.txt")
            except Exception as e:
                print(f"Warning: error saving plots/values: {e}")
        
        try:
            if best_params is not None:
                save_parameters(best_params)
        except Exception as e:
            print(f"Warning: error saving parameters: {e}")
        
        if aborted_by_user:
            print("Partial results saved. Clean exit.")
    
    return best_params

from multiprocessing import Pool
import numpy as np


def optimize_parameters_parallel(params, shift, states_calculated, U, Z, num_layers,
                                 embedding_dim, num_qubits, sentence_length, num_workers=None,
                                 opt_maxiter=150, opt_maxfev=50):
    from multiprocessing import Pool
    from quantum_utils import get_params
    from config import OPTIMIZATION_CONFIG
    import logging, time
    import numpy as np

    logger = logging.getLogger(__name__)

    # Setup Adam
    learning_rate = OPTIMIZATION_CONFIG.get('learning_rate', 0.001)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    best_params, best_loss = params.copy(), float('inf')

    if num_workers is None:
        num_workers = get_hpc_workers_max()

    logger.info(f"🚀 BEAST MODE OPTIMIZATION START - {num_workers} workers")

    # 🔁 Training loop
    for iteration in range(opt_maxiter):
        iter_start = time.time()

        # 1️⃣ Genera varianti parallele (es. ±shift)
        shift_values = [+shift, -shift, +2*shift, -2*shift]
        variants = [
            (params, s, states_calculated, U, Z, num_layers, embedding_dim)
            for s in shift_values
        ]

        # 2️⃣ Calcola le loss in parallelo
        with Pool(num_workers) as pool:
            losses = pool.map(compute_loss_variant, variants)

        mean_loss = np.mean(losses)
        logger.info(f"Iter {iteration:3d}: mean_loss={mean_loss:.6f}")

        # 3️⃣ Calcola gradienti medi in parallelo
        with Pool(num_workers) as pool:
            grad_results = pool.map(
                compute_gradient_batch,
                [(batch, params, shift, states_calculated, U, Z, num_layers, embedding_dim, None)
                 for batch in create_smart_batches(len(params), num_workers)]
            )

        gradients = np.zeros_like(params)
        for param_indices, grads in grad_results:
            gradients[param_indices] = grads

        # 4️⃣ Adam update (unico step)
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients ** 2)
        m_hat = m / (1 - beta1 ** (iteration + 1))
        v_hat = v / (1 - beta2 ** (iteration + 1))
        params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Clip
        params = np.mod(params + np.pi, 2*np.pi) - np.pi

        # 5️⃣ Best tracking
        if mean_loss < best_loss:
            best_loss, best_params = mean_loss, params.copy()

        if iteration % 10 == 0:
            logger.info(f"   best_loss={best_loss:.6f}  grad_norm={np.linalg.norm(gradients):.4f}")

        # Early stop
        if np.linalg.norm(gradients) < 1e-6:
            logger.info(f"🎯 Convergenza raggiunta alla iterazione {iteration}")
            break

    logger.info("✅ Ottimizzazione terminata.")
    return best_params
