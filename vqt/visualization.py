"""
Visualization and file I/O utilities for quantum optimization.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from qiskit.visualization import circuit_drawer
from datasets import load_dataset


def show_circuit(qc, filename="quantum_attention_circuit.png"):
    """
    Display a quantum circuit diagram.
    
    Args:
        qc (QuantumCircuit): Circuit to display
        filename (str): Filename for saving the image
    """
    circuit_drawer(qc, output="mpl", filename=filename)
    Image.open(filename).show()


def plot_loss_all(losses, best_losses, worst_losses, times=0, nqubit=16, base_name="loss_plot"):
    """
    Plot loss curves for optimization progress.
    
    Args:
        losses (list): Average losses per iteration
        best_losses (list): Best losses per iteration
        worst_losses (list): Worst losses per iteration
        times (int): Time parameter (unused)
        nqubit (int): Number of qubits (for labeling)
        base_name (str): Base filename for saving
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker="o", label="Loss", linewidth=2)
    plt.plot(best_losses, marker="s", linestyle="--", label="Best Loss", color='green')
    plt.plot(worst_losses, marker="x", linestyle=":", label="Worst Loss", color='red')
    plt.title("Loss vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_name}_qubits_and_iterations.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_loss_values_to_file(average, best, worst, filename="loss_values.txt"):
    """
    Save loss values to a text file.
    
    Args:
        average (list): Average losses per iteration
        best (list): Best losses per iteration
        worst (list): Worst losses per iteration
        filename (str): Output filename
    """
    with open(filename, "w", encoding='utf-8') as f:
        f.write("Iteration\tAverage_Loss\tBest_Loss\tWorst_Loss\n")
        for i in range(len(average)):
            line = f"{i}\t{average[i]:.6f}\t{best[i]:.6f}\t{worst[i]:.6f}\n"
            f.write(line)
    
    print(f"Values saved to: {filename}")


def save_loss_plot(avg_per_iteration, loss_best, loss_worst, num_layers, filename=None):
    """
    Save a loss plot to PNG file.
    
    Args:
        avg_per_iteration (list): Average losses per iteration
        loss_best (list): Best losses per iteration
        loss_worst (list): Worst losses per iteration
        num_layers (int): Number of layers (for filename)
        filename (str): Custom filename (optional)
    """
    if filename is None:
        filename = f"loss_plot_{num_layers}layers.png"
    
    x = range(len(avg_per_iteration))

    plt.figure(figsize=(12, 6))
    plt.plot(x, avg_per_iteration, label='Average Loss', linewidth=2, color='blue')
    plt.plot(x, loss_best, label='Best Loss', linestyle='--', color='green', linewidth=2)
    plt.plot(x, loss_worst, label='Worst Loss', linestyle='--', color='red', linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Evolution per Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {filename}")


def save_parameters(params, filename="params_best.json"):
    """
    Save optimization parameters to JSON file.
    
    Args:
        params (array or list): Parameters to save
        filename (str): Output filename
    """
    data = params.tolist() if isinstance(params, np.ndarray) else params
    print(f"Saving parameters to {filename}...")
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_parameters(filename="params_best.json"):
    """
    Load optimization parameters from JSON file.
    
    Args:
        filename (str): Input filename
        
    Returns:
        numpy.ndarray or None: Loaded parameters or None if file not found
    """
    try:
        with open(filename, "r", encoding='utf-8') as f:
            data = json.load(f)
        return np.array(data)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return None


def get_dataset_sentences(split="train", max_sentences=100):
    """
    Get sentences from PTB dataset.
    
    Args:
        split (str): Dataset split to use
        max_sentences (int): Maximum number of sentences to return
        
    Returns:
        list: List of sentences
        
    Raises:
        ValueError: If split is not available
    """
    dataset_dict = load_dataset("ptb_text_only", trust_remote_code=True)
    if split not in dataset_dict:
        raise ValueError(f"Split '{split}' not available! Valid splits: {list(dataset_dict.keys())}")

    dataset = dataset_dict[split]
    sentences = []
    for entry in dataset:
        sentence = entry["sentence"]
        if len(sentence.split()) >= 4:  # Only sentences with at least 4 words
            sentences.append(sentence)
        if len(sentences) >= max_sentences:
            break
    return sentences


def calculate_metrics_all_zero_target(probs, threshold=0.5, eps=1e-12):
    """
    Calculate various metrics for probability predictions targeting all-zero state.
    
    Args:
        probs (array): Probability values
        threshold (float): Classification threshold
        eps (float): Small epsilon for numerical stability
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 0.0, 1.0)  # Numerical safety
    y = np.ones_like(p)  # Target is always 1

    # Regression metrics (Brier = MSE for binary target)
    err = y - p
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))  # = Brier score
    rmse = float(np.sqrt(mse))
    mean_p = float(np.mean(p))

    # R^2 doesn't make sense with constant y -> NaN
    r2 = float('nan')

    # Classification metrics (positive = "|0...0>")
    acc = float(np.mean(p >= threshold))
    # BCE with y=1: -log(p)
    bce = float(-np.mean(np.log(np.clip(p, eps, 1.0))))

    # Additional statistics
    p_min, p_q25, p_med, p_q75, p_max = map(float, np.percentile(p, [0, 25, 50, 75, 100]))

    return {
        "mean_pred": mean_p,
        "MAE": mae,
        "MSE_Brier": mse,
        "RMSE": rmse,
        "R2": r2,  # ignore this (y is constant)
        f"accuracy@{threshold:.2f}": acc,
        "BCE": bce,
        "p_min": p_min,
        "p_q25": p_q25,
        "p_median": p_med,
        "p_q75": p_q75,
        "p_max": p_max,
    }