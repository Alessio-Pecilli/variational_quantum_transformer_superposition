## Variational Quantum Transformer (VQT)

We propose a **quantum implementation of self-attention**—the core mechanism underlying transformers and large language models—based on a **variational quantum circuit**.

Building on this idea, we construct a **single-layer Variational Quantum Transformer (VQT)** capable of predicting future data points from input sequences of **classical or quantum data**.

Using **ancillary qubits**, the VQT:
- Loads tokens **in superposition**
- Efficiently extracts the value of the **loss function**
- Employs a **Rényi-1/2 cross-entropy** as the training objective

### Experimental Results

In simulation, we demonstrate that the VQT can successfully learn:
- Datasets of **classical sentences**
- Sequences of **quantum states** evolved under **random non-local Pauli Hamiltonians**

### Computational Complexity

The circuit complexity of the proposed VQT scales as:

\[
\mathcal{O}\bigl(T (d^2 + M d)\bigr)
\]

compared to the complexity of a classical transformer:

\[
\mathcal{O}\bigl(T^2 d + T M d\bigr)
\]

where:
- **T** is the sequence length  
- **d** is the embedding dimension  
- **M** is the vocabulary size  

### Key Insight

This scaling suggests a **potential quantum advantage** in regimes where the **sequence length dominates the embedding size**.
