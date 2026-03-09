# MaximusLLM: The Hyper-Efficient Long-Context Engine

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/yousefg/MaximusLLM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper 1](https://img.shields.io/badge/ArXiv-MAXIS_Loss-B31B1B.svg)](#)
[![Paper 2](https://img.shields.io/badge/ArXiv-RandNLA_Attention-B31B1B.svg)](#)

**MaximusLLM** is a hyper-efficient paradigm for long-context LLMs: redifining model scaling and training through RandNLA Attention, MAXIS Loss, and Fisher-SVD.

By combining three novel mathematical breakthroughs, MaximusLLM achieves state-of-the-art efficiency without sacrificing convergence.

### The TL;DR
* **17.5x Faster Training** vs. standard Cross-Entropy.
* **39% Less VRAM** usage during training.
* **Infinite-Horizon Context** via Random Linear Algebra Latent Attention.
---

## Performance Benchmark: Intelligence per Second

MaximusLLM is optimized for the only metric that matters: **how much the model learns per dollar of compute.**

<img width="100%" alt="Maximus Benchmark" src="https://github.com/user-attachments/assets/f47083b5-03bc-4268-bb8a-22d287d1e3e5" />


| Metric | Standard CE (Liger) | **MAXIS (Ours)** | **Improvement** |
| :--- | :--- | :--- | :--- |
| **Speed** | 0.16 steps/sec | **2.81 steps/sec** | **17.5x Faster** |
| **Peak VRAM** | 13.66 GB | **8.37 GB** | **38.7% Reduction** |
| **Convergence** | Baseline | **~96.4% Match** | **Near Lossless** |

---

## The Maximus Architecture

### 1. MAXIS Loss
**Matryoshka Accelerated X-entropy Inference-ready Sampling**
(MAXIS) Loss transforms the model's embeddings into a hierarchical Matryoshka format as the knowledge is compressed into the 64 dimensions, allowing for much faster vector search and native RAG capabilities.

#### Strided Scouting

MAXIS utilizes a "scout" mechanism on a low-rank latent projection, using strided lookups (`[::stride]`) to identify the most difficult negative candidates across vocabulary chunks. This allows the model to "see" the entire vocabulary distribution with $\mathcal{O}(1)$ compute cost.

---

#### Dynamic Ghost Logits

To match the accuracy of exact Cross-Entropy, MAXIS injects a Ghost Logit, a mathematical "probability sink." MAXIS calculates the Dynamic Variance of the unnormalized dot products  $\mathrm{Var} \propto \|h\|^2 \cdot \mathbb{E}[\|w\|^2]$ to scale the Ghost Logit in real-time, simulating the missing mass of the unsampled vocabulary. This keeps gradients active and prevents the premature saturation common in naive sampling.

---

#### Native RAG & Matryoshka

By training in a hierarchical format, Maximus natively optimizes knowledge into a $64$-dimensional latent space. This enables "Coarse-to-Fine" retrieval, allowing for $4\times$ faster vector search and native RAG capabilities directly from the transformer's hidden states.


### 2. RandNLA Latent Attention
<img width="2190" height="590" alt="randnla_gqa_lin" src="https://github.com/user-attachments/assets/d7dd0fe1-e3a2-406b-a5a0-3bf29822f520" />

Traditional attention is $O(N^2)$. **RandNLA Attention** splits the memory into two mathematically distinct paths to achieve $O(N \cdot K)$ complexity without sacrificing discrete recall.

#### The Detail Path (Top-K)
A dynamic `importance_scorer` identifies the most critical tokens in the sequence and routes them to a high-resolution, uncompressed KV-cache. This ensures that the model maintains perfect recall for specific facts, names, and variables, bypassing the "blurry recall" of purely linear models.

#### The Sketch Path (Causal Kronecker)
The background context is compressed using **Randomized Numerical Linear Algebra**. By utilizing **Kronecker-factored matrices** ($Kron_A \otimes Kron_B$), we achieve massive receptive fields with minimal parameters. To maintain autoregressive integrity, we apply **Block-Lower-Triangular Masks** to the Kronecker factors, ensuring no information leaks from the future.

### 3. Fisher SVD Initialization

**Fisher SVD** initializes the latent spaces by leveraging the **Fisher Information Matrix**, approximated from the gradient signals of a small calibration dataset. By accumulating the squared gradients $\sum (\frac{\partial L}{\partial W})^2$, we identify which parameters are most critical to the model's semantic performance.

<img width="1589" height="985" alt="fisher_svd_vs_svd" src="https://github.com/user-attachments/assets/d6eb6b98-7ab5-4742-801d-2969236f81ba" />

As shown in the benchmarks, Fisher SVD significantly reduces semantic error compared to standard SVD, keeping the model's knowledge intact during architectural shifts.

## Quick Start
```bash
git clone https://github.com/yousef-rafat/MaximusLLM.git
python example.py --prompt "Prompt Here..."
```

## License
MaximusLLM is released under the **[MIT License](LICENSE)**. 
Feel free to use, modify, and build upon this research.
