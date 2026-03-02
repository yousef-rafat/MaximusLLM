# MaximusLLM: The Hyper-Efficient Long-Context Engine

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/yousefg/MaximusLLM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper 1](https://img.shields.io/badge/ArXiv-MAXIS_Loss-B31B1B.svg)](#)
[![Paper 2](https://img.shields.io/badge/ArXiv-RandNLA_Attention-B31B1B.svg)](#)

**MaximusLLM** is a hyper-efficient paradigm for long-context LLMs: reimagining model scaling and training through RandNLA Attention, MAXIS Loss, and Fisher-SVD.

By combining three novel mathematical breakthroughs, MaximusLLM achieves state-of-the-art efficiency without sacrificing convergence.

### The TL;DR
* **17.5x Faster Training** vs. standard Cross-Entropy.
* **39% Less VRAM** usage during training.
* **Infinite-Horizon Context** via Random Linear Algebra Latent Attention.
---

## Performance Benchmark: Intelligence per Second

MaximusLLM is optimized for the only metric that matters: **how much the model learns per dollar of compute.**

<img width="100%" alt="Maximus Benchmark" src="https://github.com/user-attachments/assets/4120228b-0c46-466b-a381-3f71932c56d7" />

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

**MAXIS** solves this by injecting a **Ghost Logit**, a mathematical "probability sink" that represents the unsampled vocabulary. By dynamically calculating the variance of unnormalized dot products, MAXIS maintains the dense supervision of full Cross-Entropy at a fraction of the cost.

### 2. RandNLA Latent Attention
**Bifurcated Information Highway**

Traditional attention is $O(N^2)$. RandNLA Attention splits the memory into two paths:
1. **The Detail Path (Top-K):** A dynamic scorer keeps the most critical tokens in a high-resolution KV-cache.
2. **The Sketch Path (Kronecker):** Background context is compressed using Randomized Numerical Linear Algebra (Kronecker-factored matrices).

Using an **Asymmetric Causal Mask**, the model keeps Queries at full resolution while attending to a compressed past, ensuring autoregressive causality.

### 3. Fisher SVD Initialization

**Fisher SVD** initializes the latent spaces by leveraging the **Fisher Information Matrix**, approximated from the gradient signals of a small calibration dataset. By accumulating the squared gradients $\sum (\frac{\partial L}{\partial W})^2$, we identify which parameters are most critical to the model's semantic performance.

<img width="1589" height="985" alt="fisher_svd_vs_svd" src="https://github.com/user-attachments/assets/d6eb6b98-7ab5-4742-801d-2969236f81ba" />

As shown in the benchmarks, Fisher SVD significantly reduces semantic error compared to standard SVD, keeping the model's knowledge intact during architectural shifts.

## Quick Start

### Inference
```python
from transformers import AutoTokenizer
from model import Model, Config 

model = Model.from_pretrained("yousefg/MaximusLLM")
tokenizer = AutoTokenizer.from_pretrained("yousefg/MaximusLLM")

# 32K context inference on a single T4
inputs = tokenizer("Your long document...", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
