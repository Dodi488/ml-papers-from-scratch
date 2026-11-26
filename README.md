# ml-papers-from-scratch
Progressive implementations of 15 foundational ML papers from scratch in NumPy and PyTorch, building incrementally from backpropagation to GPT-2, DDPM, and FlashAttention.

> Based on this article: [15 ML Papers Worth Implementing From Scratch, and in What Order](https://www.tensortonic.com/guides/ml-papers-to-implement-from-scratch).

---

## Repository Structure

This repository is structured as a single growing codebase where later papers import components built in earlier stages:

```text
ml-papers-from-scratch/
├── 01_how_networks_learn/
│   ├── backprop.py
│   ├── glorot_init.py
│   ├── adam.py
│   ├── dropout.py
│   └── batchnorm.py
├── 02_architectures/
│   ├── lenet5.py
│   ├── resnet.py
│   ├── lstm.py
│   └── bahdanau.py
├── 03_transformers/
│   ├── transformer.py
│   ├── gpt2.py
│   └── lora.py
├── 04_generative_and_kernels/
│   ├── vae.py
│   ├── ddpm.py
│   └── flash_attention.py
├── papers
└── README.md
