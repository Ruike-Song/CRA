# CRA: Causal Reward Adjustment

Code implementation for **Causal Reward Adjustment (CRA)**, a method that mitigates reward hacking in Process Reward Models (PRMs) through causal inference and Sparse Autoencoders (SAEs).

## Method Overview

CRA treats reward hacking features as confounders and applies the **backdoor adjustment** to correct PRM scores:

$$P(Y \mid do(X)) = \sum_Z P(Y \mid X, Z) \cdot P(Z)$$

The pipeline consists of three stages:

1. **SAE Training** — Train Sparse Autoencoders on PRM internal activations to decompose them into interpretable sparse features.
2. **Feature Identification** — Use statistical tests to identify latent features correlated with reward hacking.
3. **Backdoor Adjustment** — Apply causal intervention to correct PRM scores during inference. *(This part involves content from an unpublished paper and will be released upon publication.)*

## Project Structure

```
CRA/
├── sae_training.py               # SAE training on PRM activations
├── feature_identification.py     # Statistical feature identification
├── reward_hacking_analyzer.py    # Utility: activation extraction & data loading
├── compute_optimal_tts/          # Beam search & model serving infrastructure
└── scripts/serve_gpu.sh          # Launch controller + workers
```

Files under `compute_optimal_tts/` are reused from [compute-optimal-tts](https://github.com/RyanLiu112/compute-optimal-tts) and marked with `[REUSED]` in their headers.

## Dependencies

```
torch>=2.3.0
transformers>=4.52.0
scipy>=1.15.0
numpy>=1.26.0
h5py>=3.10.0
fschat>=0.2.36
matplotlib>=3.5.0
tqdm>=4.66.0
accelerate>=1.7.0
```

## Acknowledgements

The beam search and model serving infrastructure is reused from [compute-optimal-tts](https://github.com/RyanLiu112/compute-optimal-tts).
