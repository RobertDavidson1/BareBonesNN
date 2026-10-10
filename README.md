# BareBonesNN

A NumPy neural network built from scratch with manual backpropagation, gradient descent, and weight pruning experiments.

Read the project write-up [here](tex/writeup.pdf) (WIP).

## Contents

- [Getting started](#getting-started)
- [Project structure](#project-structure)

## Getting started

```bash
# 1. Install uv (if you do not already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install the project dependencies
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Train the NumPy MLP on MNIST
#    The first run downloads the dataset through scikit-learn.
python src/barebonesnn/train.py
```


## Project structure

```text
BareBonesNN/
├── pyproject.toml                 # Project metadata, dependencies, and CLI entry point
├── uv.lock                        # Locked Python dependency versions
├── src/
│   └── barebonesnn/
│       ├── __init__.py            # Package initialization and public entry point
│       ├── data.py                # Dataset loading and preprocessing helpers
│       ├── model.py               # NumPy multilayer perceptron and backpropagation
│       ├── train.py               # Training loop, validation, and weight saving
│       ├── utils.py               # Evaluation and utility functions
│       ├── mlp_weights.npz        # Saved model weights
│       └── Notebooks/             # Experiments, visualizations, conversion, and pruning
├── tex/
│   ├── writeup.tex                # LaTeX source for the project write-up
│   ├── references.bib             # Bibliography for the write-up
│   └── writeup.pdf                # Compiled project write-up
└── README.md                      # Project overview and documentation
```
