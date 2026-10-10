# BareBonesNN

A NumPy neural network built from scratch with manual backpropagation, gradient descent, and weight pruning experiments.

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
