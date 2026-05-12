# Phase of Matter — Quantum Dataset Generator

A quantum Phase of Matter classification dataset generator, designed for contribution to [`qiskit-machine-learning`](https://github.com/qiskit-community/qiskit-machine-learning).

Given a spin-chain Hamiltonian, the generator samples coupling parameters from well-separated phase regions, computes the exact ground state via sparse diagonalization, and returns the ground-state amplitudes as feature vectors alongside phase labels — ready for use in quantum machine learning experiments.

---

## Background

Based on the models studied in:

> Bermejo-Vega et al., *"Quantum Convolutional Neural Networks are (Effectively) Classically Simulable"*, [arXiv:2408.12739](https://arxiv.org/abs/2408.12739) (2024)

The paper benchmarks QCNNs on phase-of-matter classification tasks. This package reproduces the four Hamiltonians from the paper (eqs. 6–9) and provides a clean dataset API matching the existing `qiskit-machine-learning` dataset conventions.

---

## Models

| Keyword | Hamiltonian | Phases |
|---|---|---|
| `"heisenberg"` | Bond-alternating XXX Heisenberg (eq. 6) | `trivial`, `topological` |
| `"haldane"` | Haldane chain (eq. 7) | `antiferromagnetic`, `paramagnetic`, `spt` |
| `"annni"` | Axial Next-Nearest-Neighbor Ising (eq. 8) | `ferromagnetic`, `paramagnetic`, `floating`, `antiphase` |
| `"cluster"` | Cluster Hamiltonian w/ periodic BC (eq. 9) | `haldane`, `ferromagnetic`, `antiferromagnetic`, `trivial` |

---

## Installation

```bash
pip install qiskit scipy numpy
```

Clone this repo and import directly:

```bash
git clone https://github.com/rogue-infinity/qml-phase-of-matter.git
cd qml-phase-of-matter
```

---

## Quick Start

```python
from phase_of_matter import phase_of_matter_data

# 10 training + 5 test samples, 4-qubit Heisenberg chain
x_train, y_train, x_test, y_test = phase_of_matter_data(
    10, 5, n=4, model="heisenberg", seed=0
)

print(x_train.shape)  # (10, 16)  — complex ground-state amplitudes
print(y_train.shape)  # (10, 2)   — one-hot encoded phase labels
```

### All four models

```python
for model, n_classes in [("heisenberg", 2), ("haldane", 3), ("annni", 4), ("cluster", 4)]:
    x_tr, y_tr, x_te, y_te = phase_of_matter_data(20, 10, n=4, model=model, seed=42)
    print(f"{model:12s}  train={x_tr.shape}  labels={y_tr.shape}")
```

### String labels instead of one-hot

```python
x_tr, y_tr, x_te, y_te = phase_of_matter_data(
    10, 5, n=4, model="annni", one_hot=False, seed=0
)
print(set(y_tr))  # {'ferromagnetic', 'paramagnetic', 'floating', 'antiphase'}
```

### Statevector output

```python
from qiskit.quantum_info import Statevector

x_tr, y_tr, x_te, y_te = phase_of_matter_data(
    4, 2, n=4, model="heisenberg", formatting="statevector", seed=0
)
print(type(x_tr[0]))  # <class 'qiskit.quantum_info.Statevector'>
```

---

## API Reference

```python
phase_of_matter_data(
    training_size,          # total training samples (balanced across classes)
    test_size,              # total test samples
    n,                      # number of qubits / lattice sites (≥ 4)
    *,
    model       = "heisenberg",   # see Models table above
    one_hot     = True,           # True → ndarray labels; False → string labels
    formatting  = "ndarray",      # "ndarray" or "statevector"
    class_labels= None,           # optional custom label names
    include_sample_total = False, # if True, appends per-class count as 5th return value
    seed        = None,           # integer seed for reproducibility
    backend     = None,           # None = exact diag (recommended); Qiskit backend = VQE
)
```

**Returns** `(x_train, y_train, x_test, y_test)` where:
- Features — shape `(n_samples, 2**n)` complex ndarray, or `list[Statevector]`
- Labels — shape `(n_samples, n_classes)` one-hot ndarray, or array of strings

> **Note on VQE:** Passing a Qiskit backend activates a VQE approximation pathway (EfficientSU2 + COBYLA). This is intended for hardware-experiment workflows only — VQE approximations near phase boundaries can produce incorrect labels. Use the default exact diagonalization for dataset generation.

---

## Package Structure

```
phase_of_matter/
├── __init__.py          # exports phase_of_matter_data
├── phase_of_matter.py   # public API + orchestration
├── _base.py             # shared utilities (pauli_term, exact diag, VQE)
├── _heisenberg.py       # Bond-alternating XXX Heisenberg
├── _haldane.py          # Haldane chain
├── _annni.py            # ANNNI model
├── _cluster.py          # Cluster Hamiltonian
└── test_phase_of_matter.py  # test suite (55 tests)
```

---

## Tests

```bash
pip install ddt
pytest phase_of_matter/test_phase_of_matter.py -v
# 55 passed
```

The test suite covers Hermiticity of all Hamiltonians, ground-state normalization and eigenstate residuals, phase label coverage, API shape contracts, seed reproducibility, and error handling.

---

## Contributing to qiskit-machine-learning

This package is structured for direct contribution. To integrate it into the official repo:

1. Copy `phase_of_matter/` → `qiskit_machine_learning/datasets/phase_of_matter/`
2. In `qiskit_machine_learning/datasets/__init__.py` add:
   ```python
   from .phase_of_matter import phase_of_matter_data
   # add "phase_of_matter_data" to __all__
   ```
3. Move `test_phase_of_matter.py` → `test/datasets/`
4. Replace the `try/except` base class import in the test file with:
   ```python
   from test import QiskitMachineLearningTestCase
   ```
5. Add a release note under `releasenotes/notes/`

---

## License

Apache License 2.0 — see [LICENSE](https://www.apache.org/licenses/LICENSE-2.0).
