# Quantum-Computing-Model-for-Genetic-Algorithms

This repository implements a **Genetic Algorithm (GA)** that evolves **quantum circuit architectures**.
A genome encodes a **parameterized quantum circuit** (PQC): gate sequence, wiring, and continuous parameters.

## Features
- Architecture-as-genome: discrete (gates/wires) + continuous (angles)
- Selection rules:
  - **elitism**
  - **tournament selection**
  - **rank/roulette selection**
  - **age-fitness Pareto** (diversity pressure)
- Operators:
  - mutation of gates / wires / parameters
  - 1-point crossover
  - optional lightweight simplification
- Complexity penalties (depth / 2-qubit gates)
- Objectives:
  - **Toy VQE energy minimization** (simple Hamiltonian)
  - **Toy MaxCut** (small graph)
- Backends:
  - **Qiskit** (default if installed)
  - **PennyLane** (optional)

## Install
Clone and install:

```bash
pip install -e .[qiskit]
# or
pip install -e .[pennylane]
# or both
pip install -e .[qiskit,pennylane]

```

## How to run

```bash
python -m qarchga run --config configs/vqe_h2_pennylane.yaml

#Expect output:
Saved outputs to: results/run_20260129_203412

#To analyze and generate plots:
python scripts/analyze_run.py --run_dir results/run_20260129_203412

```
