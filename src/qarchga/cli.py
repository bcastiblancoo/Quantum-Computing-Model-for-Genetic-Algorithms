from __future__ import annotations
import argparse

from .utils.config import Config
from .backends.qiskit_backend import QiskitBackend
from .backends.pennylane_backend import PennyLaneBackend
from .objectives import vqe_toy_objective, maxcut_toy_objective
from .evolution.engine import run_ga

OBJECTIVES = {
    "vqe_toy": vqe_toy_objective,
    "maxcut_toy": maxcut_toy_objective,
}

def main():
    p = argparse.ArgumentParser(prog="qarchga")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run GA experiment")
    run.add_argument("--config", required=True, help="Path to YAML config")

    args = p.parse_args()

    if args.cmd == "run":
        cfg = Config.load(args.config)
        backend_name = str(cfg.get("backend", default="qiskit"))
        obj_name = str(cfg.get("objective"))

        if obj_name not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{obj_name}'. Available: {list(OBJECTIVES.keys())}")

        if backend_name == "qiskit":
            backend = QiskitBackend(shots=0)
        elif backend_name == "pennylane":
            backend = PennyLaneBackend(shots=0)
        else:
            raise ValueError("backend must be 'qiskit' or 'pennylane'")

        best_genome, history = run_ga(cfg, backend, OBJECTIVES[obj_name])
        print("\nBest genome struct:\n", best_genome.to_struct())
        print("\nLast gen stats:\n", history[-1])
