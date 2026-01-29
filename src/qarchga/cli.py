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

from qarchga.objectives import (
    maxcut_toy_objective,
    build_molecular_hamiltonian,
    vqe_molecule_objective,
)

OBJECTIVES = {
    "maxcut_toy": ("plain", maxcut_toy_objective),
    "vqe_molecule": ("molecule", vqe_molecule_objective),
}


def main():
    p = argparse.ArgumentParser(prog="qarchga")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run GA experiment")
    run.add_argument("--config", required=True, help="Path to YAML config")

    args = p.parse_args()

    if args.cmd == "run":
        obj_kind, obj_fn = OBJECTIVES[obj_name]
        mol_cache = None
        if obj_kind == "molecule":
            mol_cfg = cfg.get("molecule")
            if mol_cfg is None:
                raise ValueError("Config missing 'molecule:' section for vqe_molecule objective.")
            H, n_qubits, hf_state = build_molecular_hamiltonian(mol_cfg)
            mol_cache = {"H": H, "n_qubits": n_qubits, "hf_state": hf_state}

    # override genome qubit count if config says null
    if cfg.get("genome", "n_qubits", default=None) is None:
        cfg.raw["genome"]["n_qubits"] = int(n_qubits)
        
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
        if obj_kind == "molecule":
            objective = lambda backend, genome: obj_fn(backend, genome, mol_cache)
        else:
            objective = obj_fn
            
        best_genome, history = run_ga(cfg, backend, objective)

        print("\nBest genome struct:\n", best_genome.to_struct())
        print("\nLast gen stats:\n", history[-1])
