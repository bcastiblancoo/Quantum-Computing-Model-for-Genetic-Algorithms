from __future__ import annotations
import argparse

from .utils.config import Config
from .backends.qiskit_backend import QiskitBackend
from .backends.pennylane_backend import PennyLaneBackend
from .objectives import (
    maxcut_toy_objective,
    build_molecular_hamiltonian,
    vqe_molecule_objective,
)
from .evolution.engine import run_ga

OBJECTIVES = {
    "maxcut_toy": ("plain", maxcut_toy_objective),
    "vqe_molecule": ("molecule", vqe_molecule_objective),
}

def main():
    p = argparse.ArgumentParser(prog="qarchga")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run GA experiment")
    run.add_argument("--config", required=True, help="Path to YAML config")
    run.add_argument("--out", default="results", help="Output base folder (default: results). Use 'none' to disable.")

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

        obj_kind, obj_fn = OBJECTIVES[obj_name]

        mol_cache = None
        if obj_kind == "molecule":
            mol_cfg = cfg.get("molecule")
            if mol_cfg is None:
                raise ValueError("Config missing 'molecule:' section for vqe_molecule objective.")
            H, n_qubits, hf_state = build_molecular_hamiltonian(mol_cfg)
            mol_cache = {"H": H, "n_qubits": n_qubits, "hf_state": hf_state}

            if cfg.get("genome", "n_qubits", default=None) is None:
                cfg.raw["genome"]["n_qubits"] = int(n_qubits)

            objective = lambda b, g: obj_fn(b, g, mol_cache)
        else:
            objective = obj_fn

        out_dir = None if str(args.out).lower() == "none" else args.out

        best_genome, history, produced = run_ga(cfg, backend, objective, run_dir=out_dir)
        print("\nBest genome struct:\n", best_genome.to_struct())
        print("\nLast gen stats:\n", history[-1] if history else None)
        print("\nSaved outputs to:", produced)

