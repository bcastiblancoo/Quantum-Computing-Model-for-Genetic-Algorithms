from __future__ import annotations
import numpy as np

Z_MAP = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15, "S": 16, "Cl": 17,"Ar": 18,
}

def _to_bohr(coords_angstrom: list[list[float]]) -> np.ndarray:
    return np.array(coords_angstrom, dtype=float) * 1.8897259886

def _electron_count(symbols: list[str], charge: int) -> int:
    try:
        return int(sum(Z_MAP[s] for s in symbols) - charge)
    except KeyError as e:
        raise ValueError(f"Unknown element symbol {e}. Extend Z_MAP in vqe_molecule.py.")

def build_molecular_hamiltonian(mol_cfg: dict):
    """
    Returns (H, n_qubits, hf_state) using PennyLane qchem.
    """
    try:
        import pennylane as qml
    except Exception as e:
        raise ImportError("PennyLane not installed. Install with: pip install -e .[pennylane]") from e

    symbols = list(mol_cfg["symbols"])
    coords = _to_bohr(mol_cfg["coordinates_angstrom"])
    charge = int(mol_cfg.get("charge", 0))
    multiplicity = int(mol_cfg.get("multiplicity", 1))
    basis = str(mol_cfg.get("basis", "sto-3g"))

    H, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coords,
        charge=charge,
        mult=multiplicity,
        basis=basis
    )

    n_electrons = _electron_count(symbols, charge)

    # Optional active space (keep API simple & compatible)
    active_e = mol_cfg.get("active_electrons", None)
    active_o = mol_cfg.get("active_orbitals", None)
    if active_e is not None or active_o is not None:
        active_e = n_electrons if active_e is None else int(active_e)
        # active_orbitals is in spatial orbitals; n_qubits=2*n_orbitals
        active_o = (int(n_qubits) // 2) if active_o is None else int(active_o)

        H, n_qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            coords,
            charge=charge,
            mult=multiplicity,
            basis=basis,
            active_electrons=active_e,
            active_orbitals=active_o
        )
        n_electrons = active_e

    hf_state = qml.qchem.hf_state(n_electrons, int(n_qubits))
    return H, int(n_qubits), np.array(hf_state, dtype=int)

def vqe_molecule_objective(backend, genome, mol_cache: dict):
    """
    fitness = - <psi|H|psi>, with psi prepared from HF reference + evolved genome.
    Implemented for PennyLane backend.
    """
    if not backend.__class__.__name__.lower().startswith("pennylane"):
        raise NotImplementedError("Molecular VQE objective is implemented for PennyLane backend only.")

    import pennylane as qml

    H = mol_cache["H"]
    n_qubits = int(mol_cache["n_qubits"])
    hf_state = np.array(mol_cache["hf_state"], dtype=int)

    if getattr(genome, "n_qubits", None) != n_qubits:
        return -1e9, {"energy": None, "note": "n_qubits mismatch"}

    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(hf_state, wires=range(n_qubits))
        for layer in genome.layers:
            for (name, wires, param) in layer:
                nm = name.lower()
                if nm == "rx":
                    qml.RX(float(param), wires=wires[0])
                elif nm == "ry":
                    qml.RY(float(param), wires=wires[0])
                elif nm == "rz":
                    qml.RZ(float(param), wires=wires[0])
                elif nm == "cx":
                    qml.CNOT(wires=list(wires))
                else:
                    raise ValueError(f"Unsupported gate: {name}")
        return qml.expval(H)

    energy = float(circuit())
    return float(-energy), {"energy": float(energy)}
