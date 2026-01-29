from __future__ import annotations
import numpy as np

def _to_bohr(coords_angstrom: list[list[float]]) -> np.ndarray:
    # 1 Angstrom = 1.8897259886 Bohr
    return np.array(coords_angstrom, dtype=float) * 1.8897259886

def build_molecular_hamiltonian(mol_cfg: dict):
    """
    Returns (H, n_qubits, hf_state)
    Uses PennyLane qchem to build the electronic Hamiltonian.
    """
    try:
        import pennylane as qml
    except Exception as e:
        raise ImportError("PennyLane not installed. Install with: pip install -e .[pennylane]") from e

    symbols = mol_cfg["symbols"]
    coords = _to_bohr(mol_cfg["coordinates_angstrom"])
    charge = int(mol_cfg.get("charge", 0))
    multiplicity = int(mol_cfg.get("multiplicity", 1))
    basis = str(mol_cfg.get("basis", "sto-3g"))

    # molecular_hamiltonian builds second-quantized electronic Hamiltonian mapped to qubits
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coords,
        charge=charge,
        mult=multiplicity,
        basis=basis
    )

    # number of electrons
    n_electrons = qml.qchem.electron_number(symbols) - charge

    # active space (optional)
    active_e = mol_cfg.get("active_electrons", None)
    active_o = mol_cfg.get("active_orbitals", None)

    if active_e is not None or active_o is not None:
        # active space truncation
        active_e = n_electrons if active_e is None else int(active_e)
        # orbitals = spin-orbitals/2; qml.qchem expects active orbitals in spatial orbitals
        # If not given, keep all orbitals:
        active_o = (n_qubits // 2) if active_o is None else int(active_o)
        core, active = qml.qchem.active_space(active_e, active_o, n_electrons, n_qubits // 2)

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

    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    return H, int(n_qubits), np.array(hf_state, dtype=int)

def vqe_molecule_objective(backend, genome, mol_cache: dict):
    """
    Proper molecular VQE fitness:
      fitness = - <psi(theta, architecture) | H_mol | psi(...)>
    where H_mol is built from qml.qchem.molecular_hamiltonian.

    mol_cache is a dict that contains prebuilt (H, n_qubits, hf_state).
    """
    try:
        import pennylane as qml
    except Exception as e:
        raise ImportError("PennyLane not installed. Install with: pip install -e .[pennylane]") from e

    H = mol_cache["H"]
    n_qubits = mol_cache["n_qubits"]
    hf_state = mol_cache["hf_state"]

    # If genome qubit count mismatches, reject strongly
    if getattr(genome, "n_qubits", None) != n_qubits:
        return -1e9, {"energy": None, "note": "n_qubits mismatch"}

    def pl_obs(qml_module):
        # This function runs inside the PennyLaneBackend qnode
        # Prepare Hartree–Fock reference in computational basis:
        qml_module.BasisState(hf_state, wires=range(n_qubits))

        # Then apply the evolved architecture (genome gates)
        # PennyLaneBackend already applied gates BEFORE calling observable_fn,
        # so we do NOT re-apply them here.
        #
        # But in our backend design, the backend applies gates and then returns observable_fn(qml).
        # We want HF first -> then genome gates.
        #
        # Therefore: We will use a special backend entry point in the engine
        # by passing a wrapper observable_fn that includes HF prep and then measures H.
        return qml_module.expval(H)

    # We need HF -> genome -> measure(H). The current PennyLaneBackend applies genome first.
    # So we implement a local circuit via a specialized call using backend internals if available.
    if backend.__class__.__name__.lower().startswith("pennylane"):
        # Use a custom qnode here to guarantee ordering.
        import pennylane as qml2
        dev = qml2.device("default.qubit", wires=n_qubits, shots=None)

        @qml2.qnode(dev)
        def circuit():
            qml2.BasisState(hf_state, wires=range(n_qubits))
            # apply genome
            for layer in genome.layers:
                for (name, wires, param) in layer:
                    nm = name.lower()
                    if nm == "rx":
                        qml2.RX(float(param), wires=wires[0])
                    elif nm == "ry":
                        qml2.RY(float(param), wires=wires[0])
                    elif nm == "rz":
                        qml2.RZ(float(param), wires=wires[0])
                    elif nm == "cx":
                        qml2.CNOT(wires=list(wires))
                    else:
                        raise ValueError(f"Unsupported gate: {name}")
            return qml2.expval(H)

        energy = float(circuit())
    else:
        # If you later add a Qiskit-Nature backend, implement molecular measurement there.
        raise NotImplementedError("Molecular VQE objective is implemented for PennyLane backend.")

    fitness = -energy
    return float(fitness), {"energy": float(energy)}
