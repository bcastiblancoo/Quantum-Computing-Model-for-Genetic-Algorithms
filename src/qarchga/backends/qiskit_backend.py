from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

from ..genomes import Genome

@dataclass
class QiskitBackend:
    shots: int = 0  # 0 => statevector if available

    def build_circuit(self, genome: Genome):
        try:
            from qiskit import QuantumCircuit
        except Exception as e:
            raise ImportError("Qiskit not installed. Install with: pip install -e .[qiskit]") from e

        qc = QuantumCircuit(genome.n_qubits)
        for layer in genome.layers:
            for (name, wires, param) in layer:
                if name.lower() == "rx":
                    qc.rx(float(param), wires[0])
                elif name.lower() == "ry":
                    qc.ry(float(param), wires[0])
                elif name.lower() == "rz":
                    qc.rz(float(param), wires[0])
                elif name.lower() == "cx":
                    qc.cx(wires[0], wires[1])
                else:
                    raise ValueError(f"Unsupported gate: {name}")
        return qc

    def expectation(self, genome: Genome, observable_fn: Callable):
        """
        observable_fn(qc, backend_ctx) -> float
        We keep it generic: objectives implement observable measurement.
        """
        qc = self.build_circuit(genome)
        return float(observable_fn(qc, {"shots": self.shots}))
