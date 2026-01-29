from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable
from ..genomes import Genome

@dataclass
class PennyLaneBackend:
    shots: int = 0

    def expectation(self, genome: Genome, observable_fn: Callable):
        try:
            import pennylane as qml
        except Exception as e:
            raise ImportError("PennyLane not installed. Install with: pip install -e .[pennylane]") from e

        dev = qml.device("default.qubit", wires=genome.n_qubits, shots=None if self.shots == 0 else self.shots)

        @qml.qnode(dev)
        def circuit():
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
            return observable_fn(qml)

        return float(circuit())
