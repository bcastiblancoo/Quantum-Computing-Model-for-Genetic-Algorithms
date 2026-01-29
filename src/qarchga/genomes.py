from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

Gate = Tuple[str, Tuple[int, ...], Optional[float]]  # (name, wires, param)

@dataclass
class Genome:
    n_qubits: int
    layers: List[List[Gate]] = field(default_factory=list)
    age: int = 0  # for age-fitness selection pressure

    def copy(self) -> "Genome":
        return Genome(
            n_qubits=self.n_qubits,
            layers=[[ (g, tuple(w), p) for (g, w, p) in layer ] for layer in self.layers],
            age=self.age
        )

    def depth(self) -> int:
        return len(self.layers)

    def count_2q(self) -> int:
        c = 0
        for layer in self.layers:
            for (name, wires, _) in layer:
                if len(wires) == 2:
                    c += 1
        return c

    def to_struct(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "age": self.age,
            "layers": [
                [{"name": n, "wires": list(w), "param": p} for (n, w, p) in layer]
                for layer in self.layers
            ],
        }

    @staticmethod
    def from_struct(d: dict) -> "Genome":
        g = Genome(n_qubits=int(d["n_qubits"]), age=int(d.get("age", 0)))
        g.layers = []
        for layer in d["layers"]:
            g.layers.append([(x["name"], tuple(x["wires"]), x.get("param", None)) for x in layer])
        return g

def random_genome(
    rng,
    n_qubits: int,
    n_layers: int,
    gate_set_1q: list[str],
    gate_set_2q: list[str],
) -> Genome:
    layers: List[List[Gate]] = []
    for _ in range(n_layers):
        layer: List[Gate] = []
        # 1-qubit gates: sample a few wires
        k1 = int(rng.integers(1, max(2, n_qubits)))
        wires1 = rng.choice(np.arange(n_qubits), size=k1, replace=False)
        for q in wires1:
            name = str(rng.choice(gate_set_1q))
            param = float(rng.uniform(-np.pi, np.pi))
            layer.append((name, (int(q),), param))
        # 2-qubit gates: sample a few pairs
        if n_qubits >= 2 and len(gate_set_2q) > 0:
            k2 = int(rng.integers(0, max(1, n_qubits // 2 + 1)))
            for _ in range(k2):
                a, b = rng.choice(np.arange(n_qubits), size=2, replace=False)
                name = str(rng.choice(gate_set_2q))
                layer.append((name, (int(a), int(b)), None))
        layers.append(layer)
    return Genome(n_qubits=n_qubits, layers=layers, age=0)
