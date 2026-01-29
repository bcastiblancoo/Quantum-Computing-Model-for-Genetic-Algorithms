from __future__ import annotations
import numpy as np
from .genomes import Genome, Gate

def mutate(
    rng,
    genome: Genome,
    gate_set_1q: list[str],
    gate_set_2q: list[str],
    max_layers: int,
    p_add_layer: float = 0.15,
    p_del_layer: float = 0.10,
    p_gate_edit: float = 0.60,
    p_param_jitter: float = 0.50,
) -> Genome:
    g = genome.copy()

    # maybe add a layer
    if rng.random() < p_add_layer and g.depth() < max_layers:
        insert_at = int(rng.integers(0, g.depth() + 1))
        g.layers.insert(insert_at, [])

    # maybe delete a layer
    if g.depth() > 1 and rng.random() < p_del_layer:
        del_at = int(rng.integers(0, g.depth()))
        g.layers.pop(del_at)

    # edit gates within layers
    for li in range(g.depth()):
        layer = g.layers[li]
        if len(layer) == 0 and rng.random() < 0.35:
            # add a random 1q gate
            q = int(rng.integers(0, g.n_qubits))
            name = str(rng.choice(gate_set_1q))
            param = float(rng.uniform(-np.pi, np.pi))
            layer.append((name, (q,), param))

        if len(layer) > 0 and rng.random() < p_gate_edit:
            idx = int(rng.integers(0, len(layer)))
            name, wires, param = layer[idx]

            # random action: delete / change type / rewire
            action = int(rng.integers(0, 3))
            if action == 0 and len(layer) > 1:
                layer.pop(idx)
            elif action == 1:
                # change gate name (preserve arity)
                if len(wires) == 1:
                    name = str(rng.choice(gate_set_1q))
                    if param is None:
                        param = float(rng.uniform(-np.pi, np.pi))
                else:
                    name = str(rng.choice(gate_set_2q)) if gate_set_2q else name
                    param = None
                layer[idx] = (name, wires, param)
            else:
                # rewire
                if len(wires) == 1:
                    q = int(rng.integers(0, g.n_qubits))
                    layer[idx] = (name, (q,), param)
                else:
                    if g.n_qubits >= 2:
                        a, b = rng.integers(0, g.n_qubits, size=2)
                        while b == a:
                            b = int(rng.integers(0, g.n_qubits))
                        layer[idx] = (name, (int(a), int(b)), None)

        # param jitter
        if rng.random() < p_param_jitter:
            for gi in range(len(layer)):
                n, w, p = layer[gi]
                if len(w) == 1 and p is not None:
                    p = float(p + rng.normal(0.0, 0.25))
                    # wrap to [-pi, pi]
                    p = float(((p + np.pi) % (2 * np.pi)) - np.pi)
                    layer[gi] = (n, w, p)

    return g

def crossover_1p(rng, a: Genome, b: Genome) -> tuple[Genome, Genome]:
    # 1-point crossover at the layer level
    ga, gb = a.copy(), b.copy()
    da, db = ga.depth(), gb.depth()
    if da < 2 or db < 2:
        return ga, gb
    cut_a = int(rng.integers(1, da))
    cut_b = int(rng.integers(1, db))
    child1 = Genome(n_qubits=ga.n_qubits, layers=ga.layers[:cut_a] + gb.layers[cut_b:], age=0)
    child2 = Genome(n_qubits=gb.n_qubits, layers=gb.layers[:cut_b] + ga.layers[cut_a:], age=0)
    return child1, child2

def simplify_light(genome: Genome) -> Genome:
    # remove empty layers and trivial duplicates inside a layer
    g = genome.copy()
    new_layers = []
    for layer in g.layers:
        if not layer:
            continue
        seen = set()
        dedup = []
        for (n, w, p) in layer:
            key = (n, w, None if p is None else round(float(p), 6))
            if key not in seen:
                seen.add(key)
                dedup.append((n, w, p))
        new_layers.append(dedup)
    g.layers = new_layers if new_layers else [[]]
    return g
