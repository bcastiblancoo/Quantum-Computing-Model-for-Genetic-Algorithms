from __future__ import annotations
import numpy as np

# Toy graph (5 nodes)
EDGES = [(0,1),(1,2),(2,3),(3,4),(4,0),(1,3)]

def maxcut_toy_objective(backend, genome):
    """
    MaxCut objective uses cost Hamiltonian:
    C = sum_{(i,j)} (1 - Zi Zj)/2
    We maximize <C>.
    """

    def qiskit_obs(qc, ctx):
        from qiskit.quantum_info import Statevector, Operator
        sv = Statevector.from_instruction(qc)
        Z = np.array([[1,0],[0,-1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        def kron_ops(ops):
            out = ops[0]
            for o in ops[1:]:
                out = np.kron(out, o)
            return out

        n = qc.num_qubits
        def op_two(q1, q2):
            ops = [I]*n
            ops[q1] = Z
            ops[q2] = Z
            return kron_ops(ops)

        C = 0.0
        for (i,j) in EDGES:
            zz = sv.expectation_value(Operator(op_two(i,j))).real
            C += (1.0 - zz)/2.0
        return float(C)

    def pl_obs(qml):
        import pennylane as qml2
        C = 0.0
        for (i,j) in EDGES:
            C += 0.5 * (1.0 - qml2.expval(qml2.PauliZ(i) @ qml2.PauliZ(j)))
        return C

    if backend.__class__.__name__.lower().startswith("qiskit"):
        cost = backend.expectation(genome, qiskit_obs)
    else:
        cost = backend.expectation(genome, pl_obs)

    fitness = float(cost)
    return fitness, {"maxcut": float(cost)}
