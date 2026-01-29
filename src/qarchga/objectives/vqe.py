from __future__ import annotations
import numpy as np

def vqe_toy_objective(backend, genome):
    """
    Toy 'energy' objective: E = <Z0> + 0.5<Z1> + 0.25<Z0 Z1> - 0.1<X2 X3>
    We return FITNESS = -E (maximize fitness).
    """

    def qiskit_obs(qc, ctx):
        from qiskit.quantum_info import Statevector, Operator
        # statevector evaluation
        sv = Statevector.from_instruction(qc)
        Z = np.array([[1,0],[0,-1]], dtype=complex)
        X = np.array([[0,1],[1,0]], dtype=complex)
        I = np.eye(2, dtype=complex)

        def kron_ops(ops):
            out = ops[0]
            for o in ops[1:]:
                out = np.kron(out, o)
            return out

        n = qc.num_qubits
        def op_on(qubit, single):
            ops = [I]*n
            ops[qubit] = single
            return kron_ops(ops)

        def op_two(q1, q2, single1, single2):
            ops = [I]*n
            ops[q1] = single1
            ops[q2] = single2
            return kron_ops(ops)

        E = 0.0
        E += (sv.expectation_value(Operator(op_on(0, Z))).real)
        E += 0.5*(sv.expectation_value(Operator(op_on(1, Z))).real)
        E += 0.25*(sv.expectation_value(Operator(op_two(0, 1, Z, Z))).real)
        E += -0.1*(sv.expectation_value(Operator(op_two(2, 3, X, X))).real)
        return float(E)

    def pl_obs(qml):
        import pennylane as qml2
        # qml passed is module; return expectation directly
        E = 0.0
        E += qml2.expval(qml2.PauliZ(0))
        E += 0.5 * qml2.expval(qml2.PauliZ(1))
        E += 0.25 * qml2.expval(qml2.PauliZ(0) @ qml2.PauliZ(1))
        E += -0.1 * qml2.expval(qml2.PauliX(2) @ qml2.PauliX(3))
        return E

    if backend.__class__.__name__.lower().startswith("qiskit"):
        energy = backend.expectation(genome, qiskit_obs)
    else:
        energy = backend.expectation(genome, pl_obs)

    fitness = -float(energy)
    return fitness, {"energy": float(energy)}
