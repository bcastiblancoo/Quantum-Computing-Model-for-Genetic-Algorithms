from qarchga.utils.config import Config
from qarchga.backends.qiskit_backend import QiskitBackend
from qarchga.objectives.vqe import vqe_toy_objective
from qarchga.evolution.engine import run_ga

cfg = Config.load("configs/vqe_toy.yaml")
backend = QiskitBackend(shots=0)

best, hist = run_ga(cfg, backend, vqe_toy_objective)
print("BEST:", best.to_struct())
