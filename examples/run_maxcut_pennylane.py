from qarchga.utils.config import Config
from qarchga.backends.pennylane_backend import PennyLaneBackend
from qarchga.objectives.maxcut import maxcut_toy_objective
from qarchga.evolution.engine import run_ga

cfg = Config.load("configs/maxcut_toy.yaml")
backend = PennyLaneBackend(shots=0)

best, hist = run_ga(cfg, backend, maxcut_toy_objective)
print("BEST:", best.to_struct())
