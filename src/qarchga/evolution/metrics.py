from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GenStats:
    gen: int
    best: float
    mean: float
    best_depth: int
    best_n2q: int
    unique: int
