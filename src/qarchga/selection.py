from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Individual:
    genome: object
    fitness: float
    depth: int
    n2q: int
    age: int

def elitism(pop: list[Individual], k: int) -> list[Individual]:
    return sorted(pop, key=lambda x: x.fitness, reverse=True)[:k]

def tournament(rng, pop: list[Individual], k: int) -> Individual:
    idxs = rng.choice(np.arange(len(pop)), size=k, replace=False)
    cand = [pop[int(i)] for i in idxs]
    return max(cand, key=lambda x: x.fitness)

def rank_selection(rng, pop: list[Individual]) -> Individual:
    # linear rank weights
    sorted_pop = sorted(pop, key=lambda x: x.fitness)
    n = len(sorted_pop)
    ranks = np.arange(1, n + 1)  # worst=1, best=n
    probs = ranks / ranks.sum()
    i = int(rng.choice(np.arange(n), p=probs))
    return sorted_pop[i]

def roulette(rng, pop: list[Individual]) -> Individual:
    # requires non-negative fitness
    f = np.array([max(0.0, x.fitness) for x in pop], dtype=float)
    if f.sum() == 0:
        return pop[int(rng.integers(0, len(pop)))]
    p = f / f.sum()
    i = int(rng.choice(np.arange(len(pop)), p=p))
    return pop[i]

def age_fitness_pareto(rng, pop: list[Individual]) -> Individual:
    # Prefer individuals that are good in fitness but also not all the same age.
    # We sample a few and choose by Pareto dominance on (fitness, -age).
    m = min(5, len(pop))
    idxs = rng.choice(np.arange(len(pop)), size=m, replace=False)
    cand = [pop[int(i)] for i in idxs]

    def dominates(a: Individual, b: Individual) -> bool:
        return (a.fitness >= b.fitness and a.age <= b.age) and (a.fitness > b.fitness or a.age < b.age)

    nondominated = []
    for c in cand:
        if not any(dominates(o, c) for o in cand if o is not c):
            nondominated.append(c)
    return nondominated[int(rng.integers(0, len(nondominated)))]
