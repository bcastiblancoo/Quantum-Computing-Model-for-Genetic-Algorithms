from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table

from ..rng import RNG
from ..genomes import random_genome, Genome
from ..operators import mutate, crossover_1p, simplify_light
from ..selection import (
    Individual, elitism, tournament, rank_selection, roulette, age_fitness_pareto
)
from ..utils.hashing import stable_hash
from ..utils.io import make_run_dir, write_json, write_yaml

console = Console()

CSV_FIELDS = ["gen", "best_fitness", "mean_fitness", "best_depth", "best_n2q", "unique"]

def _select_one(rng, pop, mode, tournament_k=3):
    if mode == "tournament":
        return tournament(rng, pop, tournament_k)
    if mode == "rank":
        return rank_selection(rng, pop)
    if mode == "roulette":
        return roulette(rng, pop)
    if mode == "age_pareto":
        return age_fitness_pareto(rng, pop)
    raise ValueError(f"Unknown selection mode: {mode}")

def run_ga(cfg, backend, objective_fn, run_dir: str | None = "results"):
    seed = int(cfg.get("seed", default=0))
    rng = RNG(seed)._rng

    n_qubits = int(cfg.get("genome", "n_qubits"))
    max_layers = int(cfg.get("genome", "max_layers"))
    init_lo, init_hi = cfg.get("genome", "init_layers", default=[3, 6])
    gate_set_1q = list(cfg.get("genome", "gate_set_1q"))
    gate_set_2q = list(cfg.get("genome", "gate_set_2q"))

    generations = int(cfg.get("ga", "generations"))
    pop_size = int(cfg.get("ga", "population_size"))
    elit_k = int(cfg.get("ga", "elitism", default=0))
    selection_mode = str(cfg.get("ga", "selection", default="tournament"))
    tournament_k = int(cfg.get("ga", "tournament_k", default=3))
    crossover_rate = float(cfg.get("ga", "crossover_rate"))
    mutation_rate = float(cfg.get("ga", "mutation_rate"))
    lam_depth = float(cfg.get("ga", "lambda_depth", default=0.0))
    lam_2q = float(cfg.get("ga", "lambda_2q", default=0.0))

    param_restarts = int(cfg.get("fitness", "param_restarts", default=1))
    local_opt_steps = int(cfg.get("fitness", "local_opt_steps", default=0))

    # -------------------------
    # Output folder (optional)
    # -------------------------
    out_dir: Path | None = None
    csv_path: Path | None = None
    if run_dir is not None:
        out_dir = make_run_dir(run_dir)
        write_yaml(out_dir / "config_snapshot.yaml", cfg.raw)
        csv_path = out_dir / "history.csv"
        # Initialize CSV with header immediately (prevents history[0] bugs)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()

    # -------------------------
    # Init population
    # -------------------------
    population: list[Genome] = []
    for _ in range(pop_size):
        nl = int(rng.integers(int(init_lo), int(init_hi) + 1))
        population.append(random_genome(rng, n_qubits, nl, gate_set_1q, gate_set_2q))

    history: list[dict] = []

    def score(genome: Genome):
        # Optionally do cheap local parameter tweaks by random search
        best_fit = -1e18
        best_meta = {}
        best_gen = genome
        for _ in range(param_restarts):
            gtry = genome.copy()

            # local random coordinate search by parameter jitter-only mutations
            for _ in range(local_opt_steps):
                gmut = mutate(
                    rng, gtry, gate_set_1q, gate_set_2q, max_layers,
                    p_add_layer=0.0, p_del_layer=0.0, p_gate_edit=0.0, p_param_jitter=1.0
                )
                f, meta = objective_fn(backend, gmut)
                if f > best_fit:
                    best_fit, best_meta, best_gen = f, meta, gmut

            f0, meta0 = objective_fn(backend, gtry)
            if f0 > best_fit:
                best_fit, best_meta, best_gen = f0, meta0, gtry

        # regularize complexity
        depth = best_gen.depth()
        n2q = best_gen.count_2q()
        reg = lam_depth * depth + lam_2q * n2q
        return float(best_fit - reg), depth, n2q, best_meta, best_gen

    # -------------------------
    # Evolution loop
    # -------------------------
    for gen in range(generations):
        evaluated: list[Individual] = []
        unique_hashes = set()

        scored = []
        for g in population:
            fit, depth, n2q, meta, best_variant = score(g)
            scored.append((fit, depth, n2q, meta, best_variant))

        for (g, (fit, depth, n2q, meta, best_variant)) in zip(population, scored):
            evaluated.append(Individual(genome=best_variant, fitness=fit, depth=depth, n2q=n2q, age=g.age))
            unique_hashes.add(stable_hash(best_variant.to_struct()))

        best_ind = max(evaluated, key=lambda x: x.fitness)
        mean_fit = float(np.mean([x.fitness for x in evaluated]))

        row = {
            "gen": gen,
            "best_fitness": float(best_ind.fitness),
            "mean_fitness": float(mean_fit),
            "best_depth": int(best_ind.depth),
            "best_n2q": int(best_ind.n2q),
            "unique": int(len(unique_hashes)),
        }
        history.append(row)

        # Write one CSV row per generation (safe always)
        if csv_path is not None:
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                w.writerow(row)

        # pretty print
        tbl = Table(title=f"Generation {gen}")
        tbl.add_column("best_fitness", justify="right")
        tbl.add_column("mean_fitness", justify="right")
        tbl.add_column("best_depth", justify="right")
        tbl.add_column("best_2q", justify="right")
        tbl.add_column("unique", justify="right")
        tbl.add_row(
            f"{best_ind.fitness:.6f}",
            f"{mean_fit:.6f}",
            str(best_ind.depth),
            str(best_ind.n2q),
            str(len(unique_hashes)),
        )
        console.print(tbl)

        # next generation
        next_pop: list[Genome] = []

        if elit_k > 0:
            elites = elitism(evaluated, elit_k)
            for e in elites:
                gg = e.genome.copy()
                gg.age += 1
                next_pop.append(gg)

        while len(next_pop) < pop_size:
            p1 = _select_one(rng, evaluated, selection_mode, tournament_k)
            p2 = _select_one(rng, evaluated, selection_mode, tournament_k)

            c1, c2 = p1.genome.copy(), p2.genome.copy()
            if rng.random() < crossover_rate:
                c1, c2 = crossover_1p(rng, p1.genome, p2.genome)

            if rng.random() < mutation_rate:
                c1 = simplify_light(mutate(rng, c1, gate_set_1q, gate_set_2q, max_layers))
            if rng.random() < mutation_rate:
                c2 = simplify_light(mutate(rng, c2, gate_set_1q, gate_set_2q, max_layers))

            c1.age = 0
            c2.age = 0
            next_pop.append(c1)
            if len(next_pop) < pop_size:
                next_pop.append(c2)

        population = next_pop

    # -------------------------
    # Final best
    # -------------------------
    final_eval = []
    for g in population:
        fit, depth, n2q, meta, best_variant = score(g)
        final_eval.append((fit, best_variant))

    best_fit, best_genome = max(final_eval, key=lambda x: x[0])

    if out_dir is not None:
        write_json(out_dir / "best_genome.json", best_genome.to_struct())

    return best_genome, history, (str(out_dir) if out_dir is not None else None)

