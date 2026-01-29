from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gate_counts(best_struct: dict):
    counts = {}
    for layer in best_struct.get("layers", []):
        for g in layer:
            nm = g["name"].lower()
            counts[nm] = counts.get(nm, 0) + 1
    return counts

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="results/run_YYYYMMDD_HHMMSS")
    ap.add_argument("--out", default=None, help="Output folder (default: <run_dir>/plots)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_path = run_dir / "history.csv"
    best_path = run_dir / "best_genome.json"

    if not hist_path.exists():
        raise FileNotFoundError(f"Missing {hist_path}. Did the run produce outputs?")
    if not best_path.exists():
        raise FileNotFoundError(f"Missing {best_path}. Did the run finish and save best_genome.json?")

    hist = pd.read_csv(hist_path)
    best = json.loads(best_path.read_text(encoding="utf-8"))

    # If fitness = -E (VQE), show energy estimate too
    hist["best_energy_est"] = -hist["best_fitness"]
    hist["mean_energy_est"] = -hist["mean_fitness"]

    # 1) Fitness curves
    plt.figure()
    plt.plot(hist["gen"], hist["best_fitness"], label="best fitness")
    plt.plot(hist["gen"], hist["mean_fitness"], label="mean fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Progress (Fitness)")
    plt.legend()
    savefig(out_dir / "01_fitness.png")

    # 2) Energy curves
    plt.figure()
    plt.plot(hist["gen"], hist["best_energy_est"], label="best energy (=-fitness)")
    plt.plot(hist["gen"], hist["mean_energy_est"], label="mean energy (=-fitness)")
    plt.xlabel("Generation")
    plt.ylabel("Energy (Hartree, estimated)")
    plt.title("VQE Progress (Energy)")
    plt.legend()
    savefig(out_dir / "02_energy.png")

    # 3) Complexity
    plt.figure()
    plt.plot(hist["gen"], hist["best_depth"], label="best depth")
    plt.plot(hist["gen"], hist["best_n2q"], label="best #2Q gates")
    plt.xlabel("Generation")
    plt.ylabel("Complexity")
    plt.title("Best Circuit Complexity")
    plt.legend()
    savefig(out_dir / "03_complexity.png")

    # 4) Diversity proxy
    if "unique" in hist.columns:
        plt.figure()
        plt.plot(hist["gen"], hist["unique"])
        plt.xlabel("Generation")
        plt.ylabel("Unique genomes (proxy)")
        plt.title("Population Diversity (Proxy)")
        savefig(out_dir / "04_diversity.png")

    # 5) Colormap scatter: depth vs fitness colored by generation
    plt.figure()
    sc = plt.scatter(hist["best_depth"], hist["best_fitness"], c=hist["gen"])
    plt.xlabel("Best depth")
    plt.ylabel("Best fitness")
    plt.title("Architecture Tradeoff Map (color = generation)")
    cb = plt.colorbar(sc)
    cb.set_label("Generation")
    savefig(out_dir / "05_tradeoff_colormap.png")

    # 6) Run signature heatmap (normalized metrics) — NumPy 2.0 safe
    cols = ["best_fitness", "mean_fitness", "best_depth", "best_n2q", "unique"]
    present = [c for c in cols if c in hist.columns]
    M = hist[present].to_numpy(dtype=float)

    M_min = M.min(axis=0, keepdims=True)
    M_ptp = np.ptp(M, axis=0, keepdims=True)  # ✅ FIXED
    M_norm = (M - M_min) / (M_ptp + 1e-12)

    plt.figure()
    plt.imshow(M_norm.T, aspect="auto")
    plt.yticks(np.arange(len(present)), present)
    plt.xlabel("Generation index")
    plt.title("Run Signature Heatmap (normalized)")
    cb = plt.colorbar()
    cb.set_label("Normalized value")
    savefig(out_dir / "06_signature_heatmap.png")

    # 7) Best genome gate composition
    counts = gate_counts(best)
    if counts:
        names = sorted(counts.keys())
        vals = [counts[n] for n in names]
        plt.figure()
        plt.bar(names, vals)
        plt.xlabel("Gate type")
        plt.ylabel("Count")
        plt.title("Best Genome: Gate Composition")
        savefig(out_dir / "07_gate_composition.png")

    # Summary text
    best_idx = int(hist["best_fitness"].idxmax())
    summary = [
        f"Run: {run_dir.name}",
        f"Generations: {int(hist['gen'].max()) + 1}",
        f"Best fitness: {float(hist.loc[best_idx, 'best_fitness']):.6f}",
        f"Best energy (est): {float(hist.loc[best_idx, 'best_energy_est']):.6f} Hartree",
        f"Best depth: {int(hist.loc[best_idx, 'best_depth'])}",
        f"Best #2Q gates: {int(hist.loc[best_idx, 'best_n2q'])}",
        f"Plots saved to: {out_dir}",
    ]
    (out_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("Saved plots to:", out_dir)

if __name__ == "__main__":
    main()
