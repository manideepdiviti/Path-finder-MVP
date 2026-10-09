"""Runs the experiments for Q1 (heuristics) and Q2 (multi-robot) and saves graphs + CSVs.

    python experiments.py q1            # heuristic comparison
    python experiments.py q2            # multi-robot comparison
    python experiments.py all --runs 50 # both, 50 random instances per configuration
    python experiments.py all --quick   # tiny run to test everything
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astar import HEURISTICS, astar
from grid import make_instance
from mapf import PLANNERS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ======================================================================= Q1
def q1_instance(args):
    rows, cols, density, seed = args
    grid, s, g = make_instance(rows, cols, density, 1, seed=seed)
    out = {}
    for name in HEURISTICS:
        r = astar(grid, s[0], g[0], name)
        out[name] = (r["expanded"], r["time"] * 1000, r["max_open"], r["cost"])
    return out


def q1_sweep(configs, runs, pool):
    """configs: list of (label, rows, cols, density). Returns {label: {heur: array(runs, 4)}}"""
    res = {}
    for label, rows, cols, density in configs:
        jobs = [(rows, cols, density, 1000 + i) for i in range(runs)]
        outs = list(pool.map(q1_instance, jobs))
        res[label] = {h: np.array([o[h] for o in outs], dtype=float) for h in HEURISTICS}
        print(f"  Q1 {label}: done")
    return res


def q1_plot(res, title, fname):
    metrics = [("Nodes expanded", 0), ("Runtime (ms)", 1), ("Peak open-list size", 2), ("Path length", 3)]
    labels = list(res)
    heurs = list(HEURISTICS)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    w = 0.8 / len(heurs)
    for ax, (mname, mi) in zip(axes, metrics):
        for j, h in enumerate(heurs):
            means = [res[l][h][:, mi].mean() for l in labels]
            errs = [res[l][h][:, mi].std() / np.sqrt(len(res[l][h])) for l in labels]
            ax.bar(np.arange(len(labels)) + j * w, means, w, yerr=errs, capsize=2, label=h)
        ax.set_xticks(np.arange(len(labels)) + w * (len(heurs) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_title(mname)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=140)
    plt.close(fig)


def q1_csv(res, fname):
    with open(os.path.join(OUT, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "heuristic", "avg_nodes_expanded", "avg_time_ms", "avg_peak_open", "avg_path_len"])
        for l, d in res.items():
            for h, a in d.items():
                w.writerow([l, h, *[round(x, 3) for x in a.mean(axis=0)]])


def run_q1(runs, pool):
    print("Q1: comparing heuristics")
    sizes = [(f"{n}x{n}", n, n, 0.25) for n in (20, 40, 60, 80)]
    r1 = q1_sweep(sizes, runs, pool)
    q1_plot(r1, f"Q1 - effect of grid size (25% obstacles, {runs} random grids each)", "q1_grid_size.png")
    q1_csv(r1, "q1_grid_size.csv")
    dens = [(f"{int(d*100)}%", 50, 50, d) for d in (0.10, 0.20, 0.30, 0.35)]
    r2 = q1_sweep(dens, runs, pool)
    q1_plot(r2, f"Q1 - effect of obstacle density (50x50 grid, {runs} random grids each)", "q1_density.png")
    q1_csv(r2, "q1_density.csv")


# ======================================================================= Q2
def q2_instance(args):
    rows, cols, density, k, seed = args
    try:
        grid, s, g = make_instance(rows, cols, density, k, seed=seed)
    except RuntimeError:
        return None
    out = {}
    for name, planner in PLANNERS.items():
        r = planner(grid, s, g)
        out[name] = dict(success=r["success"], soc=r["soc"], runtime=r["runtime"] * 1000,
                         conflicts=(r["vertex"] + r["swap"]) if r["vertex"] is not None else None)
    return out


def q2_sweep(configs, runs, pool):
    """configs: list of (label, rows, cols, density, k)."""
    res = {}
    for label, rows, cols, density, k in configs:
        jobs = [(rows, cols, density, k, 5000 + i) for i in range(runs)]
        outs = [o for o in pool.map(q2_instance, jobs) if o is not None]
        agg = {}
        for name in PLANNERS:
            recs = [o[name] for o in outs]
            ok = [r for r in recs if r["success"]]
            agg[name] = dict(
                n=len(recs),
                success=100.0 * len(ok) / max(1, len(recs)),
                soc=np.mean([r["soc"] for r in ok]) if ok else np.nan,
                runtime=np.mean([r["runtime"] for r in recs]),
                conflicts=np.mean([r["conflicts"] for r in recs if r["conflicts"] is not None] or [np.nan]),
            )
        res[label] = agg
        print(f"  Q2 {label}: " + ", ".join(f"{n}={a['success']:.0f}%" for n, a in agg.items()))
    return res


def q2_plot(res, xlabel, title, fname):
    labels = list(res)
    x = np.arange(len(labels))
    metrics = [("Success rate (%) - conflict-free plan found", "success"),
               ("Sum of costs (successful runs)", "soc"),
               ("Avg runtime (ms)", "runtime"),
               ("Avg #conflicts left in the plan", "conflicts")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    for ax, (mname, key) in zip(axes, metrics):
        for name, mk in zip(PLANNERS, ("s", "^", "o")):
            ax.plot(x, [res[l][name][key] for l in labels], marker=mk, label=name, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_title(mname, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].text(0.02, 0.02, "Plain A* ignores conflicts,\nso its cost is only a lower bound",
                 transform=axes[1].transAxes, fontsize=7, color="gray")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=140)
    plt.close(fig)


def q2_csv(res, fname):
    with open(os.path.join(OUT, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "planner", "instances", "success_pct", "avg_soc", "avg_runtime_ms", "avg_conflicts"])
        for l, d in res.items():
            for n, a in d.items():
                w.writerow([l, n, a["n"], round(a["success"], 1), round(a["soc"], 2),
                            round(a["runtime"], 2), round(a["conflicts"], 2)])


def run_q2(runs, pool):
    print("Q2: multi-robot")
    r = q2_sweep([(str(k), 15, 15, 0.25, k) for k in (5, 10, 15, 20, 25, 30)], runs, pool)
    q2_plot(r, "number of robots", f"Q2 - varying #robots (15x15, 25% obstacles, {runs} instances each)", "q2_agents.png")
    q2_csv(r, "q2_agents.csv")
    r = q2_sweep([(f"{n}x{n}", n, n, 0.25, 15) for n in (12, 15, 20, 25, 30)], runs, pool)
    q2_plot(r, "grid size", f"Q2 - varying grid size (15 robots, 25% obstacles, {runs} instances each)", "q2_grid_size.png")
    q2_csv(r, "q2_grid_size.csv")
    r = q2_sweep([(f"{int(d*100)}%", 20, 20, d, 15) for d in (0.10, 0.20, 0.30, 0.35)], runs, pool)
    q2_plot(r, "obstacle density", f"Q2 - varying obstacles (20x20, 15 robots, {runs} instances each)", "q2_density.png")
    q2_csv(r, "q2_density.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["q1", "q2", "all"])
    ap.add_argument("--runs", type=int, default=30, help="random instances per configuration")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    runs = 3 if a.quick else a.runs
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor() as pool:
        if a.which in ("q1", "all"):
            run_q1(runs, pool)
        if a.which in ("q2", "all"):
            run_q2(runs, pool)
    print("Saved graphs and CSVs in", OUT)
