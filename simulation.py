"""Live simulations for the demo.

    python simulation.py q1 --live            # 4 heuristics exploring the same grid, side by side
    python simulation.py q2 --live            # plain A* (collisions!) vs proposed method
    python simulation.py q2 --live --agents 12 --size 15 --density 0.25 --seed 3

Without --live the animation is only saved as a GIF (in ./results). Add --save to
save the GIF as well while showing the live window.
"""
import argparse
import math
import os

import matplotlib
import numpy as np

from astar import HEURISTICS, astar
from grid import make_instance
from mapf import adaptive_priority, count_conflicts, plain_astar

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CMAP_GRID = matplotlib.colors.ListedColormap(["#f7f7f7", "#2b2b2b"])


def _setup_ax(ax, grid, title):
    ax.imshow(grid, cmap=CMAP_GRID, vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=0.4)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=11)


# ------------------------------------------------------------------- Q1
def simulate_q1(size, density, seed, live, save):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    grid, s, g = make_instance(size, size, density, 1, seed=seed)
    s, g = s[0], g[0]
    runs = {h: astar(grid, s, g, h, record_order=True) for h in HEURISTICS}
    step = max(1, math.ceil(max(r["expanded"] for r in runs.values()) / 120))
    n_frames = math.ceil(max(r["expanded"] for r in runs.values()) / step) + 25

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = {}
    for ax, h in zip(axes.ravel(), HEURISTICS):
        _setup_ax(ax, grid, h)
        overlay = np.zeros(grid.shape + (4,))
        im = ax.imshow(overlay)
        ax.plot(s[1], s[0], "o", color="limegreen", markersize=9, markeredgecolor="k")
        ax.plot(g[1], g[0], "*", color="red", markersize=14, markeredgecolor="k")
        line, = ax.plot([], [], color="royalblue", linewidth=2.5)
        panels[h] = (ax, overlay, im, line)
    fig.suptitle(f"A* with different heuristics - same {size}x{size} grid, start (green) -> goal (red)", fontsize=13)

    def update(frame):
        for h, (ax, overlay, im, line) in panels.items():
            r = runs[h]
            n = min(r["expanded"], (frame + 1) * step)
            for (x, y) in r["order"][max(0, n - step):n]:
                overlay[x, y] = (1.0, 0.65, 0.1, 0.55)
            im.set_data(overlay)
            done = n >= r["expanded"]
            if done:
                line.set_data([p[1] for p in r["path"]], [p[0] for p in r["path"]])
            ax.set_title(f"{h}\nnodes expanded: {n}" + (f"  |  path length {r['cost']}  DONE" if done else ""),
                         fontsize=10, color="darkgreen" if done else "black")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=70, blit=False, repeat=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _finish(anim, fig, "sim_q1.gif", live, save, PillowWriter)


# ------------------------------------------------------------------- Q2
def simulate_q2(size, density, agents, seed, live, save):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    grid, starts, goals = make_instance(size, size, density, agents, seed=seed)
    plain = plain_astar(grid, starts, goals)
    prop = adaptive_priority(grid, starts, goals)
    if not prop["success"]:
        raise SystemExit("Proposed method failed on this instance - try another --seed")
    T = max(max(len(p) for p in plain["paths"]), max(len(p) for p in prop["paths"])) + 2
    SUB = 6
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    def pos(path, t):
        return path[min(t, len(path) - 1)]

    def conflicts_at(paths, t):
        cells = {}
        for p in paths:
            cells.setdefault(pos(p, t), 0)
            cells[pos(p, t)] += 1
        hits = [c for c, n in cells.items() if n > 1]
        moves = {}
        for i, p in enumerate(paths):
            a, b = pos(p, t), pos(p, t + 1)
            if a != b:
                moves[(a, b)] = i
        hits += [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for (a, b) in moves if (b, a) in moves and a < b]
        return hits

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.3))
    sims = []
    for ax, name, res in zip(axes, ("Plain A* (independent paths)", "Proposed AP-STA*"), (plain, prop)):
        _setup_ax(ax, grid, name)
        for i, (s, g) in enumerate(zip(starts, goals)):
            ax.plot(g[1], g[0], "*", color=colors[i % 20], markersize=13, markeredgecolor="k", zorder=2)
        dots = ax.scatter([s[1] for s in starts], [s[0] for s in starts], s=170,
                          c=[colors[i % 20] for i in range(agents)], edgecolors="k", zorder=4)
        trails = [ax.plot([], [], color=colors[i % 20], alpha=0.5, linewidth=2, zorder=3)[0] for i in range(agents)]
        crash = ax.scatter([], [], marker="X", s=260, c="red", edgecolors="k", zorder=5)
        sims.append(dict(ax=ax, name=name, res=res, dots=dots, trails=trails, crash=crash, hits=[], seen=set()))
    fig.suptitle(f"{agents} robots on a {size}x{size} grid  (stars = goals, red X = collision)", fontsize=13)

    def update(frame):
        t, sub = divmod(frame, SUB)
        f = sub / SUB
        for sm in sims:
            paths = sm["res"]["paths"]
            xy, hist_len = [], t + 1
            for i, p in enumerate(paths):
                a, b = pos(p, t), pos(p, t + 1)
                xy.append((a[1] + (b[1] - a[1]) * f, a[0] + (b[0] - a[0]) * f))
                cells = [pos(p, k) for k in range(hist_len)]
                sm["trails"][i].set_data([c[1] for c in cells], [c[0] for c in cells])
            sm["dots"].set_offsets(xy)
            if sub == 0 and t not in sm["seen"]:
                sm["seen"].add(t)
                sm["hits"] += conflicts_at(paths, t)
            if sm["hits"]:
                sm["crash"].set_offsets([(c[1], c[0]) for c in sm["hits"]])
            arrived = sum(1 for p, g in zip(paths, goals) if pos(p, t) == g)
            sm["ax"].set_title(f"{sm['name']}\nt = {t}   |   at goal: {arrived}/{agents}   |   collisions: {len(sm['hits'])}",
                               fontsize=11, color="darkred" if sm["hits"] else "darkgreen")
        return []

    anim = FuncAnimation(fig, update, frames=T * SUB, interval=40, blit=False, repeat=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    v, w = plain["vertex"], plain["swap"]
    print(f"Plain A*: {v} vertex + {w} swap conflicts | Proposed: {prop['vertex']}+{prop['swap']} "
          f"(sum of costs {prop['soc']}, {prop['attempts']} planning attempt(s))")
    _finish(anim, fig, "sim_q2.gif", live, save, PillowWriter)


def _finish(anim, fig, gif_name, live, save, PillowWriter):
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    if save or not live:
        path = os.path.join(OUT, gif_name)
        anim.save(path, writer=PillowWriter(fps=20))
        print("Saved", path)
    if live:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["q1", "q2"])
    ap.add_argument("--live", action="store_true", help="open an interactive window")
    ap.add_argument("--save", action="store_true", help="also save GIF when using --live")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--agents", type=int, default=10)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    if not a.live:
        matplotlib.use("Agg")
    if a.which == "q1":
        simulate_q1(a.size or 30, a.density if a.density is not None else 0.25, a.seed, a.live, a.save)
    else:
        simulate_q2(a.size or 12, a.density if a.density is not None else 0.2, a.agents, a.seed, a.live, a.save)
