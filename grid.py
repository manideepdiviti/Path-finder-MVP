"""Grid + random instance generation (single and multi robot).

Grid convention: numpy int8 array, 0 = free cell, 1 = obstacle.
Cells are (row, col) tuples. Robots move in 4 directions (or wait, in MAPF).
"""
import random
from collections import deque

import numpy as np

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def random_grid(rows, cols, density, rng):
    """Grid where each cell is an obstacle with probability `density`."""
    return np.array(
        [[1 if rng.random() < density else 0 for _ in range(cols)] for _ in range(rows)],
        dtype=np.int8,
    )


def largest_component(grid):
    """Cells of the biggest 4-connected free region (guarantees reachability)."""
    rows, cols = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    best = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 and not seen[r, c]:
                comp, q = [], deque([(r, c)])
                seen[r, c] = True
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in DIRS:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0 and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
                if len(comp) > len(best):
                    best = comp
    return best


def make_instance(rows, cols, density, n_agents=1, seed=None, max_tries=2000):
    """Random grid + random distinct starts/goals, every start can reach its goal.

    Returns (grid, starts, goals). For n_agents == 1 you can unpack with
    starts[0], goals[0]. Start-goal pairs are forced to be reasonably far apart
    (Manhattan >= (rows+cols)//3) so that instances are not trivial.
    """
    rng = random.Random(seed)
    min_d = (rows + cols) // 3
    for _ in range(max_tries):
        grid = random_grid(rows, cols, density, rng)
        comp = largest_component(grid)
        if len(comp) < 2 * n_agents:
            continue
        used, starts, goals, ok = set(), [], [], True
        for _a in range(n_agents):
            free = [c for c in comp if c not in used]
            s = rng.choice(free)
            used.add(s)
            cands = [c for c in comp if c not in used and abs(c[0] - s[0]) + abs(c[1] - s[1]) >= min_d]
            if not cands:
                ok = False
                break
            g = rng.choice(cands)
            used.add(g)
            starts.append(s)
            goals.append(g)
        if ok:
            return grid, starts, goals
    raise RuntimeError("Could not generate an instance - lower the density or agent count")
