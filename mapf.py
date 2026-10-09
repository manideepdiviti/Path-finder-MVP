"""Q2: Multi-robot path finding.

Three planners are provided:

1. plain_astar()      - every robot runs ordinary A* independently (baseline).
                        Fast, but robots collide (vertex + swap conflicts).
2. fixed_priority()   - classic prioritized planning: robots planned one by one in
                        index order with space-time A* and a reservation table.
3. adaptive_priority()- PROPOSED METHOD (AP-STA*). Same reservation idea, plus:
     a) exact-distance heuristic (one reverse BFS per robot) instead of Manhattan,
     b) "hardest-first" initial priority (longest shortest-path goes first),
     c) goal-parking reservations - a robot that reached its goal keeps blocking
        that cell, and a robot is not allowed to finish on a cell that an
        earlier robot still needs to cross later,
     d) conflict-driven priority repair - if a robot cannot be planned, it is
        promoted to the front of the order and everything is re-planned; cycles
        in the ordering are broken by a random shuffle.
"""
import heapq
import random
import time
from collections import Counter, deque

import numpy as np

from astar import astar
from grid import DIRS

INF = 10 ** 9


# -------------------------------------------------------------- utilities
def bfs_dist(grid, goal):
    """True shortest distance from every cell to `goal` (reverse BFS)."""
    rows, cols = grid.shape
    dist = np.full(grid.shape, INF, dtype=np.int64)
    dist[goal] = 0
    q = deque([goal])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0 and dist[nx, ny] == INF:
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
    return dist


def _at(path, t):
    return path[min(t, len(path) - 1)]  # robots wait at their goal after finishing


def count_conflicts(paths):
    """Returns (vertex_conflicts, swap_conflicts) over the whole timeline."""
    T = max(len(p) for p in paths)
    vertex = swap = 0
    for t in range(T):
        pos = [_at(p, t) for p in paths]
        for c in Counter(pos).values():
            vertex += c * (c - 1) // 2
        if t + 1 < T:
            nxt = [_at(p, t + 1) for p in paths]
            moves = {(a, b) for a, b in zip(pos, nxt) if a != b}
            swap += sum(1 for (a, b) in moves if (b, a) in moves) // 2
    return vertex, swap


def path_cost(path, goal):
    """Time at which the robot arrives at its goal for the last time."""
    i = len(path) - 1
    while i > 0 and path[i - 1] == goal:
        i -= 1
    return i


def _summarise(paths, goals, runtime, expanded, attempts, require_conflict_free):
    if paths is None:
        return {"success": False, "paths": None, "soc": None, "makespan": None,
                "runtime": runtime, "expanded": expanded, "attempts": attempts,
                "vertex": None, "swap": None}
    v, s = count_conflicts(paths)
    costs = [path_cost(p, g) for p, g in zip(paths, goals)]
    return {"success": (v + s == 0) if require_conflict_free else True,
            "paths": paths, "soc": sum(costs), "makespan": max(costs),
            "runtime": runtime, "expanded": expanded, "attempts": attempts,
            "vertex": v, "swap": s}


# ---------------------------------------------------------- 1. plain A*
def plain_astar(grid, starts, goals):
    t0 = time.perf_counter()
    paths, expanded = [], 0
    for s, g in zip(starts, goals):
        r = astar(grid, s, g, "Manhattan")
        if r["path"] is None:
            return _summarise(None, goals, time.perf_counter() - t0, expanded, 1, True)
        paths.append(r["path"])
        expanded += r["expanded"]
    # "success" for plain A* means: it happened to be conflict-free
    return _summarise(paths, goals, time.perf_counter() - t0, expanded, 1, True)


# ------------------------------------------------- reservation table + ST-A*
class Reservations:
    def __init__(self):
        self.vertex = set()   # (cell, t) occupied
        self.edge = set()     # (a, b, t): someone moves a->b between t and t+1
        self.last = {}        # cell -> last time it is reserved by a moving robot
        self.park = {}        # cell -> time from which a robot sits there forever

    def add(self, path, goal):
        for t, cell in enumerate(path):
            self.vertex.add((cell, t))
            self.last[cell] = max(self.last.get(cell, -1), t)
            if t + 1 < len(path) and path[t + 1] != cell:
                self.edge.add((cell, path[t + 1], t))
        self.park[goal] = path_cost(path, goal)


def st_astar(grid, start, goal, dist, R, max_t, budget):
    """Space-time A*. `dist` = exact distance-to-goal map (admissible & consistent).
    Returns (path or None, expansions_used)."""
    rows, cols = grid.shape
    if dist[start] >= INF:
        return None, 0
    counter, expanded = 0, 0
    h0 = int(dist[start])
    open_heap = [(h0, h0, 0, start, 0)]
    parent = {(start, 0): None}
    closed = set()
    while open_heap:
        _f, _h, _c, cell, t = heapq.heappop(open_heap)
        if (cell, t) in closed:
            continue
        closed.add((cell, t))
        expanded += 1
        if expanded > budget:
            return None, expanded
        # goal reached and nobody planned earlier needs this cell afterwards
        if cell == goal and t >= R.last.get(goal, -1):
            path, node = [], (cell, t)
            while node is not None:
                path.append(node[0])
                node = parent[node]
            path.reverse()
            return path, expanded
        if t >= max_t:
            continue
        nt = t + 1
        for dx, dy in DIRS + [(0, 0)]:  # 4 moves + wait
            nb = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols) or grid[nb]:
                continue
            if (nb, nt) in R.vertex or R.park.get(nb, INF) <= nt:  # vertex conflict / parked robot
                continue
            if (nb, cell, t) in R.edge:                              # swap conflict
                continue
            if nt + dist[nb] > max_t or (nb, nt) in closed:
                continue
            if (nb, nt) not in parent:  # g = t is fixed per state, so any parent is fine
                parent[(nb, nt)] = (cell, t)
            counter += 1
            hn = int(dist[nb])
            heapq.heappush(open_heap, (nt + hn, hn, counter, nb, nt))
    return None, expanded


def _prioritized(grid, starts, goals, adaptive, budget, seed=0):
    t0 = time.perf_counter()
    k = len(starts)
    rng = random.Random(seed)
    dists = [bfs_dist(grid, g) for g in goals]
    if any(dists[i][starts[i]] >= INF for i in range(k)):
        return _summarise(None, goals, time.perf_counter() - t0, 0, 0, False)
    max_t = max(int(dists[i][starts[i]]) for i in range(k)) + 2 * k + 10

    if adaptive:  # hardest-first initial priority
        order = sorted(range(k), key=lambda i: -dists[i][starts[i]])
    else:         # plain index order
        order = list(range(k))

    attempts, total_exp, seen = 0, 0, set()
    max_attempts = 3 * k + 5 if adaptive else 1
    while attempts < max_attempts and total_exp < budget:
        attempts += 1
        R, paths, failed = Reservations(), [None] * k, None
        for idx in order:
            p, used = st_astar(grid, starts[idx], goals[idx], dists[idx], R, max_t, budget - total_exp)
            total_exp += used
            if p is None:
                failed = idx
                break
            paths[idx] = p
            R.add(p, goals[idx])
        if failed is None:
            return _summarise(paths, goals, time.perf_counter() - t0, total_exp, attempts, False)
        if not adaptive or order[0] == failed:  # can't repair further
            break
        order.remove(failed)                    # priority repair: promote failed robot
        order.insert(0, failed)
        if tuple(order) in seen:                # ordering cycle -> random restart
            rng.shuffle(order)
        seen.add(tuple(order))
    return _summarise(None, goals, time.perf_counter() - t0, total_exp, attempts, False)


def fixed_priority(grid, starts, goals, budget=200000):
    return _prioritized(grid, starts, goals, adaptive=False, budget=budget)


def adaptive_priority(grid, starts, goals, budget=200000):
    return _prioritized(grid, starts, goals, adaptive=True, budget=budget)


PLANNERS = {
    "Plain A*": plain_astar,
    "Fixed-priority ST-A*": fixed_priority,
    "Proposed AP-STA*": adaptive_priority,
}
