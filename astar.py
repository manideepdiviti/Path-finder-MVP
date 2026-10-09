"""Q1: A* search on a 4-connected grid with several heuristics."""
import heapq
import math
import time

from grid import DIRS


# ---------------------------------------------------------------- heuristics
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def zero(a, b):  # h = 0  ->  A* degenerates to Dijkstra / uniform-cost search
    return 0


HEURISTICS = {
    "Manhattan": manhattan,
    "Euclidean": euclidean,
    "Chebyshev": chebyshev,
    "Dijkstra (h=0)": zero,
}
# All four are admissible AND consistent for 4-connected unit-cost moves, so A*
# returns an optimal path with each. They differ in how *informed* they are:
#   h=0 <= Chebyshev <= Euclidean <= Manhattan <= true distance.


# ------------------------------------------------------------------- search
def astar(grid, start, goal, heuristic="Manhattan", record_order=False):
    """Returns dict: path, cost, expanded, max_open, time, order (optional)."""
    h = HEURISTICS[heuristic] if isinstance(heuristic, str) else heuristic
    rows, cols = grid.shape
    t0 = time.perf_counter()

    h0 = h(start, goal)
    g = {start: 0}
    parent = {start: None}
    open_heap = [(h0, h0, 0, start)]  # (f, h, tie-counter, node) -> ties broken by smaller h
    closed = set()
    counter, expanded, max_open, order = 0, 0, 1, []
    path = None

    while open_heap:
        _f, _h, _c, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if record_order:
            order.append(node)
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            break
        for dx, dy in DIRS:
            nb = (node[0] + dx, node[1] + dy)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols) or grid[nb] or nb in closed:
                continue
            ng = g[node] + 1
            if ng < g.get(nb, 1e18):
                g[nb] = ng
                parent[nb] = node
                counter += 1
                hn = h(nb, goal)
                heapq.heappush(open_heap, (ng + hn, hn, counter, nb))
        max_open = max(max_open, len(open_heap))

    return {
        "path": path,
        "cost": (len(path) - 1) if path else None,
        "expanded": expanded,
        "max_open": max_open,
        "time": time.perf_counter() - t0,
        "order": order,
    }
