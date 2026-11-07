"""
Geodesic (obstacle- and hazard-aware) distance to the target.

Shortest 8-connected path cost (cardinal 1, diagonal sqrt 2, no corner
cutting) through free, currently dry cells, computed with SciPy's C Dijkstra
on a sparse graph whose static edge list is built once per map.
Used by the v1.1 training reward (progress along the geodesic, not octile).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class GeodesicField:
    def __init__(self, blocked: np.ndarray):
        h, w = blocked.shape
        self.h, self.w = h, w
        free = ~blocked.astype(bool)
        idx = np.arange(h * w).reshape(h, w)
        us, vs, ws, c1, c2 = [], [], [], [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            for y in range(h):
                for x in range(w):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < w and 0 <= ny < h) or not free[y, x] or not free[ny, nx]:
                        continue
                    if dx and dy:
                        if not (free[y, nx] and free[ny, x]):
                            continue
                        c1.append(idx[y, nx])
                        c2.append(idx[ny, x])
                        ws.append(np.sqrt(2.0))
                    else:
                        c1.append(idx[y, x])
                        c2.append(idx[y, x])
                        ws.append(1.0)
                    us.append(idx[y, x])
                    vs.append(idx[ny, nx])
        self.u, self.v = np.array(us), np.array(vs)
        self.c1, self.c2 = np.array(c1), np.array(c2)
        self.wt = np.array(ws)

    def distances(self, target, flood: np.ndarray) -> np.ndarray:
        """(H, W) geodesic distance from every cell to ``target`` avoiding flooded cells (inf if cut off)."""
        dry = ~flood.astype(bool).ravel()
        ok = dry[self.u] & dry[self.v] & dry[self.c1] & dry[self.c2]
        n = self.h * self.w
        g = csr_matrix((self.wt[ok], (self.u[ok], self.v[ok])), shape=(n, n))
        d = dijkstra(g, directed=False, indices=target[1] * self.w + target[0])
        return d.reshape(self.h, self.w)
