import numpy as np
import math

class Graph:

    def __init__(self, n, edges = None):
        self.verts = n
        self.edges = [dict() for lst in edges] if edges is not None else [dict() for _ in range(n)]
        # self.edges = [set(lst) for lst in edges] if edges is not None else [set() for _ in range(N)]
        # self.colors = [c for c in colors] if colors is not None else [None for _ in range(N)]

    def add_vert(self):
        self.verts += 1
        self.edges.append(dict())

    def add_edge(self, u, v, w):
        assert(v not in self.edges[u])
        assert(u not in self.edges[v])
        self.edges[u][v] = w
        self.edges[v][u] = w
        return self
  
def generate_complete_graph(n):
    g = Graph(n)
    for i in range(n):
        for j in range(n):
            if i < j:
                g.add_edge(i, j, np.random.rand())
    return g