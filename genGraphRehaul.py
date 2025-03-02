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
  
def genCompGraph(n):
    g = Graph(n)
    for i in range(n):
        for j in range(n):
            if i < j:
                g.add_edge(i, j, np.random.rand())
    return g

def genHyperCube(n):
    g = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j and math.log2(abs(i-j)).is_integer():
                g.add_edge(i, j, np.random.rand())
    return g

def genGeoCompGraph(n):
    x_values = []
    y_values = []
    for _ in range(n):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())

    g = Graph(n)
    for i in range(n):
        for  j in range(n):
            if i != j:
                weight = math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2))
                g.add_edge(i, j, weight)

    return g

def genGeoCubeGraph(n):
    x_values = []
    y_values = []
    z_values = []
    for _ in range(n):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())
        z_values.append(np.random.rand())

    g = Graph(n)
    for i in range(n):
        for  j in range(n):
            if i != j:
                weight = math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2)+(abs(z_values[i-1]-y_values[j-1]) ** 2))
                g.add_edge(i, j, weight)

    return g    

def genGeoHyperCube(n):
    x_values = []
    y_values = []
    z_values = []
    a_values = []
    for _ in range(n):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())
        z_values.append(np.random.rand())
        a_values.append(np.random.rand())

    g = Graph(n)
    for i in range(n):
        for  j in range(n):
            if i != j:
                weight = math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2)+(abs(z_values[i-1]-z_values[j-1]) ** 2)+(abs(a_values[i-1]-a_values[j-1]) ** 2))
                g.add_edge(i, j, weight)

    return g