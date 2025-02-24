import numpy as np
import math
def genCompGraph(n):
    verts = []
    edges = []

    for i in range(1,n+1):
        verts.append(i)
        for j in range(1, n+1):
            if i != j:
                edges.append(((i,j),np.random.rand()))
    graph = [verts, edges]
    return graph

def genHyperCube(n):
    verts = []
    edges = []

    for i in range(1, n+1):
        verts.append(i)
        for j in range(1, n+1):
            if i != j and math.log2(math.abs(i-j)).is_integer():
                edges.append(((i,j),np.random.rand()))
    graph = [verts, edges]
    return graph

def genGeoCompGraph(n):
    verts = []
    edges = []
    x_values = []
    y_values = []
    for _ in range(1, n+1):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())
    for i in range(1, n+1):
        verts.append(i)
        for j in range(1, n+1):
            if i != j:
                edges.append(((i,j),math.sqrt((math.abs(x_values[i-1]-x_values[j-1]) ** 2) + (math.abs(y_values[i-1] - y_values[j-1]) ** 2))))
    graph = [verts, edges]
    return graph

def genGeoCubeGraph(n):
    verts = []
    edges = []
    x_values = []
    y_values = []
    z_values = []
    for _ in range(1, n+1):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())
        z_values.append(np.random.rand())
    for i in range(1, n+1):
        verts.append(i)
        for j in range(1, n+1):
            if i != j:
                edges.append(((i,j),math.sqrt((math.abs(x_values[i-1]-x_values[j-1]) ** 2) + (math.abs(y_values[i-1] - y_values[j-1]) ** 2)+(math.abs(z_values[i-1]-y_values[j-1]) ** 2))))
    graph = [verts, edges]
    return graph

def genGeoHyperCube(n):
    verts = []
    edges = []
    x_values = []
    y_values = []
    z_values = []
    for _ in range(1, n+1):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())
        z_values.append(np.random.rand())
    for i in range(1, n+1):
        verts.append(i)
        for j in range(1, n+1):
            if i != j and math.log2(math.abs(i-j)).is_integer():
                edges.append(((i,j), math.sqrt((math.abs(x_values[i-1]-x_values[j-1]) ** 2) + (math.abs(y_values[i-1] - y_values[j-1]) ** 2)+(math.abs(z_values[i-1]-y_values[j-1]) ** 2))))
    graph = [verts, edges]
    return graph


