import numpy as np
import math
  
def genCompGraph(n):
    verts = []
    edges = []

    for i in range(n):
        verts.append(i)
        for j in range(n):
            if i != j:
                edges.append(((i,j),np.random.rand()))
    graph = [verts, edges]
    return graph

def genHyperCube(n):
    verts = []
    edges = []

    for i in range(n):
        verts.append(i)
        for j in range(n):
            if i != j and math.log2(abs(i-j)).is_integer():
                edges.append(((i,j),np.random.rand()))
    graph = [verts, edges]
    return graph

def genGeoCompGraph(n):
    verts = []
    edges = []
    x_values = []
    y_values = []
    for _ in range(n):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())
    for i in range(n):
        verts.append(i)
        for j in range(n):
            if i != j:
                edges.append(((i,j),math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2))))
    graph = [verts, edges]
    return graph

def genGeoCubeGraph(n):
    verts = []
    edges = []
    x_values = []
    y_values = []
    z_values = []
    for _ in range(n):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())
        z_values.append(np.random.rand())
    for i in range(n):
        verts.append(i)
        for j in range(n):
            if i != j:
                edges.append(((i,j),math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2)+(abs(z_values[i-1]-y_values[j-1]) ** 2))))
    graph = [verts, edges]
    return graph

def genGeoHyperCube(n):
    verts = []
    edges = []
    x_values = []
    y_values = []
    z_values = []
    for _ in range(n):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())
        z_values.append(np.random.rand())
    for i in range(n):
        verts.append(i)
        for j in range(n):
            if i != j and math.log2(abs(i-j)).is_integer():
                edges.append(((i,j), math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2)+(abs(z_values[i-1]-y_values[j-1]) ** 2))))
    graph = [verts, edges]
    return graph


