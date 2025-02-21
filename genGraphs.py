import numpy as np
import math
def genGraph0(n):
    graph = {}

    for i in range(1,n+1):
        for j in range(1, n+1):
            if i != j:
                graph[(i,j)] = graph[(j,i)] = np.random.rand()
    return graph

def genGraph1(n):
    graph = {}
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j and math.log2(math.abs(i-j)).is_integer():
                graph[(i,j)] = graph[(j,i)] = np.random.rand()
    return graph
def genGraph2(n):
    graph = {}
    x_values = []
    y_values = []
    for _ in range(1, n+1):
        x_values.append(np.random.rand())
        y_values.append(np.random.rand())
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j:
                graph[(i,j)] = graph[(j,i)] = math.sqrt((math.abs(x_values[i-1]-x_values[j-1]) ** 2) + (math.abs(y_values[i-1] - y_values[j-1]) ** 2))
    return graph
def genGraph3(n):
    return 0
def genGraph4(n):
    return 0


