import numpy as np
def genGraph0(n):
    graph = {}

    for i in range(1,n+1):
        for j in range(1, n+1):
            if i != j:
                graph[(i,j)] = graph[(j,i)] = np.random.rand()
    return graph

def genGraph1(n):
    return 0
def genGraph2(n):
    return 0
def genGraph3(n):
    return 0
def genGraph4(n):
    return 0


