import math
import random
import numpy as np
import time
import matplotlib.pyplot as plt

def bellmanFord(graph, source):
    vertices = set()
    for u,v in graph:
        vertices.add(u)
        vertices.add(v)
    dist = {vertex: float('infinity') for vertex in vertices}
    dist[source] = 0
    prev = {source: None}

    for i in range(len(dist)-1):
        for u,v in graph:
            if dist[u] + graph[(u,v)] < dist[v]:
                prev[v] = u
                dist[v] = dist[u] + graph[(u,v)]
    return dist, prev

def bellmanFordQueue(graph, source):
    vertices = set()
    for u, v in graph:
        vertices.add(u)
        vertices.add(v)
    dist = {vertex: float('infinity') for vertex in vertices}
    dist[1] = 0
    Q = [1]
    while Q != []:
        u = Q.pop(0)
        for v in vertices:
            if (u,v) in graph and dist[u] + graph[(u,v)] < dist[v]:
                dist[v] = dist[u] + graph[(u,v)]
                if v not in Q:
                    Q.append(v)
    
    return dist

def genGraph(n):
    graph = {}

    for i in range(1,n+1):
        for j in range(1, n+1):
            if i != j:
                graph[(i,j)] = graph[(j,i)] = np.random.rand()
    return graph

def randomSample():
    sizes = list(range(10,210,10))
    bf_times = []
    bfq_times = []

    for n in sizes:
        bf_total_time = 0
        bfq_total_time = 0

        for _ in range(75):
            graph = genGraph(n)
            start = time.time()
            bellmanFord(graph,0)
            bf_total_time += time.time() - start
            start = time.time()
            bellmanFordQueue(graph,0)
            bfq_total_time += time.time() - start
        bf_times.append(bf_total_time / 75)
        bfq_times.append(bfq_total_time / 75)
    plt.figure(figsize=(10,6))
    plt.plot(sizes,bf_times,label="Bellman-Ford",marker='o')
    plt.plot(sizes,bfq_times,label="Bellman-Ford Queue", marker='s')
    plt.xlabel("Number of Vertices (n)")
    plt.ylabel("Average Runtime (seconds)")
    plt.title("Bellman-Ford vs. Bellman-Ford Queue Performance")
    plt.legend()
    plt.grid(True)
    plt.show()