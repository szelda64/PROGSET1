import math
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from genGraphRehaul import Graph, genCompGraph, genHyperCube, genGeoCompGraph, genGeoCubeGraph, genGeoHyperCube

class Heap:
    def __init__(self):
        self.array = []

    def parent(self, v):
        return (v-1) // 2
    def leftchild(self, v):
        return 2 * v + 1
    def rightchild(self, v):
        return 2 * v + 2

    def insert(self, v):
        #print("\nCurrently running insert on heap for element: " + str(v))
        self.array.append(v)
        #print("Current heap: " + str(self.array))
        n = len(self.array) - 1
        #print("Index of last element: " + str(n))
        while (n > 0) and self.array[self.parent(n)][1] > self.array[n][1]:
            #print('self.array[self.parent(n)][1]:',self.array[self.parent(n)][1])
            #print('self.array[n][1]:',self.array[n][1])
            self.array[self.parent(n)], self.array[n] = self.array[n], self.array[self.parent(n)]
            n = self.parent(n)
        #print("Current heap: " + str(self.array),'\n')
 
    #the problem is here
    def minHeapify(self, v):
        #print("\nCurrently running minHeapify on heap for element: " + str(v))
        #print("Current array",self.array)
        n = self.array.index(v)
        #print("Position of element in array: " + str(n))
        smallest = n
        l,r = self.leftchild(n), self.rightchild(n)
        #print(n,"has children", l, "and", r)
        #print('Current heap:', self.array)
        if l < len(self.array): 
            if self.array[l] < self.array[n]:
                smallest = l
        else:
            smallest = n

        if r < len(self.array):
            if self.array[r] < self.array[smallest]:
                smallest = r
        #print('Smallest:', smallest, "\n")
        if smallest != n: 
            self.array[n], self.array[smallest] = self.array[smallest], self.array[n]
            self.minHeapify(self.array[smallest])

    def buildHeap(self):
        for i in range(len(self.array)/2, 1, -1):
            self.minHeapify(self, i)
    
    def extractMin(self):
        #print("\nCurrently running extractMin on heap")
        #print("Array before: " + str(self.array))
        min = self.array[0]
        self.array[0] = self.array[-1]
        self.array.pop()
        #print("Array after popping: " + str(self.array))
        #print("Length of array after popping: " + str(len(self.array)))
        if len(self.array) != 0:
            self.minHeapify(self.array[0])
        return min


    

def prims_algo(graph, source):
    verts = graph.verts
    print("Vertices:", verts)
    edges = graph.edges
    print("Edges: " + str(edges), "with length", len(edges)) 

    #initialize all distances to inf, aside from source
    dist = [math.inf]*verts
    dist[source] = 0
    prev = [-1]*verts
    print(dist)
    print(prev)
    
    mst = []
    heap_graph = Heap()

    heap_graph.insert([source, 0])

    print(heap_graph.array)

    while heap_graph.array:
        #u = [vertex label, vertex value]
        u = heap_graph.extractMin()

        if u not in mst:
            mst.append(u)
            
            for i in range(verts):
                if i in graph.edges[u]:
                    if i not in mst and graph.edges[u][i] > dist[i]:
                        dist[i] = graph.edges[u][i]
                        prev[i] = u
                        heap_graph.insert([i, dist[i]])

    return mst




    # while heap_graph.array:
    #     #print("Current heap: " + str(heap_graph.array))
    #     u = heap_graph.extractMin()
    #     #print('min:', u)
    #     #print('current array', heap_graph.array)
    #     mst.append(u)
    #     #print('mst:', mst)

        



        # #e = ((start, end), weight)
        # #e[0] = (start, end)
        # unvisitedAdjE = []
        # for e in edges:
        #     #for (u,v) in Edges
        #     if e[0][0] == u[0]:
        #         #for v not in mst
        #         mstv = [item[0] for item in mst]
        #         if e[0][1] not in mstv:
        #             unvisitedAdjE.append(e)
        # #print('Unvisited:', unvisitedAdjE)

    #     for e in unvisitedAdjE:
    #         #if dist[v] > weight
    #         if dist[e[0][1]] > e[1]:
    #             dist[e[0][1]] = e[1]
    #             prev[e[0][1]] = u
    #             #print([e[0][1], e[1]])
    #             heap_graph.insert([e[0][1], e[1]])

    # return (dist, prev)

    

def count_weight(graph):
    #print("\nRunning count weight on graph...")
    #print("mst: " + str(graph))
    sum = 0
    for i in graph[1]:
        #print("Current i in mst: " + str(i))
        if(i):
            sum += i[1]
    return sum

def randomSample(numpoints, numtrials, dimension):
    for _ in range(numtrials):
        curr_weight = 0
        if(dimension == 0):
            graph = genCompGraph(numpoints)
            curr_weight += count_weight(prims_algo(graph,0))
        if(dimension == 1):
            graph = genHyperCube(numpoints)
            curr_weight += count_weight(prims_algo(graph, 0))
        if(dimension == 2):
            graph = genGeoCompGraph(numpoints)
            curr_weight += count_weight(prims_algo(graph, 0))
        if(dimension == 3):
            graph = genGeoCubeGraph(numpoints)
            curr_weight += count_weight(prims_algo(graph, 0))
        if(dimension == 4):
            graph = genGeoHyperCube(numpoints)
            curr_weight += count_weight(prims_algo(graph, 0))
    average = curr_weight / numtrials
    print(average, numpoints, numtrials, dimension)