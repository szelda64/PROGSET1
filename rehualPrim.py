import math
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from genGraphRehaul import AMGraph, AMgenCompGraph, AMgenHyperCube, AMgenGeoCompGraph, AMgenGeoCubeGraph, AMgenGeoHyperCube, ELGraph, ELgenCompGraph, ELgenHyperCube, ELgenGeoCompGraph, ELgenGeoCubeGraph, ELgenGeoHyperCube

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
  
class UnionFind:
    def __init__(self, n, x):
        self.parentNode = list(range(n))
        self.parentNode[x] = x
        self.rank = [0]*n
    
    def find(self, x):
        if self.parentNode[x] != x:
            self.parentNode[x] = self.find(self.parentNode[x])
        return self.parentNode[x]

    def link(self, x,y):
        if self.rank[x] > self.rank[y]:
            return self.link(self, y, x)
        elif self.rank[x] == self.rank[y]:
            self.rank[y] += 1
        self.parentNode[x] = y
        return y
    
    def union(self, x,y):
        self.link(self.find(x), self.find(y))
        
def prims_algo(graph, source):
    verts = graph.verts
    #print("Vertices:", verts)
    edges = graph.edges
    #print("Edges: " + str(edges), "with length", len(edges)) 

    #initialize all distances to inf, aside from source
    dist = [math.inf]*verts
    dist[source] = 0
    prev = [-1]*verts
    #print(dist)
    #print(prev)
    
    mst = []
    in_mst = [False]*verts
    heap_graph = Heap()

    heap_graph.insert([source, 0])

    #print(heap_graph.array)

    while heap_graph.array:
        #u = [vertex label, vertex value]
        u = heap_graph.extractMin()
        #print('Min:', u)

        if not in_mst[u[0]]:

            mst.append(u)
            in_mst[u[0]] = True
            
            for i in range(verts):
                #print("We check if", i, "is key in", edges[u[0]])
                if i in edges[u[0]]:
                    #print("Tis")
                    #print(i, "has weight", graph.edges[u[0]][i])
                    if i not in mst and graph.edges[u[0]][i] < dist[i]:
                        dist[i] = graph.edges[u[0]][i]
                        prev[i] = u[0]
                        heap_graph.insert([i, dist[i]])
            #print(heap_graph.array,'\n')

    return mst


    

def count_weight(graph):
    #print("\nRunning count weight on graph...")
    #print("mst: " + str(graph))
    sum = 0
    for i in graph:
        #print("Current i in mst: " + str(i))
        if(i):
            sum += i[1]
    return sum

def randomSample(numpoints, numtrials, dimension):
    curr_weight = 0
    for _ in range(numtrials):
        if(dimension == 0):
            graph = AMgenCompGraph(numpoints)
        if(dimension == 1):
            graph = AMgenHyperCube(numpoints)
        if(dimension == 2):
            graph = AMgenGeoCompGraph(numpoints)
        if(dimension == 3):
            graph = AMgenGeoCubeGraph(numpoints)
        if(dimension == 4):
            graph = AMgenGeoHyperCube(numpoints)
        #print("New weight:",count_weight(prims_algo(graph,0)))
        curr_weight += count_weight(prims_algo(graph,0))
        #print("Summed weight:",curr_weight)
    average = curr_weight / numtrials
    print(average, numpoints, numtrials, dimension)
    return average
def figureGen(numtrials, dimension):
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    MSTweights = []
    if(dimension == 1):
        sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    for n in sizes:
        MSTweights.append(randomSample(n,numtrials,dimension))
    plt.figure(figsize=(10,6))
    plt.plot(sizes,MSTweights,label="Average Weights",marker='o')
    plt.xlabel("Number of Vertices (n)")
    plt.ylabel("Average Weight")
    plt.title("Average Type",dimension,"Graph Weights Over Large Values of n")
    plt.legend()
    plt.grid(True)
    plt.show()