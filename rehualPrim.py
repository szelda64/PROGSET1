import math
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from genGraphs import genCompGraph, genHyperCube, genGeoCompGraph, genGeoCubeGraph, genGeoHyperCube

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
        self.array.append(v)
        print(self.array)
        n = len(self.array) - 1
        print(n)
        while (n > 0) and self.array[self.parent(n)][1] > self.array[n][1]:
            print('self.array[self.parent(n)][1]:',self.array[self.parent(n)][1])
            print('self.array[n][1]:',self.array[n][1])
            self.array[self.parent(n)], self.array[n] = self.array[n], self.array[self.parent(n)]
            n = self.parent(n)
        print(self.array,'\n')
 
    #the problem is here
    def minHeapify(self, v):
        n = self.array.index(v)
        l,r = self.leftchild(n), self.rightchild(n)
        print(n,"has children", l, "and", r)
        print('Current heap:', self.array)
        
        if l < len(self.array): 
            if self.array[l] < self.array[n]:
                smallest = l
        else:
            smallest = n

        if r < len(self.array):
            if self.array[r] < self.array[smallest]:
                smallest = r
        print('Smallest:', smallest, "\n")
        if smallest != n: 
            self.array[n], self.array[smallest] = self.array[smallest], self.array[n]
            self.minHeapify(smallest)

    def buildHeap(self):
        for i in range(len(self.array)/2, 1, -1):
            self.minHeapify(self, i)
    
    def extractMin(self):
        min = self.array[0]
        self.array[0] = self.array[-1]
        self.array.pop()
        if len(self.array) != 0:
            self.minHeapify(self.array[0])
        return min


    

def prims_algo(graph, source):
    #graph = [[list of vertices], [list of ((start, end), weight)]]
    verts = graph[0]
    print(verts, "with length", len(verts))
    edges = graph[1] 

    #initialize all distances to inf, aside from source
    dist = [math.inf]*len(verts)
    prev = [None]*len(verts)
    dist[source] = 0
    
    mst = []
    heap_graph = Heap()

    heap_graph.insert([source, 0])

    while heap_graph:
        u = heap_graph.extractMin()
        print('min:', u)
        print(heap_graph.array)
        mst.append(u)
        print('mst:', mst)

        #e = ((start, end), weight)
        #e[0] = (start, end)
        unvisitedAdjE = []
        for e in edges:
            #for (u,v) in Edges
            if e[0][0] == u[0]:
                #for v not in mst
                mstv = [item[0] for item in mst]
                if e[0][1] not in mstv:
                    unvisitedAdjE.append(e)
        #print('Unvisited:', unvisitedAdjE)

        for e in unvisitedAdjE:
            #if dist[v] > weight
            if dist[e[0][1]] > e[1]:
                dist[e[0][1]] = e[1]
                prev[e[0][1]] = u
                print([e[0][1], e[1]])
                heap_graph.insert([e[0][1], e[1]])

    return (dist, prev)

    # heap_graph = heapConstruct(graph)
    # pqueue = {}
    # s = source
    # keylist = list(heap_graph.keys())
    # ed = keylist
    # vertices = []
    # while ed:
    #     if(ed[0][0] not in vertices):
    #         vertices.append([0][0])
    #         ed.pop(0)
    #     if(ed[0][1] not in vertices):
    #         vertices.append([0][1])
    #         ed.pop(0)

    # dist = {vertex: float('infinity') for vertex in vertices}
    # dist[s] = 0
    # mst_included = {vertex: False for vertex in vertices}
    # while pqueue:
    #     curr_key = list(pqueue.keys)[0]
    #     curr_vert = curr_key[0]
    #     mst_included[curr_vert] = True
    #     for key, value in heap_graph():
    #         if key[0] not in mst_included.keys() and key[1] == curr_key[1] and dist[key[0]] > value:
    #             dist[key[0]] = value
    #             heap_graph.append(key, dist[key[0]])
    # return heap_graph

def count_weight(graph):
    sum = 0
    for key in graph.keys:
        sum += graph[key]
    return sum

def randomSample():
    sizes = list(range(10,210,10))
    MST_weights0 = []
    MST_weights1 = []
    MST_weights2 = []
    MST_weights3 = []
    MST_weights4 = []

    for n in sizes:
        curr_weight0 = 0
        curr_weight1 = 0
        curr_weight2 = 0
        curr_weight3 = 0
        curr_weight4 = 0
        for _ in range(75):
            graph0 = genCompGraph(n)
            curr_weight0 += count_weight(prims_algo(graph0, 0))
            graph1 = genHyperCube(n)
            curr_weight1 += count_weight(prims_algo(graph1, 0))
            graph2 = genGeoCompGraph(n)
            curr_weight2 += count_weight(prims_algo(graph2, 0))
            graph3 = genGeoCubeGraph(n)
            curr_weight3 += count_weight(prims_algo(graph3, 0))
            graph4 = genGeoHyperCube(n)
            curr_weight4 += count_weight(prims_algo(graph4, 0))
        MST_weights0.append(curr_weight0 / 75)
        MST_weights1.append(curr_weight1 / 75)
        MST_weights2.append(curr_weight2 / 75)
        MST_weights3.append(curr_weight3 / 75)
        MST_weights4.append(curr_weight4 / 75)
    plt.figure(figsize=(10,6))
    plt.plot(sizes,MST_weights0,label="Type 0 Graphs",marker='o')
    plt.plot(sizes,MST_weights1,label="Type 1 Graphs", marker='g')
    plt.plot(sizes,MST_weights2,label="Type 2 Graphs", marker='b')
    plt.plot(sizes,MST_weights3,label="Type 3 Graphs", marker='c')
    plt.plot(sizes,MST_weights4,label="Type 4 Graphs", marker='k')
    plt.xlabel("Number of Vertices (n)")
    plt.ylabel("Average Weight")
    plt.title("Expected Weight of Type 0-4 Graphs Based on Size")
    plt.legend()
    plt.grid(True)
    plt.show()