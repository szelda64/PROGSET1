import math
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from genGraphs.py import genCompGraph, genHyperCube, genGeoCompGraph, genGeoCubeGraph, genGeoHyperCube

def parent(v):
    return v // 2
def leftchild(v):
    return 2 * v + 1
def rightchild(v):
    return 2 * v + 2

def pushup(arr,s):
    result = arr
    end = len(result)
    start = s
    next = result[s]
    minchild = 2 * s + 1
    while minchild < end:
        rightchild = minchild + 1
        if rightchild < end and result[minchild] > result[rightchild]:
            minchild = rightchild
        result[s] = result[minchild]
        s = minchild
        minchild = 2 * s + 1
    result[s] = next
    pushdown(result,start,s)

def pushdown(arr,start,s):
    result = arr
    next = result[s]
    while s > start:
        max_pos = math.floor((s-1)/2)
        max_parent = result[max_pos]
        if next < max_parent:
            result[s] = max_parent
            s = max_pos
        else:
            break
    result[s] = next
    return result

def heapify(arr):
    result = arr
    stop = len(arr)
    for i in range((n // 2) - 1, 0,-1):
        result = pushup(result,i)
    return result
    

def prims_algo(graph, source):
    heap_graph = heapConstruct(graph)
    pqueue = {}
    s = source
    keylist = list(heap_graph.keys())
    ed = keylist
    vertices = []
    while ed:
        if(ed[0][0] not in vertices):
            vertices.append([0][0])
            ed.pop(0)
        if(ed[0][1] not in vertices):
            vertices.append([0][1])
            ed.pop(0)

    dist = {vertex: float('infinity') for vertex in vertices}
    dist[s] = 0
    mst_included = {vertex: False for vertex in vertices}
    while pqueue:
        curr_key = list(pqueue.keys)[0]
        curr_vert = curr_key[0]
        mst_included[curr_vert] = True
        for key, value in heap_graph():
            if key[0] not in mst_included.keys() and key[1] == curr_key[1] and dist[key[0]] > value:
                dist[key[0]] = value
                heap_graph.append(key, dist[key[0]])
    return heap_graph

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