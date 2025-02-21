import math
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from genGraphs.py import genGraph0, genGraph1, genGraph2, genGraph3, genGraph4

def parent(v):
    return v // 2
def leftchild(v):
    return 2 * v
def rightchild(v):
    return 2 * v + 1
def heapify(arr, s):
    min = s
    left = leftchild(s)
    right = rightchild(s)
    print("arr[min]" + str(arr[min]))
    if left < len(arr) and arr[left] < arr[min]:
        print("arr[left]: " + str(arr[left]))
        min = left
    if right < len(arr) and arr[right] < arr[min]:
        print("arr[right]: " + str(arr[right]))
        min = right
    if min != s:
        arr[s], arr[min] = arr[min], arr[s]
        return heapify(arr,min)
    else:
        return arr
        


def heapConstruct(graph):
    value_heap = []
    for i in graph.values():
        value_heap.append(i)
    value_heap = heapify(value_heap,1)
    print(value_heap)
    heap_graph = {}
    for i in range(len(value_heap)):
        for key in graph.keys():
            value = graph[key]
            if value == value_heap[i]:
                heap_graph[key] = value
    return heap_graph
    

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
            graph0 = genGraph0(n)
            curr_weight0 += count_weight(prims_algo(graph0, 0))
            graph1 = genGraph1(n)
            curr_weight1 += count_weight(prims_algo(graph1, 0))
            graph2 = genGraph2(n)
            curr_weight2 += count_weight(prims_algo(graph2, 0))
            graph3 = genGraph3(n)
            curr_weight3 += count_weight(prims_algo(graph3, 0))
            graph4 = genGraph4(n)
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