import sys
import numpy as np
import math


def cutoff(n,weight,dim):
    if(n < 128):
        return True
    if(dim == 0):
        return not(weight > 1/(n/3))
    if(dim == 1):
        return not(weight > 1/(math.log2(n)/2))
    if(dim == 2):
        return not(weight > 1/((n ** (1/2))/5))
    if(dim == 3):
        return not(weight > 1/((n ** (2/3))/(5)))
    if(dim == 4):
        return not(weight > 1/((n ** (3/4))/(5)))

def ELgenCompGraph(n):
    edges=[]
    for i in range(n):
        for j in range(n):
            if i < j:
                weight = np.random.rand()
                if(cutoff(n,weight,0)):
                    edges.append([[i,j], weight])
    g = [n,edges]
    return g


def ELgenHyperCube(n):
    edges=[]
    for i in range(n):
        for j in range(n):
            if i != j and math.log2(abs(i-j)).is_integer():
                #print("Condition fulfilled")
                #print("i, j:",i,j)
                weight = np.random.rand()
                if(cutoff(n,weight,1)):
                    edges.append([[i,j], weight])
    g = [n,edges]
    return g

def ELgenGeoCompGraph(n):
    x_values = np.random.uniform(low=0,high=1,size=n)
    y_values = np.random.uniform(low=0,high=1,size=n)

    edges=[]

    for i in range(n):
        for j in range(n):
            if i != j:
                weight = math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2))
                if(cutoff(n,weight,2)):
                    edges.append([[i,j], weight])
    g = [n,edges]
    return g

def ELgenGeoCubeGraph(n):
    x_values = np.random.uniform(low=0,high=1,size=n)
    y_values = np.random.uniform(low=0,high=1,size=n)
    z_values = np.random.uniform(low=0,high=1,size=n)

    edges=[]

    for i in range(n):
        for  j in range(n):
            if i != j:
                weight = math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2)+(abs(z_values[i-1]-y_values[j-1]) ** 2))
                if(cutoff(n,weight,3)):
                    edges.append([[i,j], weight])
    g = [n,edges]
    return g    

def ELgenGeoHyperCube(n):
    x_values = np.random.uniform(low=0,high=1,size=n)
    y_values = np.random.uniform(low=0,high=1,size=n)
    z_values = np.random.uniform(low=0,high=1,size=n)
    a_values = np.random.uniform(low=0,high=1,size=n)
    edges=[]

    for i in range(n):
        for  j in range(n):
            if i != j:
                weight = math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2)+(abs(z_values[i-1]-z_values[j-1]) ** 2)+(abs(a_values[i-1]-a_values[j-1]) ** 2))
                if(cutoff(n,weight,4)):
                    edges.append([[i,j], weight])
    g = [n,edges]
    return g

class UnionFind:
    def __init__(self, n):
        self.parentNode = list(range(n))
        self.rank = [0]*n

    def makeSet(self, x):
        self.parentNode[x] = x
    
    def find(self, x):
        if self.parentNode[x] != x:
            self.parentNode[x] = self.find(self.parentNode[x])
        # print('Found', self.parentNode[x])
        return self.parentNode[x]

    def link(self, x,y):
        if self.rank[x] > self.rank[y]:
            #print("WITHIN LINK")
            #print("We're inputting", x, "and", y)
            return self.link(y, x)
        elif self.rank[x] == self.rank[y]:
            self.rank[y] += 1
        self.parentNode[x] = y
        return y
    
    def union(self, x,y):
        #print("We're unionizing")
        #print(x,y)
        findx = self.find(x)
        findy = self.find(y)
        #print("WITHIN UNION")
        #print("We're inputting", findx, "and", findy)
        self.link(self.find(x), self.find(y))
    
def kruskals(graph):
    verts = graph[0]
    edges = graph[1]
    sortedEdges = sorted(edges, key=lambda edge: edge[1])
    #print(sortedEdges)

    mst = []

    unionfind_graph = UnionFind(verts)
    cost = 0

    for i in range(verts):
        unionfind_graph.makeSet(i)
    
    #e = [[u, v], weight]
    for e in sortedEdges:
        #print(e[0])
        #print("We're finding", e[0][0], ":", unionfind_graph.find(e[0][0]))
        #print("We're finding", e[0][1], ":", unionfind_graph.find(e[0][1]))
        if unionfind_graph.find(e[0][0]) != unionfind_graph.find(e[0][1]):
            mst.append(e)
            unionfind_graph.union(e[0][0], e[0][1])
            if len(mst) == verts - 1:
                break

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
            graph = ELgenCompGraph(numpoints)
        if(dimension == 1):
            graph = ELgenHyperCube(numpoints)
        if(dimension == 2):
            graph = ELgenGeoCompGraph(numpoints)
        if(dimension == 3):
            graph = ELgenGeoCubeGraph(numpoints)
        if(dimension == 4):
            graph = ELgenGeoHyperCube(numpoints)
        #print("New weight:",count_weight(prims_algo(graph,0)))
        curr_weight += count_weight(kruskals(graph))
        #print("Summed weight:",curr_weight)
    average = curr_weight / numtrials
    return [average, numpoints, numtrials, dimension]

#if(len(sys.argv) != 5):
    #print("Incorrect number of command lines. randmst terminal usage is: \n")
    #print("python randmst.py [flag] [numpoints] [numtrials] [dimension]")
    #sys.exit()
flag = int(sys.argv[1])
nump = int(sys.argv[2])
numt = int(sys.argv[3])
dim = int(sys.argv[4])
#start = time.time()
randomSample(nump,numt,dim)
#stop = time.time() - start
#print("Runtime:",stop,"seconds")