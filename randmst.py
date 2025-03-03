try: 
    import sys
    import numpy as np
    import math
    #Add as as many imports as you would like to check for.
except ModuleNotFoundError as e:
    print("-E- Failed to import prerequisite {}. Please install prior to running this script.".format(e.name))
    exit(1)

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
    return average, numpoints, numtrials, dimension

if(len(sys.argv) != 5):
    print("Incorrect number of command lines. randmst terminal usage is: \n")
    print("python randmst.py [flag] [numpoints] [numtrials] [dimension]")
    sys.exit()
try:
    global flag
    flag = int(sys.argv[1])
except ValueError:
    print("flag must be an integer.")
    sys.exit()
try:
    global nump
    nump = int(sys.argv[2])
except ValueError:
    print("numpoints must be an integer.")
    sys.exit()
try:
    global numt
    numt = int(sys.argv[3])
except ValueError:
    print("numtrials must be an integer.")
    sys.exit()
try:
    global dim
    dim = int(sys.argv[4])
except ValueError:
    print("dimension must be an integer.")
    sys.exit()
#start = time.time()
randomSample(nump,numt,dim)
#stop = time.time() - start
#print("Runtime:",stop,"seconds")