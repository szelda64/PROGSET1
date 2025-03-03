import numpy as np
import math

class AMGraph:

    def __init__(self, n, edges = None):
        self.verts = n
        self.edges = [dict() for _ in range(n)]

    def add_edge(self, u, v, w):
        if v not in self.edges[u] and u not in self.edges[v]:
            self.edges[u][v] = w
            self.edges[v][u] = w
        return self

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

  
def AMgenCompGraph(n):
    g = AMGraph(n)
    for i in range(n):
        for j in range(n):
            if i < j:
                weight = np.random.rand()
                if(cutoff(n,weight,0)):
                    g.add_edge(i, j, weight)
    return g

def AMgenHyperCube(n):
    g = AMGraph(n)
    for i in range(n):
        for j in range(n):
            if i != j and math.log2(abs(i-j)).is_integer():
                #print("Condition fulfilled")
                #print("i, j:",i,j)
                weight = np.random.rand()
                if(cutoff(n,weight,1)):
                    g.add_edge(i, j, weight)
    return g

def AMgenGeoCompGraph(n):
    x_values = np.random.uniform(low=0,high=1,size=n)
    y_values = np.random.uniform(low=0,high=1,size=n)
    g = AMGraph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                weight = math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2))
                if(cutoff(n,weight,2)):
                    g.add_edge(i, j, weight)

    return g

def AMgenGeoCubeGraph(n):
    x_values = np.random.uniform(low=0,high=1,size=n)
    y_values = np.random.uniform(low=0,high=1,size=n)
    z_values = np.random.uniform(low=0,high=1,size=n)

    g = AMGraph(n)
    for i in range(n):
        for  j in range(n):
            if i != j:
                weight = math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2)+(abs(z_values[i-1]-y_values[j-1]) ** 2))
                if(cutoff(n,weight,3)):
                    g.add_edge(i, j, weight)

    return g    

def AMgenGeoHyperCube(n):
    x_values = np.random.uniform(low=0,high=1,size=n)
    y_values = np.random.uniform(low=0,high=1,size=n)
    z_values = np.random.uniform(low=0,high=1,size=n)
    a_values = np.random.uniform(low=0,high=1,size=n)

    g = AMGraph(n)
    for i in range(n):
        for  j in range(n):
            if i != j:
                weight = math.sqrt((abs(x_values[i-1]-x_values[j-1]) ** 2) + (abs(y_values[i-1] - y_values[j-1]) ** 2)+(abs(z_values[i-1]-z_values[j-1]) ** 2)+(abs(a_values[i-1]-a_values[j-1]) ** 2))
                g.add_edge(i, j, weight)

    return g

#################

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