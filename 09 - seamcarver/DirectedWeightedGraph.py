from pathlib import Path
from queue import PriorityQueue
from queue import Queue
import timeit

class DirectedEdge:
    def __init__(self, v, w, weight): # v(정점1), w(정점2), weight(가중치)
        self.v, self.w, self.weight = v, w, weight
    
    def __lt__(self, other): # 두 DirectedEdge 객체의 대소 비교 (PQ에 저장하거나 정렬) : <
        assert(isinstance(other, DirectedEdge))
        return self.weight < other.weight

    def __gt__(self, other): # 두 DirectedEdge 객체의 대소 비교: >   
        assert(isinstance(other, DirectedEdge))
        return self.weight > other.weight

    def __eq__(self, other): # 두 DirectedEdge 객체의 대소 비교: ==  
        if other == None: return False
        assert(isinstance(other, DirectedEdge))
        return self.v == other.v and self.w == other.w and self.weight == other.weight

    def __str__(self): # DirectedEdge 객체 출력에 이용 
        return f"{self.v}->{self.w} ({self.weight})"

    def __repr__(self): # Called when an Edge instance is printed as an element of a list
        return self.__str__()    

 
class EdgeWeightedDigraph:
    def __init__(self, V): 
        self.V = V # vertex(정점) 개수
        self.E = 0 # edge(간선) 개수
        self.adj = [[] for _ in range(V)] 
        self.edges = [] 

    def addEdge(self, v, w, weight):
        e = DirectedEdge(v, w, weight) # 객체에 새로운 간선 추가 v->w
        self.adj[v].append(e) 
        self.edges.append(e)
        self.E += 1 # edge 총 개수 +1
    
    def outDegree(self, v):
        return len(self.adj[v])

    def __str__(self): # 그래프 출력 시 사용하는 함수
        rtList = [f"{self.V} vertices and {self.E} edges\n"]
        for v in range(self.V):
            for e in self.adj[v]: rtList.append(f"{e}\n")
        return "".join(rtList)

    def negate(self): # return an EdgeWeightedDigraph with all edge weights negated
        g = EdgeWeightedDigraph(self.V)
        for e in self.edges: g.addEdge(e.v, e.w, -e.weight)
        return g

    def reverse(self): # 모든 방향을 반대로 바꿔주는 함수
        g = EdgeWeightedDigraph(self.V)
        for e in self.edges: g.addEdge(e.w, e.v, e.weight)
        return g
 
    @staticmethod # 파일에서 그래프 정보 읽어와서 EdgeWeightedDigraph 그래프 객체 만들어 반환하는 함수
    def fromFile(fileName):
        filePath = Path(__file__).with_name(fileName)   # Use the location of the current .py file   
        with filePath.open('r') as f:
            phase = 0
            line = f.readline().strip() # Read a line, while removing preceding and trailing whitespaces
            while line:                                
                if len(line) > 0:
                    if phase == 0: # Read V, the number of vertices
                        g = EdgeWeightedDigraph(int(line))
                        phase = 1
                    elif phase == 1: # Read edges
                        edge = line.split()
                        if len(edge) != 3: raise Exception(f"Invalid edge format found in {line}")
                        g.addEdge(int(edge[0]), int(edge[1]), float(edge[2]))                        
                line = f.readline().strip()
        return g


'''
Min Priority Queue based on a binary heap 
    with decreaseKey operation added
'''
class IndexMinPQ:
    def __init__(self, maxN): # Create an indexed PQ with indices 0 to (N-1)
        if maxN < 0: raise Exception("maxN < 0")
        self.maxN = maxN # Max number of elements on PQ
        self.n = 0 # Number of elements on PQ
        self.keys = [None] * (maxN+1)  # keys[i]: key with index i
        self.pq = [-1] * (maxN+1)  # pq[i]: index of the key at heap position i (pq[0] is not used)        
        self.qp = [-1] * maxN # qp[i]: heap position of the key with index i (inverse of pq[])        

    def isEmpty(self):
        return self.n == 0

    def contains(self, i): # Is i an index on the PQ?
        self.validateIndex(i)
        return self.qp[i] != -1

    def size(self):
        return self.n

    def insert(self, i, key): # Associate key with index i
        self.validateIndex(i)
        if self.contains(i): raise Exception(f"index {i} is already in PQ")
        self.n += 1
        self.qp[i] = self.n
        self.pq[self.n] = i
        self.keys[i] = key
        self.swimUp(self.n)

    def minIndex(self): # Index associated with the minimum key
        if self.n == 0: raise Exception("PQ has no element, so no min index exists")
        return self.pq[1]

    def minKey(self):
        if self.n == 0: raise Exception("PQ has no element, so no min key exists")
        return self.keys[self.pq[1]]

    def delMin(self):
        if self.n == 0: raise Exception("PQ has no element, so no element to delete")
        minIndex = self.pq[1]
        minKey = self.keys[minIndex]
        self.exch(1, self.n)
        self.n -= 1
        self.sink(1)
        assert(minIndex == self.pq[self.n+1])
        self.qp[minIndex] = -1 # Mark the index as being deleted
        self.keys[minIndex] = None
        self.pq[self.n+1] = -1
        return minKey, minIndex

    def keyOf(self, i):
        self.validateIndex(i)
        if not self.contains(i): raise Exception(f"index {i} is not in PQ")
        else: return self.keys[i]

    def changeKey(self, i, key):
        self.validateIndex(i)
        if not self.contains(i): raise Exception(f"index {i} is not in PQ")
        self.keys[i] = key
        self.swimUp(self.qp[i])
        self.sink(self.qp[i])

    def decreaseKey(self, i, key):
        self.validateIndex(i)
        if not self.contains(i): raise Exception(f"index {i} is not in PQ")
        if self.keys[i] == key: raise Exception(f"calling decreaseKey() with key {key} equal to the previous key")
        if self.keys[i] < key: raise Exception(f"calling decreaseKey() with key {key} greater than the previous key {self.keys[i]}")
        self.keys[i] = key
        self.swimUp(self.qp[i])

    def increaseKey(self, i, key):
        self.validateIndex(i)
        if not self.contains(i): raise Exception(f"index {i} is not in PQ")
        if self.keys[i] == key: raise Exception(f"calling increaseKey() with key {key} equal to the previous key")
        if self.keys[i] > key: raise Exception(f"calling increaseKey() with key {key} smaller than the previous key {self.keys[i]}")
        self.keys[i] = key
        self.sink(self.qp[i])

    def delete(self, i):
        self.validateIndex(i)
        if not self.contains(i): raise Exception(f"index {i} is not in PQ")
        idx = self.qp[i]
        self.exch(idx, self.n)
        self.n -= 1
        self.swimUp(idx)
        self.sink(idx)
        self.keys[i] = None
        self.qp[i] = -1   

    def validateIndex(self, i):
        if i < 0: raise Exception(f"index {i} < 0")
        if i >= self.maxN: raise Exception(f"index {i} >= capacity {self.maxN}")

    def greater(self, i, j):
        return self.keys[self.pq[i]] > self.keys[self.pq[j]]

    def exch(self, i, j):
        self.pq[i], self.pq[j] = self.pq[j], self.pq[i]
        self.qp[self.pq[i]] = i
        self.qp[self.pq[j]] = j

    def swimUp(self, idx): # idx is the index in pq[]
        while idx>1 and self.greater(idx//2, idx):
            self.exch(idx, idx//2)            
            idx = idx//2

    def sink(self, idx): # idx is the index in pq[]
        while 2*idx <= self.n:    # If a child exists
            idxChild = 2*idx # Left child
            if idxChild<self.n and self.greater(idxChild, idxChild+1): idxChild = idxChild+1 # Find the smaller child
            if not self.greater(idx, idxChild): break
            self.exch(idx, idxChild) # Swap with (i.e., sink to) the greater child
            idx = idxChild


# toplogicalSort에 Cycle 있는지 탐지
def topologicalSortWithCycleDetection(g):
    def recur(v):        
        visited[v] = True
        verticesInRecurStack.add(v)
        for e in g.adj[v]:
            if e.w in verticesInRecurStack: # Edge found to a vertex in the recursive stack
                print("cycle detected on vertex", e.w)                
                return False 
            if not visited[e.w]: 
                if not recur(e.w): return False
        reverseList.append(v) # Add v to the stack if all adjacent vertices were visited
        verticesInRecurStack.remove(v)
        return True

    assert(isinstance(g, EdgeWeightedDigraph))
    visited = [False for _ in range(g.V)]
    reverseList = []
    verticesInRecurStack = set() # Initialize set before the first call of recur()
    for v in range(g.V): 
        if not visited[v]:
            #verticesInRecurStack = set() # Initialize set before the first call of recur()
            if not recur(v): # Return False if a cycle is detected                
                return None

    reverseList.reverse()
    return reverseList



# SP: Shortest Path
class SP:
    def __init__(self, g, s):
        if not isinstance(g, EdgeWeightedDigraph): 
            raise Exception(f"{g} is not an EdgeWeightedDigraph")

        self.g, self.s = g, s
        self.validateVertex(s)  
        self.edgeTo = [None] * g.V # edgeTo[w]: None으로 초기화 
        self.distTo = [float('inf')] * g.V  # distTo[w]: 무한대로 초기화
        self.distTo[s] = 0 # 출발지(s) 경로는 0

    def pathTo(self, v): # s->v 최단 경로 구해줌
        self.validateVertex(v)
        if not self.hasPathTo(v): 
            raise Exception(f"no path exists to vertex {v}")

        path = []
        e = self.edgeTo[v]

        while e != None: # 마지막 간선부터 차례로 path 리스트에 담고
            path.append(e)
            e = self.edgeTo[e.v]

        path.reverse() # 담긴 걸 뒤집어서 반환
        return path

    def hasPathTo(self, v): # 정점 v에 대한 최단경로가 존재하는지 -> T/F
        self.validateVertex(v)
        return self.distTo[v] < float('inf')
        
    def relax(self, e): # 기존에 알던 s->v 경로에 간선 e=v->w 더했을 때, w까지 더 짧은 경로가 나온다면, 이 경로를 edgeTo[w], distTo[w]에 저장 
        assert(isinstance(e, DirectedEdge))        
        if self.distTo[e.w] > self.distTo[e.v] +  e.weight:
            self.distTo[e.w] = self.distTo[e.v] +  e.weight
            self.edgeTo[e.w] = e

    def validateVertex(self, v):
        if v<0 or v>=self.g.V: 
            raise Exception(f"vertex {v} is not between 0 and {self.g.V-1}")

# SP - Dijkstra 
class DijkstraSP(SP): # Inherit SP class
    def __init__(self, g, s):
        super().__init__(g, s) 
        self.pq = IndexMinPQ(g.V)
        self.pq.insert(s, 0) # Indexed minPQ에 출발지 s(거리 0) 추가

        while not self.pq.isEmpty():
            dist, v = self.pq.delMin() # 매 iteration마다 distTo[] 가장 작은 정점 v 선정해서 v의 모든 outgoing 간선 relax
            for e in self.g.adj[v]:
                self.relax(e)

    def relax(self, e): # 기존에 알던 s->v 경로에 간선 e=v->w 더했을 때, w까지 더 짧은 경로가 나온다면, 이 경로를 edgeTo[w], distTo[w]에 저장 
        assert(isinstance(e, DirectedEdge))        
        if self.distTo[e.w] > self.distTo[e.v] +  e.weight:
            self.distTo[e.w] = self.distTo[e.v] +  e.weight
            self.edgeTo[e.w] = e

            if self.pq.contains(e.w): # distTo[] 변한 목적지 w가 있다면(= w가 이미 minPQ에 있다면) key 변경 
                self.pq.decreaseKey(e.w, self.distTo[e.w])

            else: self.pq.insert(e.w, self.distTo[e.w]) # w가 이미 minPQ에 없다면 key 추가 

# SP - Acyclic
class AcyclicSP(SP):
    def __init__(self, g, s):
        super().__init__(g, s) 
        tpOrder = topologicalSortWithCycleDetection(g) # topological order 구하기
        assert(tpOrder != None) # 사이클이 없음을 확인 

        for v in tpOrder: # topological order 순으로 v 선정하여, v의 모든 outgoing 간선 relax
           for e in self.g.adj[v]:
                self.relax(e) 
 
# SP - BellmanFord
class BellmanFordSP(SP):
    def __init__(self, g, s):
        super().__init__(g, s)
        self.q = Queue(maxsize=g.V)
        self.onQ = [False] * g.V
        self.q.put(s)        
        self.onQ[s] = True

        while self.q.qsize() > 0:        
            v = self.q.get()
            self.onQ[v] = False

            for e in self.g.adj[v]:
                self.relax(e)

    def relax(self, e):  # 기존에 알던 s->v 경로에 간선 e=v->w 더했을 때, w까지 더 짧은 경로가 나온다면, 이 경로를 edgeTo[w], distTo[w]에 저장 
        assert(isinstance(e, DirectedEdge))        

        if self.distTo[e.w] > self.distTo[e.v] +  e.weight:
            self.distTo[e.w] = self.distTo[e.v] +  e.weight
            self.edgeTo[e.w] = e

            if not self.onQ[e.w]:
                self.q.put(e.w)
                self.onQ[e.w] = True


if __name__ == "__main__":
    e1 = DirectedEdge(2,3,0.1)
    e1a = DirectedEdge(2,3,0.1)
    e2 = DirectedEdge(3,4,0.9)
    e3 = DirectedEdge(7,3,0.2)
    print("e1, e1a, e2, e3", e1, e1a, e2, e3)
    print("e1 == e1a", e1 == e1a)
    print("e1 < e2", e1 < e2)
    print("e1 > e3", e1 > e3)
    print("e2 < e3", e2 < e3)

    g1 = EdgeWeightedDigraph(8)
    g1.addEdge(4,5,0.35)
    g1.addEdge(5,4,0.35)
    g1.addEdge(4,7,0.37)
    g1.addEdge(5,7,0.28)
    g1.addEdge(7,5,0.28)
    g1.addEdge(5,1,0.32)
    g1.addEdge(0,4,0.38)
    g1.addEdge(0,2,0.26)
    g1.addEdge(7,3,0.39)
    g1.addEdge(1,3,0.29)
    g1.addEdge(2,7,0.34)
    g1.addEdge(6,2,0.40)
    g1.addEdge(3,6,0.52)
    g1.addEdge(6,0,0.58)
    g1.addEdge(6,4,0.93)
    print(g1)
    print(g1.adj[0])       
    print(g1.adj[7])
    print()

    g8i = EdgeWeightedDigraph.fromFile("wdigraph8i.txt")
    print("g8i", g8i)
    print("dijkstraSP on g8i")
    sp8i = DijkstraSP(g8i, 0)
    for i in range(g8i.V):
        if sp8i.hasPathTo(i): print(i, sp8i.distTo[i], sp8i.pathTo(i))
        else: print(i, "no path exists")
    print()

    g8a = EdgeWeightedDigraph.fromFile("wdigraph8a.txt")    
    print("g8a", g8a)
    print("dijkstraSP on g8a")
    sp8a = DijkstraSP(g8a, 0)
    print(sp8a.distTo)
    print(sp8a.edgeTo)
    for i in range(g8a.V):
        if sp8a.hasPathTo(i): print(i, sp8a.distTo[i], sp8a.pathTo(i))
        else: print(i, "no path exists")
    print()

    print("BellmanFordSP on g8a")
    sp8a = BellmanFordSP(g8a, 0)
    for i in range(g8a.V):
        if sp8a.hasPathTo(i): print(i, sp8a.distTo[i], sp8a.pathTo(i))
        else: print(i, "no path exists")
    print()

    print("acyclicSP on g8a")
    sp8a = AcyclicSP(g8a, 4)
    print(sp8a.distTo)
    print(sp8a.edgeTo)
    for i in range(g8a.V):
        if sp8a.hasPathTo(i): print(i, sp8a.distTo[i], sp8a.pathTo(i))
        else: print(i, "no path exists")
    print()

    g8bn = EdgeWeightedDigraph.fromFile("wdigraph8b.txt").negate()
    print("acyclicSP on -g8b for vertex 5 as the source to find longest paths")
    sp8bn = AcyclicSP(g8bn, 5)
    for i in range(g8bn.V):
        if sp8bn.hasPathTo(i): print(i, -sp8bn.distTo[i], sp8bn.pathTo(i))
        else: print(i, "no path exists")    
    print()
    