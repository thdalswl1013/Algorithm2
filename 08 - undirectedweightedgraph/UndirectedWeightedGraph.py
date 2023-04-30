from pathlib import Path
from queue import PriorityQueue
import timeit

# weight는 있고 방향성은 없는 간선
class Edge: 
    def __init__(self, v, w, weight): # Edge 객체의 생성자
        if v<=w: # v, w 중 더 작은 값을 self.v에 저장하고, 더 큰 값을 self.w에 저장 -> index가 증가하는 순서대로 출력하기 위함
            self.v, self.w = v, w 
        else: 
            self.v, self.w = w, v 
        self.weight = weight
    
    def __lt__(self, other): # 두 Edge 객체의 대소 비교: <
        assert(isinstance(other, Edge))
        return self.weight < other.weight

    def __gt__(self, other): # 두 Edge 객체의 대소 비교: >
        assert(isinstance(other, Edge))
        return self.weight > other.weight

    def __eq__(self, other): # 두 Edge 객체의 대소 비교: =
        assert(isinstance(other, Edge))
        return self.v == other.v and self.w == other.w and self.weight == other.weight

    def __str__(self): # Edge 객체 출력에 이용 
        return f"{self.v}-{self.w} ({self.weight})"

    def __repr__(self): # Called when an Edge instance is printed as an element of a list
        return self.__str__()

    def other(self, v): # Edge 객체의 정점 하나(v)를 입력으로 받아 다른 정점을 반환하는 함수
        if self.v == v: # v-w 간선에서 e.other(v)=w
            return self.w
        else: 
            return self.v

# 그래프의 간선을 Edge 클래스의 객체로
class WUGraph:
    def __init__(self, V): # 생성자
        self.V = V # 정점(vertex) 개수
        self.E = 0 # 간선(edge) 개수
        self.adj = [[] for _ in range(V)] # adj[]: 인접한 모든 간선 저장 리스트
        self.edges = [] # edge[]: WUgraph에 속한 모든 간선 저장 리스트

    def addEdge(self, v, w, weight): # 새로운 간선 추가
        e = Edge(v, w, weight) # v-w 연결하는 간선 (비중: weight)

        self.adj[v].append(e)
        self.adj[w].append(e)
        self.edges.append(e)
        self.E += 1

    def degree(self, v): # v의 degree(v에 연결된 간선 수) 반환
        return len(self.adj[v])

    def __str__(self): # WUGraph 객체를 출력할 때 사용하는 함수
        rtList = [f"{self.V} vertices and {self.E} edges\n"]
        for v in range(self.V):
            for e in self.adj[v]:
                if v == e.v:
                    rtList.append(f"{e}\n") # Do not print the same edge twice
        return "".join(rtList)

    @staticmethod
    def fromFile(fileName): # 파일에서 그래프 정보 읽어와서 WUGraph 객체 만들어서 반환하는 함수
        filePath = Path(__file__).with_name(fileName)   # Use the location of the current .py file   
        with filePath.open('r') as f:
            phase = 0
            line = f.readline().strip() # Read a line, while removing preceding and trailing whitespaces
            while line:                                
                if len(line) > 0:
                    if phase == 0: # Read V, the number of vertices
                        g = WUGraph(int(line))
                        phase = 1
                    elif phase == 1: # Read edges
                        edge = line.split()
                        if len(edge) != 3: raise Exception(f"Invalid edge format found in {line}")
                        g.addEdge(int(edge[0]), int(edge[1]), float(edge[2]))                        
                line = f.readline().strip()
        return g

# weighted quick union 
class UF:
    def __init__(self, V): # V: 정점(vertex) 개수
        self.ids = [] # ids[i]: i's parent
        self.size = [] # size[i]: size of tree rooted at i

        for idx in range(V):
            self.ids.append(idx)
            self.size.append(1)       

    def root(self, i):
        while i != self.ids[i]: 
            i = self.ids[i]
        return i

    def connected(self, p, q):
        return self.root(p) == self.root(q)

    def union(self, p, q):    
        id1, id2 = self.root(p), self.root(q)
        if id1 == id2: 
            return
        if self.size[id1] <= self.size[id2]: 
            self.ids[id1] = id2
            self.size[id2] += self.size[id1]
        else:
            self.ids[id2] = id1
            self.size[id1] += self.size[id2]

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


# Kruskal 알고리즘을 사용한 MST
def mstKruskal(g): 
    assert(isinstance(g, WUGraph))
    
    edgesInMST = [] # 지금까지 MST에 포함한 간선 저장
    weightSum = 0 # MST에 포함한 간선 weight의 합

    pq = PriorityQueue() # PQ(Priority Queue) 생성
    for e in g.edges: # 모든 간선을 PQ에 넣어줘
        pq.put(e)

    uf = UF(g.V)
    while not pq.empty() and len(edgesInMST) < g.V-1: # 간선 개수가 v-1개가 되면 종료
        e = pq.get()

        if not uf.connected(e.v, e.w): # 간선(edge) e가 사이클을 만들지 않는 간선이라면, MST에 e 추가
            uf.union(e.v, e.w)
            edgesInMST.append(e)
            weightSum += e.weight

    return edgesInMST, weightSum

# Prim 알고리즘(Lazy ver.)을 사용한 MST 
def mstPrimLazy(g):
    def include(v): # v를 MST에 추가  
        included[v] = True

        for e in g.adj[v]: # 인접한 간선 중 MST 외부로 향하는 간선 모두 추가
            if not included[e.other(v)]: 
                pq.put(e)

    assert(isinstance(g, WUGraph))

    edgesInMST = [] 
    included = [False] * g.V # v가 MST에 있다면, included[v] == True 
    weightSum = 0  
    
    pq = PriorityQueue() # Build a priority queue
    include(0) # 정점 0에 대해 호출하여 0에 인접한 간선이 다 PQ에 들어가게 하고나서 시작

    while not pq.empty() and len(edgesInMST) < g.V-1:
        e = pq.get() # PQ에서 꺼내서 사이클을 만드는지/안만드는지 확인

        if included[e.v] and included[e.w]: # v-w 모두 MST 상에 있는 것은 무시
            continue 

        edgesInMST.append(e)
        weightSum += e.weight 

        # v, w 중 아직 MST에 포함 안 한 정점과 간선 포함
        # 새로운 정점 포함 시, 한 정점은 외부정점이고 다른 정점은 내부 정점임(외부정점만 포함하면 됨)
        if not included[e.v]: # v가 외부 정점이면 v를 포함
            include(e.v) 
        if not included[e.w]: # w가 외부정점이면 w를 포함
            include(e.w)

    return edgesInMST, weightSum    
 
# Prim 알고리즘(Eager ver.)을 사용한 MST 
def mstPrimEager(g): # g: WUGraph
    def include(w): # w를 MST에 추가 
        included[w] = True 

        for e in g.adj[w]: # w에 인접한 각 간선 e=w-x에 대해
            # find target
            index = e.w

            if e.v != w:
                index = e.v

            if included[index] == False: 
                if pq.contains(index) == False: # 정점 x가 아직 pq에 없다면 pq.insert(x, e)
                    pq.insert(index, e)

                elif pq.keyOf(index).weight > e.weight: # 정점 x가 이미 pq에 있고, pq에 저장된 간선보다 e의 weight가 작다면 pq.decreaseKey(x, e) 
                    pq.decreaseKey(index, e) 

    included = [False] * g.V
    result = []
    
    pq = IndexMinPQ(g.V)
    include(0)
    
    while len(result) != g.V-1: # 결과 리스트에 v-2개의 간선을 포함하지 않았다면
        e, w = pq.delMin() # pq.delMin
        result.append(e) # e를 결과 리스트에 추가
        include(w)
        
    sumOfWeight = 0
    for i in result:
        sumOfWeight += i.weight
    
    return (result, sumOfWeight)


if __name__ == "__main__":
    # Unit test for Edge and WUGraph
    '''
    e1 = Edge(2,3,0.1)
    e2 = Edge(2,3,0.1)
    e3 = Edge(2,3,0.2)
    print(e1 == e1)
    print(e1 == e2)
    print(e1 == e3)
    print(e1.other(3))
    print(e1.other(2))
    
    g8 = WUGraph.fromFile("wugraph8.txt")
    print(g8)'''    

    # Unit test for the min PQ
    '''
    minPQ = IndexMinPQ(10)
    minPQ.insert(0,'P')
    print(minPQ.pq, minPQ.keys, minPQ.qp)
    minPQ.insert(1,'Q')
    print(minPQ.pq, minPQ.keys, minPQ.qp)
    minPQ.changeKey(0,'R')
    print(minPQ.pq, minPQ.keys, minPQ.qp)
    minPQ.insert(2,'E')
    minPQ.insert(3,'X')
    minPQ.insert(4,'A')
    minPQ.insert(5,'M')
    minPQ.insert(6,'P')
    minPQ.insert(7,'L')
    minPQ.insert(8,'E')
    print(minPQ.pq, minPQ.keys, minPQ.qp)
    print(minPQ.delMin())    
    print(minPQ.delMin())    
    print(minPQ.delMin())
    print(minPQ.delMin())
    print(minPQ.delMin())
    minPQ.decreaseKey(3,'B')
    print(minPQ.delMin())
    print(minPQ.delMin())
    print(minPQ.delMin())
    print(minPQ.delMin())    
    '''
    
    # Unit test for mstPrimEager()
    g8 = WUGraph.fromFile("wugraph8.txt")
    print("Kruskal on g8", mstKruskal(g8))    
    print("Prim lazy on g8", mstPrimLazy(g8))    
    print("Prim eager on g8", mstPrimEager(g8))
    edges, weightSum = mstPrimEager(g8)
    failCorrectness = False    
    if edges == [Edge(0,7,0.16), Edge(1,7,0.19), Edge(0,2,0.26), Edge(2,3,0.17), Edge(5,7,0.28), Edge(4,5,0.35), Edge(2,6,0.4)]: 
        print ("pass")
    else: 
        print ("fail")
        failCorrectness = True
    if weightSum == 1.81: 
        print ("pass")
    else: 
        print ("fail")
        failCorrectness = True
    print()
    
    if failCorrectness: 
        print("fail")
    else:
        n = 100
        tKruskal = timeit.timeit(lambda: mstKruskal(g8), number=n)/n
        tPrimLazy = timeit.timeit(lambda: mstPrimLazy(g8), number=n)/n
        tPrimEager = timeit.timeit(lambda: mstPrimEager(g8), number=n)/n
        print(f"Average running time for g8 with Kruskal ({tKruskal:.10f}), PrimLazy ({tPrimLazy:.10f}), and PrimEager({tPrimEager:.10f})")
        if tPrimEager < tKruskal and tPrimEager < tPrimLazy: 
            print ("pass")
        else: print ("fail")
    print()

    g8a = WUGraph.fromFile("wugraph8a.txt")    
    print("Kruskal on g8a", mstKruskal(g8a))    
    print("Prim lazy on g8a", mstPrimLazy(g8a))    
    print("Prim eager on g8a", mstPrimEager(g8a))
    edges, weightSum = mstPrimEager(g8a)
    failCorrectness = False
    if weightSum == 50: 
        print ("pass")
    else: 
        print ("fail")
        failCorrectness = True
    print()

    if failCorrectness: 
        print("fail")
    else:
        n = 100
        tKruskal = timeit.timeit(lambda: mstKruskal(g8a), number=n)/n
        tPrimLazy = timeit.timeit(lambda: mstPrimLazy(g8a), number=n)/n
        tPrimEager = timeit.timeit(lambda: mstPrimEager(g8a), number=n)/n
        print(f"Average running time for g8a with Kruskal ({tKruskal:.10f}), PrimLazy ({tPrimLazy:.10f}), and PrimEager({tPrimEager:.10f})")
        if tPrimEager < tKruskal and tPrimEager < tPrimLazy: 
            print ("pass")
        else: 
            print ("fail")
    print()

    
    