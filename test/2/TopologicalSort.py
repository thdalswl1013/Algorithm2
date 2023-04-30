from pathlib import Path
import random
import timeit

'''
Class for storing directed graphs
'''
class Digraph:
    def __init__(self, V): # Constructor
        self.V = V # Number of vertices
        self.E = 0 # Number of edges
        self.adj = [[] for _ in range(V)]   # adj[v] is a list of vertices pointed from v

    def addEdge(self, v, w): # Add a directed edge v->w. Self-loops and parallel edges are allowed
        self.adj[v].append(w)        
        self.E += 1

    def outDegree(self, v):
        return len(self.adj[v])

    def __str__(self):
        rtList = [f"{self.V} vertices and {self.E} edges\n"]
        for v in range(self.V):
            for w in self.adj[v]:
                rtList.append(f"{v}->{w}\n")
        return "".join(rtList)        

    def reverse(self): # return a digraph with all edges reversed
        g = Digraph(self.V)
        for v in range(self.V):
            for w in self.adj[v]: g.addEdge(w, v)
        return g

    @staticmethod
    def fromFile(fileName):
        filePath = Path(__file__).with_name(fileName)   # Use the location of the current .py file   
        with filePath.open('r') as f:
            phase = 0
            line = f.readline().strip() # Read a line, while removing preceding and trailing whitespaces
            while line:                                
                if len(line) > 0:
                    if phase == 0: # Read V, the number of vertices
                        g = Digraph(int(line))
                        phase = 1
                    elif phase == 1: # Read edges
                        edge = line.split()
                        if len(edge) != 2: raise Exception(f"Invalid edge format found in {line}")
                        g.addEdge(int(edge[0]), int(edge[1]))                        
                line = f.readline().strip()
        return g


'''
Class that finds SCC (Strongly-Connected Components) based on topological sort
    and stores the results

This class is used to evaluate the correctness of topological sort
'''
class SCC:
    def __init__(self, g): # Do strongly-connected-components pre-processing, based on Kosaraju-Sharir algorithm
        def recur(v): # DFS to mark all vertices connected to v            
            self.id[v] = self.count
            for w in g.adj[v]:
                if self.id[w] < 0: 
                    recur(w)                            
        self.g = g
        self.id = [-1 for i in range(g.V)] # id[v] is the ID of component to which v belongs (-1 for not visited)
        self.count = 0 # Number of strongly-connected components
        for v in topologicalSort(g.reverse()):
            if self.id[v] < 0:
                recur(v)
                self.count += 1        

    def connected(self, v, w): # Are v and w connected?
        return self.id[v] == self.id[w]


# This function is used to evaluate the speed of topological sort
# 
def DFSforEvaluation(g):
    def recur(v):        
            visited[v] = True            
            for w in g.adj[v]:
                if not visited[w]: 
                    recur(w)
                    fromVertex[w] = v
    assert(isinstance(g, Digraph))    
    visited = [False for _ in range(g.V)]
    fromVertex = [None for _ in range(g.V)]
    for v in range(g.V):
        if not visited[v]:
            recur(v)        
    
    return visited, fromVertex


'''
첨부된 TopologicalSort.py 파일에서 topologicalSort(g) 함수가 "DFS 알고리즘에 기반"해 topological order를 찾도록 구현해 제출하시오.

- 입력 g: Digraph 객체
- 반환 값: topological order에 따라 정점 번호를 저장한 리스트
- 사이클이 있는 그래프도 입력으로 들어올 수 있으며, 이럴 때 topological sort 결과는 SCC(Strongly-Connected Component) 탐지에 사용됨
'''
# DFS 사용 -> recur
# 결과.reverse()

# 2020112099 송민지 
def topologicalSort(g):
    visitList = [False for _ in range(g.V)]
    topologicalorderList = []

    def recur(v): # DFS 사용 -> v 방문 
        visitList[v] = True 

        for w in g.adj[v]:    
            if not visitList[w]:
                recur(w)
        topologicalorderList.append(v) # 더 방문할 곳이 없을 때 v 추가

    for v in range(g.V):
        if not visitList[v]:
            recur(v)

    topologicalorderList.reverse()
    return topologicalorderList

if __name__ == "__main__":
    # Unit test for topological Sort
    correct = True

    print("Correctness test with digraph4.txt")
    g4 = Digraph.fromFile("digraph4.txt")    
    g4_result = topologicalSort(g4)
    print(g4_result)
    if len(g4_result)==4 and (g4_result==[3,0,1,2] or g4_result==[0,3,1,2] or g4_result==[0,1,3,2]): print("pass")
    else:
        print("fail")
        correct = False
    print()

    print("Correctness test with digraph5.txt")
    g5 = Digraph.fromFile("digraph5.txt")    
    g5_result = topologicalSort(g5)
    print(g5_result)
    if len(g5_result)==5 and g5_result == [0,1,2,3,4]: print("pass")
    else:
        print("fail")
        correct = False
    print()

    print("Correctness test with digraph7.txt")
    g7 = Digraph.fromFile("digraph7.txt")
    g7_result = topologicalSort(g7)
    print(g7_result)
    if len(g7_result)==7 and g7_result[0]==3 and g7_result[1]==6 and g7_result[2]==0: print("pass")
    else:
        print("fail")
        correct = False
    print()

    print("Correctness test with digraph6.txt and Kosaraju-Sharir algorithm")
    g6 = Digraph.fromFile("digraph6.txt")
    scc6 = SCC(g6)
    if scc6.count == 2 and scc6.connected(1,4) and not scc6.connected(2,5) and scc6.connected(3,5): print("pass")
    else:
        print("fail")
        correct = False
    print()

    print("Correctness test with digraph7a.txt and Kosaraju-Sharir algorithm")
    g7a = Digraph.fromFile("digraph7a.txt")
    scc7a = SCC(g7a)    
    if scc7a.count == 6 and scc7a.connected(5,6) and not scc7a.connected(0,1) and not scc7a.connected(2,6) and not scc7a.connected(3,4): print("pass")
    else:
        print("fail")
        correct = False
    print()

    print("Correctness test with digraph13.txt and Kosaraju-Sharir algorithm")
    g13 = Digraph.fromFile("digraph13.txt")
    scc13 = SCC(g13)        
    if scc13.count == 5 and scc13.connected(0,4) and not scc13.connected(0,6) and not scc13.connected(6,9) and scc13.connected(9,12) and not scc13.connected(7,11): print("pass")
    else:
        print("fail")
        correct = False
    print()

    print("Speed test with a random graph")
    if not correct: print("fail (since the algorithm is not correct)")
    else:
        v, e = 1000, 2000
        gr = Digraph(v)
        for _ in range(e): gr.addEdge(random.randint(0,v-1), random.randint(0,v-1))
        n=100
        tTS = timeit.timeit(lambda: topologicalSort(gr), number=n)/n    
        tDFS = timeit.timeit(lambda: DFSforEvaluation(gr), number=n)/n
        print(f"{n} calls of topologicalSort() took {tTS:.10f} sec on average, and the same number of calls of DFS() took {tDFS:.10f} sec on average")
        if tTS < tDFS * 1.45: print("pass")
        else: print("fail")
        if tTS < tDFS * 1.3: print("pass")
        else: print("fail")
    print()   

    
