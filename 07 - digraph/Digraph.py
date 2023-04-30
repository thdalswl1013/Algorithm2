from pathlib import Path
from queue import Queue
import math # To use infinity in sap()
import timeit
 
class Digraph:
    def __init__(self, V): # 객체의 생성자 함수 정의 
        self.V = V # vertex(정점)
        self.E = 0 # edge(간선)
        self.adj = [[] for _ in range(V)]  # adj 리스트: 시작할 때는 간선이 0개이기 때문에, 빈 리스트

    def addEdge(self, v, w): # 정점v와 정점w를 연결하는 간선 추가
        if v<0 or v>=self.V: 
            raise Exception(f"Vertex id {v} is not within the range [{0}-{(self.V-1)}]")
        if w<0 or w>=self.V: 
            raise Exception(f"Vertex id {w} is not within the range [{0}-{(self.V-1)}]")
        
        self.adj[v].append(w) # v -> w 간선 존재
        self.E += 1  # edge(간선) 개수 한 개 추가

    def outDegree(self, v): # 정점 v의 outdegree(정점 v에 연결된, 나가는 간선 수) 반환
        return len(self.adj[v])

    def __str__(self): # 그래프 객체의 정보를 문자열 형태로 반환
        rtList = [f"{self.V} vertices and {self.E} edges\n"]
        for v in range(self.V):
            for w in self.adj[v]:
                rtList.append(f"{v}->{w}\n")
        return "".join(rtList)

    def reverse(self): # 모든 간선을 반대 방향으로 뒤집은 Reverse Graph 생성 
        g = Digraph(self.V)
        for v in range(self.V): # G 그래프에 v-> w 가 있다면
            for w in self.adj[v]: g.addEdge(w, v) # g 그래프에 w -> v가 있다
        return g
 
    #Create a Digraph instance from a file
    @staticmethod
    def digraphFromFile(fileName):
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
                        vw = line.split()
                        if len(vw) != 2: raise Exception(f"Invalid edge format found in {line}")
                        g.addEdge(int(vw[0]), int(vw[1]))                        
                line = f.readline().strip()
        return g
  
# sap() 함수 속도 평가 함수
def BFSforEvaluation(g):
    def bfs(s):
        queue = Queue()
        queue.put(s)        
        visited[s] = True
        distance[s] = 0
        while queue.qsize() > 0:         
            v = queue.get()
            for w in g.adj[v]:
                if not visited[w]:
                    queue.put(w)
                    visited[w] = True
                    fromVertex[w] = v
                    distance[w] = distance[v] + 1

    visited = [False for _ in range(g.V)]
    fromVertex = [None for _ in range(g.V)]
    distance = [None for _ in range(g.V)]
    for v in range(g.V):
        if not visited[v]: bfs(v) 

def topologicalSort(g): # Digraph g의 정점을 topological order 순으로 나열한 목록을 반환하는 함수
    def recur(v): # DFS로 v에 방문하는 함수
        visited[v] = True        

        for w in g.adj[v]:            
            if not visited[w]: 
                recur(w)
        reverseList.append(v) # v로부터 더는 방문할 곳 없을 때 v를 목록에 추가               

    assert(isinstance(g, Digraph)) # 함수를 호출하면 이 라인부터 시작, g가 Digraph 객체임을 확인

    visited = [False for _ in range(g.V)]
    reverseList = []

    for v in range(g.V): # 아직 방문하지 않은 정점 v를 시작점으로 DFS 수행
        if not visited[v]: 
            recur(v)

    reverseList.reverse() # 왜 뒤집어서 쓰는지도 공부했음 - p.50
    return reverseList

'''
Perform the topological sort on a DAG g, while detecing any cycle
    If a cycle is found, return False
    Else, return list of vertices in reverse DFS postorder
'''
def topologicalSortWithCycleDetection(g):
    def recur(v):        
        visited[v] = True
        verticesInRecurStack.add(v)
        for w in g.adj[v]:
            if w in verticesInRecurStack: # Edge found to a vertex in the recursive stack
                print("cycle detected on vertex", w)                
                return True
            if not visited[w]: 
                if recur(w):
                    return True
        reverseList.append(v) # Add v to the stack if all adjacent vertices were visited
        verticesInRecurStack.remove(v)
        return False

    assert(isinstance(g, Digraph))
    visited = [False for _ in range(g.V)]
    reverseList = []
    verticesInRecurStack = set() # Initialize set before the first call of recur()

    for v in range(g.V): 
        if not visited[v]:
            if recur(v): # Return None if a cycle is detected                
                return None

    reverseList.reverse()
    return reverseList

def cycleDetection(g): # 사이클 탐지 함수 -> 사이클이 있는지 없는지 판단
    def recur(v):
        visited[v] = True # 정점 v를 방문했음 
        verticesInRecurStack.add(v) # v를 집합 verticesInRecurStack에 추가 -> DFS(v)가 호출되었지만, 아직 반환하지는 않았음

        for w in g.adj[v]:
            if w in verticesInRecurStack: # v에 인접한 각 정점 w에 대해, w가 집합 verticesInRecurStack에 있다면 사이클이 있다
                return True # 사이클 존재

            if not visited[w]: 
                if recur(w): 
                    return True # 사이클 존재

        verticesInRecurStack.remove(v) # 갈 수 있는 모든 정점에 대한 방문이 끝나, 함수를 반환할 때임 -> v를 verticesInRecurStack에서 삭제
        return False

    assert(isinstance(g, Digraph))

    visited = [False for _ in range(g.V)] # 각 정점의 방문 여부 기록 리스트
    verticesInRecurStack = set() # 지금까지 거쳐온 정점 중 아직 DFS()가 반환하지 않은 정점 저장 집합

    for v in range(g.V): # 아직 방문하지 않은 정점 v를 시작점으로 해서 DFS를 수행하도록 recur() 함수 호출     
        if not visited[v]:        
            if recur(v): 
                return True # 사이클 존재
            
    return False

def sap(g, aList, bList): # g 탐색하여 alist 속한 정점과 blist 속한 정점 간 가장 가까운 조상 하나와 SCA까지의 거리 반환
        
    # 중복 체트
    LL = aList + bList
    sLL = set(LL)
    if len(LL) != len(sLL): # aList와 bList에 모두 속하는 정점 v가 있다면
        for v in LL:
            if LL.count(v) > 1:
                return (v,0) # 바로 (v, 0) 반환 
 
    # 중복이 없다면
    Q = Queue()
    visited = []

    for v in aList: # Q에 aList와 bList의 모든 정점 추가
        Q.put((v, 'a', 0)) # 거쳐온 거리=0으로 추가
        visited.append((v,'a',0))
    for v in bList:
        Q.put((v, 'b', 0))
        visited.append((v,'b',0))
    
    sapVertex = 0
    sapLength = math.inf

    while not Q.empty(): # Q가 비어있지 않다면
        v = Q.get()
        if v[2] >= sapLength: # v까지 거쳐온 거리 > sapLength라면 탐색 중단
            break

        for w in g.adj[v[0]]: # v->w 간선이 있는 각 정점 w에 대해
            temp_visited = [(x[0],x[1]) for x in visited] # w 방문 여부 표기하고 sapLength 업데이트 

            if v[1] == 'a': # v에 aList로부터 왔다면
                if (w,'a') not in temp_visited: # w에 aList로부터 도달한 적이 없다면
                    visited.append((w,'a',v[2]+1)) # w를 aList로부터 방문한 것으로 표기하고
                    Q.put((w, 'a', v[2]+1)) # Q.put(W)

                for i, vertex in enumerate(temp_visited):
                    if (w,'b') == vertex: # w에 bList로부터 방문했었다면
                        if sapLength > v[2] + 1 + visited[i][2]: # SAP 길이를 구해서, sapLength보다 작다면 업데이트
                            sapLength = v[2] + 1 + visited[i][2]
                            sapVertex = w

            elif v[1] == 'b': # v에 bList로부터 왔다면
                if (w,'b') not in temp_visited: # w에 bList로부터 도달한 적이 없다면
                    visited.append((w,'b', v[2]+1)) # w를 bList로부터 방문한 것으로 표기하고
                    Q.put((w, 'b', v[2]+1))  # Q.put(W)

                for i, vertex in enumerate(temp_visited):
                    if (w,'a') == vertex: # w에 bList로부터 방문했었다면
                        if sapLength > v[2] + 1 + visited[i][2]: # SAP 길이를 구해서, sapLength보다 작다면 업데이트
                            sapLength = v[2] + 1 + visited[i][2]
                            sapVertex = w

    return (sapVertex, sapLength)
 
class WordNet:
    def __init__(self, synsetFileName, hypernymFileName): # 생성자
        self.synsets = [] # 정점번호: 유사어 집합 관계를 저장하는 리스트
        self.nounToIndex = {} # 단어: 정점 번호 리스트 관계를 저장하는 테이블 

        # 정점(vertices) 만들기        
        synsetFilePath = Path(__file__).with_name(synsetFileName) # Use the location of the current .py file        
        with synsetFilePath.open('r') as f:            
            line = f.readline().strip() # Read a line, while removing preceding and trailing whitespaces
            while line:
                if len(line) > 0:
                    tokens = line.split(',')
                    self.synsets.append(tokens[1])                    
                    for word in tokens[1].split():
                        if word not in self.nounToIndex: self.nounToIndex[word] = []
                        self.nounToIndex[word].append(int(tokens[0]))
                line = f.readline().strip()
        self.g = Digraph(len(self.synsets))        

        # 간선(edges) 만들기
        hypernymFilePath = Path(__file__).with_name(hypernymFileName) # Use the location of the current .py file 
        with hypernymFilePath.open('r') as f:
            line = f.readline().strip() # Read a line, while removing preceding and trailing whitespaces
            while line:
                if len(line) > 0:
                    tokens = line.split(',')
                    v = int(tokens[0])
                    for idx in range(1,len(tokens)):
                        self.g.addEdge(v, int(tokens[idx]))                    
                line = f.readline().strip()
         
        # 그래프가 rooted된 DAG인지 확인 
        numVerticesWithZeroOutdegree = 0
        for v in range(self.g.V):
            if self.g.outDegree(v) == 0: 
                numVerticesWithZeroOutdegree += 1
                #print("vertex with 0 outdegree", self.synsets[v])
        if numVerticesWithZeroOutdegree != 1: raise Exception(f"The graph has {numVerticesWithZeroOutdegree} vertices with outdegree=0")

        if cycleDetection(self.g): raise Exception("The graph contains a cycle")

    def nouns(self): # Return all WordNet nouns 
        return self.nounToIndex.keys()

    def isNoun(self, word): # Is word a WordNet noun?
        return word in self.nounToIndex

    # Return the shortest common ancestor of nounA and nounB and the distance in a shortest ancestral path
    def sap(self, nounA, nounB):
        if nounA not in self.nounToIndex: 
            raise Exception(f"{nounA} not in WordNet")
        if nounB not in self.nounToIndex: 
            raise Exception(f"{nounB} not in WordNet")
        sca, distance = sap(self.g, self.nounToIndex[nounA], self.nounToIndex[nounB])
        return self.synsets[sca], distance


def outcast(wordNet, wordFileName):
    words = set()
    filePath = Path(__file__).with_name(wordFileName)  # Use the location of the current .py file   
    with filePath.open('r') as f:        
        line = f.readline().strip() # Read a line, while removing preceding and trailing whitespaces
        while line:                                
            if len(line) > 0:
                words.update(line.split())                              
            line = f.readline().strip()
    
    maxDistance = -1
    maxDistanceWord = None

    for nounA in words:
        distanceSum = 0

        for nounB in words:
            if nounA != nounB:
                _, distance = wordNet.sap(nounA, nounB)
                distanceSum += distance

        if distanceSum > maxDistance:
            maxDistance = distanceSum
            maxDistanceWord = nounA
    
    return maxDistanceWord, maxDistance, words


if __name__ == "__main__":   
    # Unit test for sap()
    print('digraph6.txt')
    d6 = Digraph.digraphFromFile('digraph6.txt')
    print(sap(d6, [1], [5]))
    if sap(d6, [1], [5]) == (0,2): print("pass")
    else: print("fail")
    print(sap(d6, [1], [1]))
    if sap(d6, [1], [1]) == (1,0): print("pass")
    else: print("fail")
    print(sap(d6, [1], [4])) # Either (0,3) or (4,3)
    tmp = sap(d6, [1], [4])
    if tmp == (0,3) or tmp == (4,3): print("pass")
    else: print("fail")
    print(sap(d6, [1], [3]))
    if sap(d6, [1], [3]) == (3,2): print("pass")
    else: print("fail")

    print('digraph12.txt')
    d12 = Digraph.digraphFromFile('digraph12.txt')
    print(sap(d12, [3], [10]))      # (1,4)
    if sap(d12, [3], [10]) == (1,4): print("pass")
    else: print("fail")
    print(sap(d12, [3], [10, 2]))   # (0,3)
    if sap(d12, [3], [10, 2]) == (0,3): print("pass")
    else: print("fail")

    print('digraph25.txt')
    d25 = Digraph.digraphFromFile('digraph25.txt')
    print(sap(d25, [13,23,24], [6,16,17]))  # (3,4)
    if sap(d25, [13,23,24], [6,16,17]) == (3,4): print("pass")
    else: print("fail")
    print(sap(d25, [13,23,24], [6,16,17,4]))  # (3,4) or (1,4)
    tmp = sap(d25, [13,23,24], [6,16,17,4])
    if tmp == (3,4) or tmp == (1,4): print("pass")
    else: print("fail")
    print(sap(d25, [13,23,24], [6,16,17,1]))  # (1,3)
    if sap(d25, [13,23,24], [6,16,17,1]) == (1,3): print("pass")
    else: print("fail")
    print(sap(d25, [13,23,24,17], [6,16,17,1]))  # (17,0)
    if sap(d25, [13,23,24,17], [6,16,17,1]) == (17,0): print("pass")
    else: print("fail")


    '''
    # Unit test with WordNet
    print('WordNet test')
    wn = WordNet("synsets.txt", "hypernyms.txt")
    print(wn.isNoun("blue"))
    print(wn.isNoun("fox"))
    print(wn.isNoun("lalala"))
    print(wn.sap("blue", "red"))
    tmp = wn.sap("blue", "red")
    if tmp != None and len(tmp) == 2 and tmp[1] == 2: print("pass")
    else: print("fail")
    print(wn.sap("blue", "fox"))
    tmp = wn.sap("blue", "fox")
    if tmp != None and len(tmp) == 2 and tmp[1] == 8: print("pass")
    else: print("fail")
    print(wn.sap("apple", "banana"))
    tmp = wn.sap("apple", "banana")
    if tmp != None and len(tmp) == 2 and tmp[1] == 2: print("pass")
    else: print("fail")
    print(wn.sap("George_W._Bush", "JFK"))
    tmp = wn.sap("George_W._Bush", "JFK")
    if tmp != None and len(tmp) == 2 and tmp[1] == 2: print("pass")
    else: print("fail")
    print(wn.sap("George_W._Bush", "Eric_Arthur_Blair"))
    tmp = wn.sap("George_W._Bush", "Eric_Arthur_Blair")
    if tmp != None and len(tmp) == 2 and tmp[1] == 7: print("pass")
    else: print("fail")
    print(wn.sap("George_W._Bush", "chimpanzee"))
    tmp = wn.sap("George_W._Bush", "chimpanzee")
    if tmp != None and len(tmp) == 2 and tmp[1] == 17: print("pass")
    else: print("fail")    
    
    print('outcast test')
    print(outcast(wn, "outcast5.txt"))
    tmp = outcast(wn, "outcast5.txt")
    if tmp != None and len(tmp) == 3 and tmp[0] == "table": print("pass")
    else: print("fail")
    print(outcast(wn, "outcast8.txt"))
    tmp = outcast(wn, "outcast8.txt")
    if tmp != None and len(tmp) == 3 and tmp[0] == "bed": print("pass")
    else: print("fail")
    print(outcast(wn, "outcast11.txt"))
    tmp = outcast(wn, "outcast11.txt")
    if tmp != None and len(tmp) == 3 and tmp[0] == "potato": print("pass")
    else: print("fail")
    print(outcast(wn, "outcast9.txt"))
    tmp = outcast(wn, "outcast9.txt")
    if tmp != None and len(tmp) == 3 and tmp[0] == "fox": print("pass")
    else: print("fail")
    '''