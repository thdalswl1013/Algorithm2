from queue import Queue

# undirected graph
class Undirected_Graph:
    def __init__(self, v):
        self.v=v #vertex(정점)
        self.e=0 #edge(간선)
        self.adj=[[] for _ in range(v)] #adjency list
    
    def addEdge(self, v, w):
        self.adj[v].append(w) # v->w 간선 존재
        self.adj[w].append(v) # w->v 간선 존재
        self.e+=1 # edge(간선) 개수 한 개 추가

    def degree(self, v):
        return len(self.adj[v])

class Directed_Graph:
    def __init__(self, v):
        self.v=v
        self.e=0
        self.adj=[[] for _ in range(v)]
    
    def addEdge(self, v, w):
        self.adj[v].append(w) # v->w 간선 존재
        self.e+=1

    def outdegree(self, v):
        return len(self.adj[v])

    def reverse(self): # 모든 간선을 반대 방향으로 뒤집은 Reverse Graph 생성 
        g=Directed_Graph(self.v)

        for v in range(self.v): # G 그래프에 v-> w 가 있다면
            for w in self.adj[v]:
                g.addEdge(w, v) # g 그래프에 w -> v가 있다

        return g

class DFS:
    def __init__(self, g, s):
        def recur(v): 
            self.visited[v] =True # 방문하고 False를 True로 바꿈

            for w in g. adj[v]: 
                if not self.visited[w]: # 방문 안한 정점을 차례로 방문
                    recur(w)
                    self.fromVertex[w]=v # 방문 직전에 거친 정점(v) 저장
        self.g = g
        self.s = s
        self.visited=[False for _ in range(g.v)] # 방문 여부 저장 리스트
        self.fromVertex=[None for _ in range(g.v)] # 방문 직전에 거친 정점 저장 리스트

        recur(s)

class SingleSource_BFS:
    def __init__(self, g, s):
        assert(isinstance(g, Directed_Graph) and s>=0 and s<g.v) # 출발점 s가 그래프 g에 속하는 정점이 맞는지 확인
        
        self.g = g
        self.s = s
        self.visited=[False for _ in range(g.v)] # 방문 여부 
        self.fromVertex=[None for _ in range(g.v)] # 방문 직전에 거친 정점  
        self.distance=[None for _ in range(g.v)] # s->w 갈 때 거치는 거리 

        queue=Queue() 
        queue.put(s) # queue에 출발점 s 추가
        self.visited[s]=True # 출발점 s의 방문여부는 true가 됨
        self.distance[s]=0 # s->s 거리는 0

        while queue.qsize()>0: # queue가 빌 때까지 반복
            v=queue.get() # q에 가장먼저 추가된 정점 v를 pop
            
            for w in g.adj[v]:
                if not self.visited[w]: # v와 인접한 정점 중 아직 방문하지 않은 모든 정점 w를 queue에 추가
                    queue.put(w)

                    self.visited[w]=True 
                    self.fromVertex[w]=v
                    self.distance[w]=self.distance[v]+1

class MultiSource_BFS:
    def __init__(self, g, sList):
        assert(isinstance(g, Directed_Graph) and s>=0 and s<g.v) # 출발점 s가 그래프 g에 속하는 정점이 맞는지 확인
        
        self.g = g
        self.s = s
        self.visited=[False for _ in range(g.v)] # 방문 여부 
        self.fromVertex=[None for _ in range(g.v)] # 방문 직전에 거친 정점  
        self.distance=[None for _ in range(g.v)] # s->w 갈 때 거치는 거리 

        queue=Queue() 
        for s in sList: # queue에 sList 안의 모든 출발점 추가
            queue.put(s) 
            self.visited[s]=True # 출발점 s의 방문여부는 true가 됨
            self.distance[s]=0 # s->s 거리는 0

        while queue.qsize()>0: # queue가 빌 때까지 반복
            v=queue.get() # q에 가장먼저 추가된 정점 v를 pop
            
            for w in g.adj[v]:
                if not self.visited[w]: # v와 인접한 정점 중 아직 방문하지 않은 모든 정점 w를 queue에 추가
                    queue.put(w)

                    self.visited[w]=True 
                    self.fromVertex[w]=v
                    self.distance[w]=self.distance[v]+1

def topologicalSort(g):
    def recur(v): # DFS로 v 방문하는 함수
        visited[v] = True 
        for w in g.adj[v]:            
            if not visited[w]: recur(w)
        reverseList.append(v) # v로부터 더는 방문할 곳이 없을 때 목록에 v를 추가

    assert(isinstance(g, Directed_Graph)) # 함수 호출 시 이 라인부터 시작

    visited = [False for _ in range(g.V)] # 정점의 방문 여부 기록 리스트
    reverseList = [] # topological order 기록 리스트

    for v in range(g.V): # 아직 방문하지 않은 정점 v를 시작점으로, DFS 수행
        if not visited[v]: 
            recur(v) # DFS

    reverseList.reverse()
    return reverseList