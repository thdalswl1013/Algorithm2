from pathlib import Path
from queue import Queue
import timeit

class FlowEdge:
    def __init__(self, v, w, capacity): # Create an edge v->w with a double capacity
        assert isinstance(v, int) and isinstance(w, int), f"v({v}) and/or w({w}) are not integers"
        assert v>=0 and w>=0, f"Vertices {v} and/or {w} have negative IDs"
        assert isinstance(capacity, int) or isinstance(capacity, float), f"Capacity {capacity} is neither an integer nor a floating-point number"
        assert capacity>=0, f"Edge capacity {capacity} must be >= 0"
        self.v, self.w, self.capacity = v, w, capacity
        self.flow = 0.0 # Initialize flow to 0
    
    def __lt__(self, other): # < operator, used to sort elements (e.g., in a PriorityQueue, sorted() function)
        self.validateInstance(other)
        return self.capacity < other.capacity

    def __gt__(self, other): # > operator, used to sort elements
        self.validateInstance(other)
        return self.capacity > other.capacity

    def __eq__(self, other): # == operator, used to compare edges for grading
        if other == None: return False
        self.validateInstance(other)
        return self.v == other.v and self.w == other.w and self.capacity == other.capacity

    def __str__(self): # Called when an Edge instance is printed (e.g., print(e))
        return f"{self.v}->{self.w} ({self.flow}/{self.capacity})"

    def __repr__(self): # Called when an Edge instance is printed as an element of a list
        return self.__str__()

    def other(self, vertex): # Return the vertex on the Edge other than vertex        
        if vertex == self.v: return self.w
        elif vertex == self.w: return self.v
        else: self.invalidIndex(vertex)

    def remainingCapacityTo(self, vertex): # Remaining capacity toward vertex
        if vertex == self.v: return self.flow
        elif vertex == self.w: return self.capacity - self.flow
        else: self.invalidIndex(vertex)
    
    def addRemainingFlowTo(self, vertex, delta): # Add delta flow toward vertex     
        assert isinstance(delta, int) or isinstance(delta, float), f"Delta {delta} is neither an integer nor a floating-point number"
        assert delta <= self.remainingCapacityTo(vertex), f"Delta {delta} is greater than the remaining capacity {self.remainingCapacity }"        
        if vertex == self.v: self.flow -= delta
        elif vertex == self.w: self.flow += delta
        else: self.invalidIndex(vertex)

    def invalidIndex(self, i):
        assert False, f"Illegal endpoint {i}, which is neither of the two end points {self.v} and {self.w}"

    @staticmethod
    def validateInstance(e):
        assert isinstance(e, FlowEdge), f"e={e} is not an instance of FlowEdge"


'''
Class that represents Digraphs with Flow/Capacity
'''
class FlowNetwork:
    def __init__(self, V): # Constructor
        assert isinstance(V, int) and V >= 0, f"V({V}) is not an integer >= 0"
        self.V = V # Number of vertices
        self.E = 0 # Number of edges
        self.adj = [[] for _ in range(V)]   # adj[v] is a list of vertices adjacent to v
        self.edges = []

    def addEdge(self, e): # Add edge v-w. Self-loops and parallel edges are allowed
        FlowEdge.validateInstance(e)
        assert 0<=e.v and e.v<self.V and 0<=e.w and e.w<self.V, f"Edge endpoints ({e.v},{e.w}) are out of the range 0 to {self.V-1}"
        self.adj[e.v].append(e) # Add forward edge
        self.adj[e.w].append(e) # Add backward edge
        self.edges.append(e)
        self.E += 1

    def __str__(self):
        rtList = [f"{self.V} vertices and {self.E} edges\n"]
        for v in range(self.V):
            for e in self.adj[v]: 
                if e.v == v: rtList.append(f"{e}\n") # Show only forward edges to not show the same edge twice
        return "".join(rtList)

    def copy(self):
        fn = FlowNetwork(self.V)
        for e in self.edges: fn.addEdge(FlowEdge(e.v, e.w, e.capacity))
        return fn
 
    @staticmethod
    def fromFile(fileName):
        filePath = Path(__file__).with_name(fileName)   # Use the location of the current .py file   
        with filePath.open('r') as f:
            phase = 0
            line = f.readline().strip() # Read a line, while removing preceding and trailing whitespaces
            while line:                                
                if len(line) > 0:
                    if phase == 0: # Read V, the number of vertices
                        g = FlowNetwork(int(line))
                        phase = 1
                    elif phase == 1: # Read edges
                        edge = line.split()
                        assert len(edge) == 3, f"Invalid edge format found in {line}"
                        if edge[2] == 'inf': g.addEdge(FlowEdge(int(edge[0]), int(edge[1]), float('inf')))                        
                        else: g.addEdge(FlowEdge(int(edge[0]), int(edge[1]), float(edge[2])))                        
                line = f.readline().strip()
        return g

    @staticmethod
    def validateInstance(g):
        assert isinstance(g, FlowNetwork), f"g={g} is not an instance of FlowNetwork"


def findAugmentingPathBFS(g, s):
    FlowNetwork.validateInstance(g)
    edgeTo = [None for _ in range(g.V)]
    visited = [False for _ in range(g.V)]
        
    q = Queue()
    q.put(s)
    visited[s] = True
    while not q.empty():
        v = q.get()
        for e in g.adj[v]:
            w = e.other(v)
            if e.remainingCapacityTo(w) > 0 and not visited[w]:
                edgeTo[w] = e
                visited[w] = True
                q.put(w)

    return edgeTo, visited


'''
Class that performs FordFulkerson algorithm to identify maxflow and mincut
    and that stores the results
'''
class FordFulkerson:
    def __init__(self, g, s, t):        
        FlowNetwork.validateInstance(g)
        assert s>=0 and s<g.V, f"s({s}) is not within 0 ~ {g.V-1}"
        assert t>=0 and t<g.V, f"t({t}) is not within 0 ~ {g.V-1}"
        assert s != t, f"s({s}) and t({t}) must be different"

        self.g = g.copy() # Make a copy to not mutate original graph
        self.s, self.t = s, t

        self.flow = 0.0
        while self.hasAugmentingPath():
            # Compute bottleneck capacity along the augmenting path
            minflow = float('inf')
            v = t
            while v != s: 
                minflow = min(minflow, self.edgeTo[v].remainingCapacityTo(v))
                v = self.edgeTo[v].other(v)
            
            # Add bottlenack capacity to the augmenting path
            v = t
            while v != s:
                self.edgeTo[v].addRemainingFlowTo(v, minflow)
                v = self.edgeTo[v].other(v)
        
            # Increase the amoung of flow
            self.flow += minflow

    # Perform BFS to find vertices reachable from s along with shortest paths to them
    def hasAugmentingPath(self):
        self.edgeTo = [None for _ in range(self.g.V)]
        self.visited = [False for _ in range(self.g.V)]
        
        q = Queue()
        q.put(self.s)
        self.visited[self.s] = True
        while not q.empty():
            v = q.get()
            for e in self.g.adj[v]:
                w = e.other(v)
                if e.remainingCapacityTo(w) > 0 and not self.visited[w]:
                    self.edgeTo[w] = e
                    self.visited[w] = True
                    q.put(w)

        return self.visited[self.t] # Is t reachable from s with current flow assignment?

    def inCut(self, vertex): # Is vertex reachable from s with current flow assignment?
        assert vertex>=0 and vertex<self.g.V, f"vertex({vertex}) is not within 0 ~ {self.g.V-1}"
        return self.visited[vertex]


class BaseballElimination:
    # Read from fileName the scores for each team and store them in member variables
    def __init__(self, fileName):
        filePath = Path(__file__).with_name(fileName)   # Use the location of the current .py file   
        with filePath.open('r') as f:
            self.teams = [] # teams[i]: name for team i
            self.team2id = {} # symbol table for team name -> index            
            self.wins = [] # Number of wins/losses/remaining games for team i
            self.losses = []
            self.remaining = [] 
            self.against = [] # against[i][j]: Number of remaining game between teams i and j            

            teamId = -1
            line = f.readline().strip() # Read a line, while removing preceding and trailing whitespaces
            while line:                                
                if len(line) > 0:
                    if teamId == -1: # Read the number of teams
                        tokens = line.split()
                        assert len(tokens) == 1, f"First non-empty line must contain a single token, but it does not ({line})"
                        assert tokens[0].isnumeric(), f"First non-empty line must contain an integer, but it does not ({tokens[0]})"
                        self.numberOfTeams = int(tokens[0])
                        teamId = 0                        
                    elif teamId >= 0: # Read team name and scores
                        tokens = line.split()
                        assert len(tokens) == 4 + self.numberOfTeams, f"Invalid team format found in {line}"
                        self.teams.append(tokens[0])
                        self.team2id[tokens[0]] = teamId
                        self.wins.append(int(tokens[1]))
                        self.losses.append(int(tokens[2]))
                        self.remaining.append(int(tokens[3]))
                        self.against.append([])
                        for i in range(4, 4 + self.numberOfTeams): self.against[teamId].append(int(tokens[i]))                        
                        teamId += 1
                line = f.readline().strip()

    def __str__(self):
        rtList = [f"{self.numberOfTeams} teams\n"]        
        for teamId in range(self.numberOfTeams):            
            rtList.append(f"{self.teams[teamId]}({self.team2id[self.teams[teamId]]}) {self.wins[teamId]} {self.losses[teamId]} {self.remaining[teamId]}")
            for i in range(self.numberOfTeams):
                rtList.append(f" {self.against[teamId][i]}")
            rtList.append("\n")        
        return "".join(rtList)

    def printResult(self):
        for team in self.teams:
            result = self.isEliminated(team)
            assert result != None and len(result)==2, f"isEliminated() must return a 2-tuple"
            eliminate, teamsResponsible = result
            if eliminate: print(f"{team} is eliminated by the subset {teamsResponsible}")
            else: print(f"{team} is not eliminated")
        print()

    # Find out whether teamName must be eliminated
    # Return (True, a list of team names responsible for the elimination), if teamName must be eliminated
    # Return (False, []), if teamName is NOT eliminated yet
    def isEliminated(self, teamName):
        
        result = []
        index_teamName = self.team2id[teamName]
        maxWinForteamName = self.wins[index_teamName] + self.remaining[index_teamName]
        
        # case 1 
        for team in self.teams:
            if team != teamName and maxWinForteamName < self.wins[self.team2id[team]]:
                result.append(team)
        if len(result) > 0: return (True, result)
        
        # case 2
        N_teams= len(self.teams)
        N_gameV = int((N_teams - 1) * (N_teams - 2)/2)
        N_vertices = 2 + N_teams + N_gameV
        gameGraph = FlowNetwork(N_vertices)
        
        # make team list without teamName
        teamlist = []           # (team, index)
        for t in self.teams:
            if t != teamName:
                teamlist.append((t, self.team2id[t]))
            else:
                teamlist.append((None, None))
        
        # make game V : 1 ~ N_gameV
        gameInfo = []
        temp = 1
        for i in range(N_teams):
            for j in range(N_teams):
                if i < j and i != index_teamName and j != index_teamName:
                    gameGraph.addEdge(FlowEdge(0, temp, self.against[i][j]))
                    gameInfo.append((i,j))
                    temp += 1

        
        # make team V : N_gameV+1 ~ N_gameV+1 + N_teams
        for i in range(N_gameV):
            gameGraph.addEdge(FlowEdge(1 + i, N_gameV + 1 + gameInfo[i][0], float('inf')))
            gameGraph.addEdge(FlowEdge(1 + i, N_gameV + 1 + gameInfo[i][1], float('inf')))
        
        for i in range(N_teams):
            if i != index_teamName:
                gameGraph.addEdge(FlowEdge(N_gameV + 1 + i, N_vertices - 1, maxWinForteamName - self.wins[teamlist[i][1]]))

        
        # FordFulkerson Algorithm
        ford_result = FordFulkerson(gameGraph, 0, N_vertices - 1)
        for i in range(N_teams):
            if ford_result.inCut(N_gameV + 1 + i):
                result.append(teamlist[i][0])
    
        if len(result) > 0 : return (True, result)
        else: return (False, [])
            


if __name__ == "__main__":
    '''# Unit test for FlowEdge
    e1 = FlowEdge(2,3,10)
    e1a = FlowEdge(2,3,10)
    e2 = FlowEdge(3,4,90)
    e3 = FlowEdge(7,3,20)
    print("e1, e1a, e2, e3", e1, e1a, e2, e3)
    print("e1 == e1a", e1 == e1a)
    print("e1 < e2", e1 < e2)
    print("e1 > e3", e1 > e3)
    print("e2 < e3", e2 < e3)
    
    print(f"{e2}'s remaining capacity to 4: {e2.remainingCapacityTo(4)}")
    print(f"{e2}'s remaining capacity to 3: {e2.remainingCapacityTo(3)}")
    e2.addRemainingFlowTo(4, 30)
    print("flow 30 added toward 4 (forward direction)")
    print(f"{e2}'s remaining capacity to 4: {e2.remainingCapacityTo(4)}")
    print(f"{e2}'s remaining capacity to 3: {e2.remainingCapacityTo(3)}")
    e2.addRemainingFlowTo(3, 20)
    print("flow 20 added toward 3 (backward direction)")
    print(f"{e2}'s remaining capacity to 4: {e2.remainingCapacityTo(4)}")
    print(f"{e2}'s remaining capacity to 3: {e2.remainingCapacityTo(3)}")
    print()

    e1.addRemainingFlowTo(3, 5)
    e1.addRemainingFlowTo(2, 4)
    print(e1.remainingCapacityTo(3), e1.remainingCapacityTo(2))'''

    '''# Unit test for FlowNetwork
    g8m = FlowNetwork(8)
    g8m.addEdge(FlowEdge(0,1,6))
    g8m.addEdge(FlowEdge(0,2,1))
    g8m.addEdge(FlowEdge(0,3,0))
    g8m.addEdge(FlowEdge(1,4,float('inf')))
    g8m.addEdge(FlowEdge(1,5,float('inf')))
    g8m.addEdge(FlowEdge(2,4,float('inf')))
    g8m.addEdge(FlowEdge(2,6,float('inf')))
    g8m.addEdge(FlowEdge(3,5,float('inf')))
    g8m.addEdge(FlowEdge(3,6,float('inf')))
    g8m.addEdge(FlowEdge(4,7,0))
    g8m.addEdge(FlowEdge(5,7,5))
    g8m.addEdge(FlowEdge(6,7,6))
    print("g8m", g8m)    
    for i in range(g8m.V):
        print(f"g8m.adj[{i}]: {g8m.adj[i]}")
    print()

    g4 = FlowNetwork.fromFile("flownet4.txt")    
    print("g4", g4)
    print("g4.adj[0]", g4.adj[0])
    print("g4.adj[1]", g4.adj[1])
    print("g4.adj[3]", g4.adj[3])
    print()

    print("g4.copy()", g4.copy())
    print()

    g6v = FlowNetwork.fromFile("flownet6v.txt")    
    print("g6v", g6v)
    print("g6v.adj[0]", g6v.adj[0])
    print("g6v.adj[3]", g6v.adj[3])
    print("g6v.adj[5]", g6v.adj[5])
    print()

    g6h = FlowNetwork.fromFile("flownet6h.txt")    
    print("g6h", g6h)
    print("g6h.adj[0]", g6h.adj[0])
    print("g6h.adj[3]", g6h.adj[3])
    print("g6h.adj[5]", g6h.adj[5])
    print()

    g8 = FlowNetwork.fromFile("flownet8.txt")    
    print("g8", g8)
    print("g8.adj[0]", g8.adj[0])
    print("g8.adj[2]", g8.adj[2])
    print("g8.adj[7]", g8.adj[7])
    print()'''

    '''# Unit test for findAugmentingPathBFS
    g8 = FlowNetwork.fromFile("flownet8.txt")
    print(findAugmentingPathBFS(g8.copy(), 0))
    ff8 = FordFulkerson(g8, 0, g8.V-1)
    print("ff8.flow", ff8.flow) # 28.0
    print("ff8.inCut:", end=' ') # 0 2 3 6
    for v in range(g8.V):
        if ff8.inCut(v): print(v, end=' ')
    print()
    print("ff8.g", ff8.g)
    print()
    print(findAugmentingPathBFS(ff8.g, 0))'''

    '''
    # Unit test for FordFulkerson
    g8 = FlowNetwork.fromFile("flownet8.txt")    
    ff8 = FordFulkerson(g8, 0, g8.V-1)
    print("ff8.flow", ff8.flow) # 28.0
    print("ff8.g", ff8.g)    
    print("ff8.inCut:", end=' ') # 0 2 3 6
    for v in range(g8.V):
        if ff8.inCut(v): print(v, end=' ')
    print()
    print()

    g8m = FlowNetwork(8)
    g8m.addEdge(FlowEdge(0,1,6))
    g8m.addEdge(FlowEdge(0,2,1))
    g8m.addEdge(FlowEdge(0,3,0))
    g8m.addEdge(FlowEdge(1,4,float('inf')))
    g8m.addEdge(FlowEdge(1,5,float('inf')))
    g8m.addEdge(FlowEdge(2,4,float('inf')))
    g8m.addEdge(FlowEdge(2,6,float('inf')))
    g8m.addEdge(FlowEdge(3,5,float('inf')))
    g8m.addEdge(FlowEdge(3,6,float('inf')))
    g8m.addEdge(FlowEdge(4,7,0))
    g8m.addEdge(FlowEdge(5,7,5))
    g8m.addEdge(FlowEdge(6,7,6))
    ff8m = FordFulkerson(g8m, 0, g8m.V-1)
    print("ff8m.flow", ff8m.flow)     
    print("ff8m.g", ff8m.g)    
    print("ff8m.inCut:", end=' ') # 
    for v in range(g8m.V):
        if ff8m.inCut(v): print(v, end=' ')
    print()
    print()
    
    g4 = FlowNetwork.fromFile("flownet4.txt")    
    ff4 = FordFulkerson(g4, 0, g4.V-1)
    print("ff4.flow", ff4.flow) # 200.0
    print("ff4.inCut:", end=' ') # 0
    for v in range(g4.V):
        if ff4.inCut(v): print(v, end=' ')
    print()
    print("ff4.g", ff4.g)    
    print()

    g6v = FlowNetwork.fromFile("flownet6v.txt")    
    ff6v = FordFulkerson(g6v, 0, g6v.V-1)
    print("ff6v.flow", ff6v.flow) # 4.0
    print("ff6v.inCut:", end=' ') # 0 2
    for v in range(g6v.V):
        if ff6v.inCut(v): print(v, end=' ')
    print()
    print("ff6v.g", ff6v.g)
    print()

    g6h = FlowNetwork.fromFile("flownet6h.txt")    
    ff6h = FordFulkerson(g6h, 0, g6h.V-1)
    print("ff6h.flow", ff6h.flow) # 15.0
    print("ff6h.inCut:", end=' ') # 0
    for v in range(g6h.V):
        if ff6h.inCut(v): print(v, end=' ')
    print()
    print("ff6h.g", ff6h.g)
    print()

    g6ha = FlowNetwork.fromFile("flownet6ha.txt")        
    ff6ha = FordFulkerson(g6ha, 0, g6ha.V-1)
    print("ff6ha.flow", ff6ha.flow) # 19.0
    print("ff6ha.inCut:", end=' ') # 0
    for v in range(g6ha.V):
        if ff6ha.inCut(v): print(v, end=' ')
    print()
    print("ff6ha.g", ff6ha.g)    
    print()

    g6hb = FlowNetwork.fromFile("flownet6hb.txt")        
    ff6hb = FordFulkerson(g6hb, 0, g6hb.V-1)
    print("ff6hb.flow", ff6hb.flow) # 4.0
    print("ff6hb.inCut:", end=' ') # 0
    for v in range(g6hb.V):
        if ff6hb.inCut(v): print(v, end=' ')
    print()
    print("ff6hb.g", ff6hb.g)    
    print()
    
    g8lions = FlowNetwork.fromFile("flownet8lions.txt")
    ff8lions = FordFulkerson(g8lions, 0, g8lions.V-1)
    print("ff8lions.flow", ff8lions.flow) #
    print("ff8lions.inCut:", end=' ') # 
    for v in range(g8lions.V):
        if ff8lions.inCut(v): print(v, end=' ')
    print()
    print("ff8lions.g", ff8lions.g)
    print()
    
    g8dinos = FlowNetwork.fromFile("flownet8dinos.txt")
    ff8dinos = FordFulkerson(g8dinos, 0, g8dinos.V-1)
    print("ff8dinos.flow", ff8dinos.flow) #
    print("ff8dinos.inCut:", end=' ') # 
    for v in range(g8dinos.V):
        if ff8dinos.inCut(v): print(v, end=' ')
    print()
    print("ff8dinos.g", ff8dinos.g)
    print()
    
    g12heros = FlowNetwork.fromFile("flownet12heros.txt")
    ff12heros = FordFulkerson(g12heros, 0, g12heros.V-1)
    print("ff12heros.flow", ff12heros.flow) #
    print("ff12heros.inCut:", end=' ') # 
    for v in range(g12heros.V):
        if ff12heros.inCut(v): print(v, end=' ')
    print()
    print("ff12heros.g", ff12heros.g)
    print()
    '''

    # Unit test for BaseballElimination
    be4 = BaseballElimination("teams4.txt")
    print(be4)
    be4.printResult()    
    if be4.isEliminated("Giants") == (False, []): print("P ",end='')
    else: print("F ",end='')    
    if be4.isEliminated("Lions") == (True, ['Giants', 'Dinos']): print("P ",end='')
    else: print("F ",end='')    
    if be4.isEliminated("Dinos") == (False, []): print("P ",end='')
    else: print("F ",end='')    
    if be4.isEliminated("Eagles") == (True, ['Giants']): print("P ",end='')
    else: print("F ",end='')
    print()
    print()
    
    be5 = BaseballElimination("teams5.txt")
    print(be5)
    be5.printResult()    
    if be5.isEliminated("Dinos") == (False, []): print("P ",end='')
    else: print("F ",end='')    
    if be5.isEliminated("Landers") == (False, []): print("P ",end='')
    else: print("F ",end='')    
    if be5.isEliminated("Bears") == (False, []): print("P ",end='')
    else: print("F ",end='')    
    if be5.isEliminated("Twins") == (False, []): print("P ",end='')
    else: print("F ",end='')    
    if be5.isEliminated("Heros") == (True, ['Dinos', 'Landers', 'Bears', 'Twins']): print("P ",end='')
    else: print("F ",end='')
    print()
    print()

    be12 = BaseballElimination("teams12.txt")
    print(be12)
    be12.printResult()
    if be12.isEliminated("Poland") == (False, []): print("P ",end='')
    else: print("F ",end='')
    if be12.isEliminated("USA") == (False, []): print("P ",end='')
    else: print("F ",end='')
    if be12.isEliminated("Brazil") == (False, []): print("P ",end='')
    else: print("F ",end='')
    if be12.isEliminated("Iran") == (False, []): print("P ",end='')
    else: print("F ",end='')
    if be12.isEliminated("Italy") == (False, []): print("P ",end='')
    else: print("F ",end='')
    if be12.isEliminated("Cuba") == (False, []): print("P ",end='')
    else: print("F ",end='')
    if be12.isEliminated("Argentina") == (False, []): print("P ",end='')
    else: print("F ",end='')
    if be12.isEliminated("Korea") == (False, []): print("P ",end='')
    else: print("F ",end='')    
    if be12.isEliminated("Japan") == (True, ['Poland', 'USA', 'Brazil', 'Iran']): print("P ",end='')
    else: print("F ",end='')    
    if be12.isEliminated("Serbia") == (True, ['Poland', 'USA', 'Brazil', 'Iran']): print("P ",end='')
    else: print("F ",end='')    
    if be12.isEliminated("Egypt") == (True, ['Poland']): print("P ",end='')
    else: print("F ",end='')    
    if be12.isEliminated("Chile") == (True, ['Poland', 'USA', 'Brazil', 'Iran']): print("P ",end='')
    else: print("F ",end='')
    print()
    print()