class UnorderedMinPQ:
    def __init__(self): # Constructor
        self.pq = []
    
    def insert(self, key):
        self.pq.append(key)

    def delMin(self):
        minId, min = None, None
        for idx, item in enumerate(self.pq):
            if minId == None or item < min:
                minId, min = idx, item
        del self.pq[minId]
        return min 

    def size(self):
        return len(self.pq)

    def isEmpty(self):
        return len(self.pq) == 0

if __name__ == "__main__":    
    minPQ = UnorderedMinPQ()
    minPQ.insert('P')
    minPQ.insert('Q')
    minPQ.insert('E')
    print(minPQ.delMin())
    minPQ.insert('X')
    minPQ.insert('A')
    minPQ.insert('M')
    print(minPQ.delMin())
    minPQ.insert('P')
    minPQ.insert('L')
    minPQ.insert('E')
    print(minPQ.delMin())    
    print(minPQ.pq)
    