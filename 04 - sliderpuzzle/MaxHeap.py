class MaxHeap:
    def __init__(self): # Constructor
        self.pq = ['']  # Empty key at pq[0]

    def insert(self, key):
        self.pq.append(key)
        idx = len(self.pq)-1
        while idx>1 and self.pq[idx//2] < key:
            self.pq[idx], self.pq[idx//2] = self.pq[idx//2], self.pq[idx]
            idx = idx//2

    def delMax(self):        
        self.pq[1], self.pq[len(self.pq)-1] = self.pq[len(self.pq)-1], self.pq[1] 
        max = self.pq.pop()
        idx = 1
        while 2*idx <= len(self.pq)-1:    # If a child exists
            idxChild = 2*idx
            if idxChild<len(self.pq)-1 and self.pq[idxChild] < self.pq[idxChild+1]: idxChild = idxChild+1 # Find the greater child
            if self.pq[idx] >= self.pq[idxChild]: break
            self.pq[idx], self.pq[idxChild] = self.pq[idxChild], self.pq[idx] # Swap with (i.e., sink to) the greater child
            idx = idxChild
        return max

    def size(self):
        return len(self.pq) - 1

    def isEmpty(self):
        return len(self.pq) <= 1

if __name__ == "__main__":    
    maxPQ = MaxHeap()
    maxPQ.insert('P')
    print(maxPQ.pq)
    maxPQ.insert('Q')
    print(maxPQ.pq)
    maxPQ.insert('E')
    print(maxPQ.pq)
    maxPQ.insert('X')
    print(maxPQ.pq)
    maxPQ.insert('A')
    print(maxPQ.pq)
    maxPQ.insert('M')
    print(maxPQ.pq)
    maxPQ.insert('P')
    print(maxPQ.pq)
    maxPQ.insert('L')
    print(maxPQ.pq)
    maxPQ.insert('E')
    print(maxPQ.pq)    
    
    print(maxPQ.delMax())
    print(maxPQ.pq)
    print(maxPQ.delMax())
    print(maxPQ.pq)
    print(maxPQ.delMax())
    print(maxPQ.pq)
    print(maxPQ.delMax())
    print(maxPQ.pq)
    print(maxPQ.delMax())
    print(maxPQ.pq)
    print(maxPQ.delMax())
    print(maxPQ.pq)
    print(maxPQ.delMax())
    print(maxPQ.pq)
    print(maxPQ.delMax())
    print(maxPQ.pq)
    print(maxPQ.delMax())
    print(maxPQ.pq)


    