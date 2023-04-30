class LLRB:      
    class Node:                
        def __init__(self, key, val): # Constructor
            self.key, self.val = key, val
            self.left = self.right = None
            self.count = 1 # Number of nodes itself and below            
            self.red = True # Color of parent link

    def __init__(self): # Constructor
        self.root = None

    @staticmethod
    def getOnNode(h, key):
        while h != None:
            if key < h.key: h = h.left
            elif key > h.key: h = h.right
            else: return h.val # key == x.key
        return None # The key was NOT found

    def get(self, key):
        return LLRB.getOnNode(self.root, key)

    def contains(self, key):
        return self.get(key) != None

    @staticmethod    
    def isRed(x):
            if x == None: return False
            return x.red

    @staticmethod 
    def fixUp(h): # Fix the tree such that it conforms to the LLRB representation
        if h == None: return None
        if LLRB.isRed(h.right) and not LLRB.isRed(h.left): h = LLRB.rotateLeft(h)  # Lean right -> lean left
        if LLRB.isRed(h.left) and LLRB.isRed(h.left.left): h = LLRB.rotateRight(h) # 4-node all leaning left -> 4-node leaning left and right
        if LLRB.isRed(h.left) and LLRB.isRed(h.right): LLRB.flipColors(h) # Split a 4-node into two 2-nodes
        return h
    
    @staticmethod 
    def rotateLeft(h):
        assert(LLRB.isRed(h.right))
        x = h.right
        h.right = x.left
        x.left = h
        x.red = h.red
        h.red = True
        return x

    @staticmethod
    def rotateRight(h):
        assert(LLRB.isRed(h.left))
        x = h.left
        h.left = x.right
        x.right = h
        x.red = h.red
        h.red = True
        return x

    @staticmethod
    def moveRedLeft(h):
        LLRB.flipColors(h)
        if LLRB.isRed(h.right.left):
            h.right = LLRB.rotateRight(h.right)
            h = LLRB.rotateLeft(h)
            LLRB.flipColors(h)
        return h
    
    @staticmethod
    def moveRedRight(h):
        LLRB.flipColors(h)
        if LLRB.isRed(h.left.left):
            h = LLRB.rotateRight(h)
            LLRB.flipColors(h)
        return h

    @staticmethod
    def flipColors(h):
        #assert((not LLRB.isRed(h) and LLRB.isRed(h.left) and LLRB.isRed(h.right)) or\
        #    (LLRB.isRed(h) and not LLRB.isRed(h.left) and not LLRB.isRed(h.right)))        
        h.red = not h.red
        h.left.red = not h.left.red
        h.right.red = not h.right.red

    @staticmethod
    def deleteMin(h):
        if h.left == None: return None
        if not LLRB.isRed(h.left) and not LLRB.isRed(h.left.left):
            h = LLRB.moveRedLeft(h)
        h.left = LLRB.deleteMin(h.left)
        h = LLRB.fixUp(h)
        h.count = LLRB.sizeOnNode(h.left) + 1 + LLRB.sizeOnNode(h.right)
        return h

    def delete(self, key):
        def deleteOnNode(h, key):
            if h == None: return None
            if key < h.key:
                if h.left != None and not LLRB.isRed(h.left) and not LLRB.isRed(h.left.left):
                    h = LLRB.moveRedLeft(h)
                h.left = deleteOnNode(h.left, key)
            else:
                if LLRB.isRed(h.left): h = LLRB.rotateRight(h)
                if key == h.key and h.right == None: return None
                if h.right != None and not LLRB.isRed(h.right) and not LLRB.isRed(h.right.left):
                    h = LLRB.moveRedRight(h)
                if key == h.key: # Hibbard deletion: place the min in the right subtree on the deleted spot 
                    h.key = LLRB.minOnNode(h.right)
                    h.value = LLRB.getOnNode(h.right, h.key)
                    h.right = LLRB.deleteMin(h.right)
                else:
                    h.right = deleteOnNode(h.right, key)
            h = LLRB.fixUp(h)
            h.count = LLRB.sizeOnNode(h.left) + 1 + LLRB.sizeOnNode(h.right)        
            return h
        self.root = deleteOnNode(self.root, key)
        if self.root != None:
            self.root.red = False # To not violate the assertion in flipColors(h), where the root splits

    def put(self, key, val):
        def putOnNode(x, key, val):
            if x == None: return self.Node(key, val)
            if key < x.key: x.left = putOnNode(x.left, key, val)
            elif key > x.key: x.right = putOnNode(x.right, key, val)
            else: x.val = val # key == x.key
            x = LLRB.fixUp(x)
            x.count = LLRB.sizeOnNode(x.left) + 1 + LLRB.sizeOnNode(x.right)
            return x     
        self.root = putOnNode(self.root, key, val)
        self.root.red = False # To not violate the assertion in flipColors(h), where the root splits

    @staticmethod
    def minOnNode(h):
        if h == None: return None
        else:
            while h.left != None:
                h = h.left
        return h.key

    def min(self):
        return LLRB.minOnNode(self.root)
        
    def max(self):
        if self.root == None: return None
        else:            
            x = self.root
            while x.right != None:
                x = x.right
            return x.key

    def floor(self, key):
        def floorOnNode(x, key):
            if x == None: return None
            if key == x.key: return x
            elif key < x.key: return floorOnNode(x.left, key)

            t = floorOnNode(x.right, key)
            if t != None: return t
            else: return x
        x = floorOnNode(self.root, key)
        if x == None: return None
        else: return x.key

    def ceiling(self, key):
        def ceilingOnNode(x, key):
            if x == None: return None
            if key == x.key: return x
            elif x.key < key: return ceilingOnNode(x.right, key)

            t = ceilingOnNode(x.left, key)
            if t != None: return t
            else: return x
        x = ceilingOnNode(self.root, key)
        if x == None: return None
        else: return x.key

    @staticmethod
    def sizeOnNode(x):
            if x == None: return 0
            else: return x.count

    def size(self):        
        return LLRB.sizeOnNode(self.root)    

    def rank(self, key): # How many keys < key?
        def rankOnNode(x, key): # rank(key) on the subtree rooted at x
            if x == None: return 0
            if key < x.key: return rankOnNode(x.left, key)
            elif key > x.key: return LLRB.sizeOnNode(x.left) + 1 + rankOnNode(x.right, key)
            else: return LLRB.sizeOnNode(x.left) # key == x.key
        return rankOnNode(self.root, key)

    def select(self, idx):
        def selectOnNode(x, idx): # idx-th element on the subtree rooted at x
            if x == None: return None # idx-th element does not exist
            if idx < LLRB.sizeOnNode(x.left): return selectOnNode(x.left, idx)
            elif idx > LLRB.sizeOnNode(x.left): return selectOnNode(x.right, idx-LLRB.sizeOnNode(x.left)-1)
            else: return x.key # idx == LLRB.sizeOnNode(x.left)
        return selectOnNode(self.root, idx)        

    def inorder(self):        
        def inorderOnNode(x, q):
            if x == None: return
            inorderOnNode(x.left, q)
            q.append(x.key)
            inorderOnNode(x.right, q)
        q = []
        inorderOnNode(self.root, q)
        return q

    def levelorder(self):
        qNode, qKey, idx = [], [], 0
        if self.root == None: return qNode
        else: qNode.append(self.root)        
        while idx < len(qNode):
            x = qNode[idx]
            if x.left != None: qNode.append(x.left)
            if x.right != None: qNode.append(x.right)
            qKey.append(x.key)
            idx += 1
        return qKey

    def rangeCount(self, lo, hi): # Number of keys between lo and hi, both inclusive
        if self.contains(hi): return self.rank(hi) - self.rank(lo) + 1
        else: return self.rank(hi) - self.rank(lo)

    def rangeSearch(self, lo, hi): # Return all keys between lo and hi, both inclusive
        def rangeSearchOnNode(x, lo, hi, q):
            if x == None: return
            if lo < x.key: rangeSearchOnNode(x.left, lo, hi, q)
            if lo <= x.key and x.key <= hi: q.append(x.key)
            if x.key < hi: rangeSearchOnNode(x.right, lo, hi, q)
        q = []
        rangeSearchOnNode(self.root, lo, hi, q)
        return q

if __name__ == "__main__":   
    '''
    bst = LLRB() 
    print(bst.size())
    print("min", bst.min())
    print("max", bst.max())
    
    bst.put("a",1)
    bst.put("c",2)
    bst.put("e",3)
    bst.put("b",4)
    bst.put("c",5)
    print("level order", bst.levelorder())
    print("size", bst.size())

    print(bst.get("a"))
    print(bst.get("b"))
    print(bst.get("c"))
    print(bst.get("d"))
    print(bst.get("e"))
    print(bst.floor("a"))
    print(bst.floor("b"))   

    print("ceiling") 
    print(bst.ceiling("a"))
    print(bst.ceiling("b"))
    print(bst.ceiling("c"))
    print(bst.ceiling("d"))
    print(bst.ceiling("e"))
    print(bst.ceiling("f"))

    print("min", bst.min())
    print("max", bst.max())

    print("rank")
    print(bst.rank("a"))
    print(bst.rank("b"))
    print(bst.rank("c"))
    print(bst.rank("d"))
    print(bst.rank("e"))
    print(bst.rank("f"))

    print("select")
    print(bst.select(-1))
    print(bst.select(0))
    print(bst.select(1))
    print(bst.select(2))
    print(bst.select(3))
    print(bst.select(4))
    print(bst.select(5))
    print(bst.select(6))

    '''
    print("inorder traversal")
    bst2 = LLRB()
    bst2.put("S",1)
    bst2.put("E",2)
    bst2.put("Y",3)
    bst2.put("A",4)
    bst2.put("R",5)
    bst2.put("C",6)
    bst2.put("H",7)
    bst2.put("M",8)
    bst2.put("L",9)
    bst2.put("P",10)
    print(bst2.rank("H"))
    print(bst2.select(4))
    print("level order", bst2.levelorder())
    print("inorder",bst2.inorder()) 
    print("range count", bst2.rangeCount("F", "T"))
    print("range search", bst2.rangeSearch("F", "T"))
    print("range count", bst2.rangeCount("B", "I"))
    print("range search", bst2.rangeSearch("B", "I"))
    print("range count", bst2.rangeCount("C", "H"))
    print("range search", bst2.rangeSearch("C", "H"))
    print("range count", bst2.rangeCount("J", "R"))
    print("range search", bst2.rangeSearch("J", "R"))

    '''
    print("delete Z")
    bst2.delete("Z")
    print("level order", bst2.levelorder())
    print("inorder",bst2.inorder())

    print("delete M")
    bst2.delete("M")
    print("level order", bst2.levelorder())
    print("inorder",bst2.inorder())

    print("delete A")
    bst2.delete("A")
    print("level order", bst2.levelorder())
    print("inorder",bst2.inorder())
    
    print("delete L")
    bst2.delete("L")
    print("level order", bst2.levelorder())
    print("inorder",bst2.inorder())

    print("delete all")
    bst2.delete("C")
    bst2.delete("E")
    bst2.delete("H")
    bst2.delete("P")
    bst2.delete("R")
    bst2.delete("S")
    bst2.delete("Y")
    bst2.delete("X") # This element does not exist in the LLRB, so it will not delete any element
    print("level order", bst2.levelorder())
    print("inorder",bst2.inorder())
    '''