class BST:
    class Node:
        def __init__(self, key, val): # Constructor
            self.key, self.val = key, val
            self.left = self.right = None
            self.count = 1 # Number of nodes itself and below

    def __init__(self): # Constructor
        self.root = None
    
    def get(self, key):
        x = self.root
        while x != None:
            if key < x.key: x = x.left
            elif key > x.key: x = x.right
            else: return x.val  # key == x.key
        return None # The key was NOT found

    def put(self, key, val):
        def putOnNode(x, key, val):
            if x == None: return self.Node(key, val)
            if key < x.key: x.left = putOnNode(x.left, key, val)
            elif key > x.key: x.right = putOnNode(x.right, key, val)
            else: x.val = val # key == x.key
            x.count = self.sizeOnNode(x.left) + 1 + self.sizeOnNode(x.right)
            return x
        self.root = putOnNode(self.root, key, val)

    def min(self):
        if self.root == None: return None
        else:
            x = self.root
            while x.left != None:
                x = x.left
            return x.key
    
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

    def sizeOnNode(self, x):
            if x == None: return 0
            else: return x.count

    def size(self):        
        return self.sizeOnNode(self.root)    

    def rank(self, key): # How many keys < key?
        def rankOnNode(x, key): # rank(key) on the subtree rooted at x
            if x == None: return 0
            if key < x.key: return rankOnNode(x.left, key)
            elif key > x.key: return self.sizeOnNode(x.left) + 1 + rankOnNode(x.right, key)
            else: return self.sizeOnNode(x.left) # key == x.key
        return rankOnNode(self.root, key)

    def select(self, idx):
        def selectOnNode(x, idx): # idx-th element on the subtree rooted at x
            if x == None: return None # idx-th element does not exist
            if idx < self.sizeOnNode(x.left): return selectOnNode(x.left, idx)
            elif idx > self.sizeOnNode(x.left): return selectOnNode(x.right, idx-self.sizeOnNode(x.left)-1)
            else: return x.key # idx == self.sizeOnNode(x.left)
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

    def delete(self, key):
        def minOnNode(x): # Find node with minimum key from the subtree rooted at x
            if x == None: return None
            else:            
                while x.left != None:
                    x = x.left
            return x
        def deleteOnNode(x, key):
            if x == None: return None
            if key < x.key: x.left = deleteOnNode(x.left, key)
            elif key > x.key: x.right = deleteOnNode(x.right, key)
            else:
                if x.right == None: return x.left
                if x.left == None: return x.right
                t = x
                x = minOnNode(t.right)
                x.right = deleteOnNode(t.right, x.key)
                x.left = t.left
            x.count = self.sizeOnNode(x.left) + 1 + self.sizeOnNode(x.right)
            return x
        self.root = deleteOnNode(self.root, key)

if __name__ == "__main__":   
    bst = BST() 
    print(bst.size())
    print("min", bst.min())
    print("max", bst.max())

    bst.put("a",1)
    bst.put("c",2)
    bst.put("e",3)
    bst.put("b",4)
    bst.put("c",5)
    print(bst.size())

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

    print("inorder traversal")
    bst2 = BST()
    bst2.put("S",1)
    bst2.put("E",2)
    bst2.put("Y",3)
    bst2.put("A",4)
    bst2.put("R",5)
    bst2.put("C",6)
    bst2.put("H",7)
    bst2.put("M",8)
    for k in bst2.inorder():
        print(k)
    print(bst2.rank("H"))
    print(bst2.select(4))

    print("delete")
    bst2.delete("C")
    print(bst2.inorder())   
    
    bst2.delete("E")
    print(bst2.inorder())   

    bst2.delete("Y")
    print(bst2.inorder())   

    bst2.delete("S")
    print(bst2.inorder())   

    bst2.delete("F") # Delete a key not in BST, so BST remains the same
    print(bst2.inorder())   


