N = 10

ids = []
size = []   # size[i]: size of tree rooted at i
for idx in range(N):    
    ids.append(idx)
    size.append(1)

def root(i):
    while i != ids[i]: i = ids[i]
    return i

def connected(p, q):
    return root(p) == root(q)

def union(p, q):    
    id1, id2 = root(p), root(q)
    if id1 == id2: return
    if size[id1] <= size[id2]: 
        ids[id1] = id2
        size[id2] += size[id1]
    else:
        ids[id2] = id1
        size[id1] += size[id2]

'''
Unit Test
'''
if __name__ == "__main__":
    union(6,5)
    print(ids)
    union(5,0)
    print(ids)
    union(2,1)
    print(ids)
    union(7,1)
    print(ids)
    union(4,3)
    print(ids)
    union(4,8)
    print(ids)
    union(6,7)
    print(ids)
    union(9,8)
    print(ids)
    union(7,3)
    print(ids)
    print(connected(5,4))
    print(connected(7,9))
