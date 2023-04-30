N = 8

ids = []
for idx in range(N):    
    ids.append(idx)

def connected(p, q):
    return ids[p] == ids[q]

def minMax(a, b):
    if a<b: return a,b
    else: return b,a

def union(p, q):
    id1, id2 = minMax(ids[p], ids[q])
    for idx, _ in enumerate(ids):
        if ids[idx] == id2: ids[idx] = id1

'''
Unit Test
'''
if __name__ == "__main__":
    union(4,1)
    print(ids)
    union(4,5)
    print(ids)
    union(2,3)
    print(ids)
    union(6,2)
    print(ids)
    union(3,6)
    print(ids)
    union(3,7)
    print(ids)
    print(connected(1,7))
    union(5,2)
    print(ids)
    print(connected(1,7))
    print(connected(0,6))
    union(0,3)
    print(ids)
    print(connected(0,6))
