import random

# Partition a[lo~hi] using a[lo] as the partitioning (pivot) item
def partition(a, lo, hi):
    i,j = lo+1,hi

    while True:
        while i<=hi and a[i]<a[lo]: i = i+1
        while j>=lo+1 and a[j]>a[lo]: j = j-1

        if (j <= i): break      # Break if the pointers cross
        a[i],a[j] = a[j],a[i]   # Swap a[i] and a[j]
        i,j = i+1,j-1

    a[lo],a[j] = a[j],a[lo]    # Swap a[j] with the partitioning item    
    return j    # Return the index of item in place    

# Find k-th smallest element, where k = 0 ~ len(a)-1
def quickSelect(a, k):
    # Randomly shuffle a, so that two partitions are nearly equally sized
    random.shuffle(a)

    lo, hi = 0, len(a)-1
    while (lo < hi):
        j = partition(a, lo, hi)
        if j<k: lo = j+1
        elif k<j: hi = j-1
        else: return a[k]
    
    return a[k]

if __name__ == "__main__":
    print(quickSelect(["k","r","a","t","e","l","e","p","u","i","m","q","c","x","o","s"], 0))
    print(quickSelect(["k","r","a","t","e","l","e","p","u","i","m","q","c","x","o","s"], 4))
    print(quickSelect(["k","r","a","t","e","l","e","p","u","i","m","q","c","x","o","s"], 10))
    print(quickSelect(["k","r","a","t","e","l","e","p","u","i","m","q","c","x","o","s"], 15))
    print(quickSelect(["k","k","k","k","k","k","k","k"],5))
    print(quickSelect([1,2,3,4,5,6,7,8,9,10],7))
    print(quickSelect([1,2,3,4,5,6,7,8,9],3))
    print(quickSelect([10,9,8,7,6,5,4,3,2,1],2))    
