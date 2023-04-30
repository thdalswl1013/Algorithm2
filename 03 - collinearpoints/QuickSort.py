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

# Partition a[lo~hi] and then continue to partition each half recursively
def divideNPartition(a, lo, hi):
    if (hi <= lo): return
    j = partition(a, lo, hi)    
    divideNPartition(a, lo, j-1)
    divideNPartition(a, j+1, hi)

def quickSort(a):
    # Randomly shuffle a, so that the partitioning item is chosen randomly
    random.shuffle(a)

    divideNPartition(a, 0, len(a)-1)
    return a

if __name__ == "__main__":
    a = ["k","r","a","t","e","l","e","p","u","i","m","q","c","x","o","s"]
    print(partition(a,0,len(a)-1))
    print(a)

    print(quickSort(["k","r","a","t","e","l","e","p","u","i","m","q","c","x","o","s"]))
    print(quickSort(["k","k","k","k","k","k","k","k"]))
    print(quickSort([1,2,3,4,5,6,7,8,9,10]))
    print(quickSort([1,2,3,4,5,6,7,8,9]))
    print(quickSort([10,9,8,7,6,5,4,3,2,1]))

    a = ["E","C","A","I","E"]
    print(partition(a, 0, 4), a)
