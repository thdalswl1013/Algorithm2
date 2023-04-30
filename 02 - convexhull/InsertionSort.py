def insertionSort(a):    
    for i in range(1, len(a)):
        key = a[i]     # Element to move at current iteration
        j = i-1
        while j>=0 and a[j] > key:
            a[j+1] = a[j]  # Move element a[j] to a[j+1] if a[j] > key
            j -= 1
        a[j+1] = key   # Place the key to the farthest it can go to the left
        #print(a)
    return a
    
if __name__ == "__main__":
    print(insertionSort([5,1,3,2]))
    print(insertionSort(["b", "f", "z", "d", "i", "k", "p", "v"]))
    #insertionSort([7,2,4,1,9,0])