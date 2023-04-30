def selectionSort(a):    
    for i in range(len(a)-1):
        # Find the minimum in a[i]~a[N-1]
        min_idx = i
        for j in range(i+1, len(a)):
            if a[j] < a[min_idx]:
                min_idx = j
        
        # Swap the found minimum with a[i]
        a[i], a[min_idx] = a[min_idx], a[i]
        #print(a)
    return a
    
if __name__ == "__main__":
    print(selectionSort([5,1,3,2]))
    print(selectionSort(["b", "f", "z", "d", "i", "k", "p", "v"]))
    #selectionSort([7,2,4,1,9,0])