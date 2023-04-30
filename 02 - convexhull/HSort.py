def shellSort(a):
    N = len(a)
    h = 1
    while h < N/3:
        h = 3*h + 1   # Knuth's Sequence 1, 4, 13, 40, 121, 364, ...

    numSwapsTotal = 0
    while h >= 1:        
        _, numSwaps = hInsertionSort(a, h)
        numSwapsTotal += numSwaps
        h = h//3
    
    return a, numSwapsTotal

def hInsertionSort(a, h):
    numSwaps = 0  
    for i in range(h, len(a)):  # Begin from a[h]
        key = a[i]     # Element to move at current iteration
        j = i-h
        while j>=0 and a[j] > key:
            a[j+h] = a[j]  # Move element a[j] to a[j+h] if a[j] > key
            numSwaps += 1
            j -= h         # Decrement j by h
        a[j+h] = key   # Place the key to the farthest it can go to the left      
    return a, numSwaps
    
if __name__ == "__main__":
    #print(hInsertionSort(["M", "O", "L", "E", "E", "X", "A", "S", "P", "R", "T"], 3))

    a1 = ["S", "H", "E", "L", "L", "S", "O", "R", "T", "E", "X", "A", "M", "P", "L", "E"]
    a2 = a1.copy()
    a3 = a1.copy()
    
    # Shell Sort with h = 13 --> 4 --> 1
    print("h-Sort with h = 13 --> 4 --> 1")
    print(hInsertionSort(a1, 13))
    print(hInsertionSort(a1, 4))
    print(hInsertionSort(a1, 1))

    # Insertion Sort
    print("Insertion Sort (i.e., Shell Sort with h = 1)")
    print(hInsertionSort(a2, 1))

    # Shell Sort with Knuth's Sequence
    #print("Shell Sort with Knuth's Sequence")
    #print(shellSort(a3))

    h = 1
    N = 40
    while h < N/3:        
        h = 3*h + 1
    print(h)
    