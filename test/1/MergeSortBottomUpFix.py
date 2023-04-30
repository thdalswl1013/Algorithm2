import random
import timeit

# Merge Sort는 stable할 수 있는 정렬 알고리즘이지만 첨부된 코드는 잘못 구현되어 stable하지 않다. 
# 이를 stable하게 수정하여 제출하시오.
# -> stable 했던 이유가 같은 값이면 왼쪽부터였기 떄문

# Merge a[lo ~ mid] with a[mid+1 ~ hi], using the extra space aux[]
def merge(a, aux, lo, mid, hi):
    # Copy elements in a[] to the auxiliary array aux[]
    for k in range(lo, hi+1):
        aux[k] = a[k]
    
    i, j = lo, mid+1
    for k in range(lo, hi+1):
        if i>mid: 
            a[k], j = aux[j], j+1

        elif j>hi: 
            a[k], i = aux[i], i+1

        elif aux[i] <= aux[j]: 
            a[k], i = aux[i], i+1

        elif aux[i] > aux[j]: 
            a[k], j = aux[j], j+1            

def mergeSort(a):
    # Create the auxiliary array once and re-use for all subsequent merges
    aux = [None] * len(a)
    size = 1

    while(size<len(a)): #size 이름 sz로 바꿔야함
        for lo in range(0, len(a)-size, size*2):
            merge(a, aux, lo, lo+size-1, min(lo+size+size-1, len(a)-1))           

        size += size  # Multiply by 2


class Record:
    def __init__(self, key, value) -> None:
        self.key, self.value = key, value
    
    def __lt__(self, other): # Behavior of < operator
        if self.key < other.key: return True
        else: return False    

    def __le__(self, other): # Behavior of <= operator
        if self.key <= other.key: return True
        else: return False

    def __eq__(self, other): # Behavior of == operator
        if self.key == other.key: return True
        else: return False

    def __gt__(self, other): # Behavior of > operator
        if self.key > other.key: return True
        else: return False

    def __ge__(self, other): # Behavior of >= operator
        if self.key >= other.key: return True
        else: return False

    def __str__(self) -> str:
        return f"{str(self.key)}:{str(self.value)}"

    def __repr__(self) -> str:
        return self.__str__()

def testStability(resultList):
    result = True
    for idx, e in enumerate(resultList):
        assert(isinstance(e, Record))
        if idx>0:
            if e.value > resultList[idx-1].value: pass
            else:
                result = False
                break
    return result

def selectionSort(a):    
    for i in range(len(a)-1):
        # Find the minimum in a[i]~a[N-1]
        min_idx = i
        for j in range(i+1, len(a)):
            if a[j] < a[min_idx]:
                min_idx = j
        
        # Swap the found minimum with a[i]
        a[i], a[min_idx] = a[min_idx], a[i]

if __name__ == "__main__":
    correct = True

    print("Stability test with 6 elements and 2 keys")
    l = [Record("a",1), Record("b",5), Record("a",2), Record("b",6), Record("a",3), Record("a",4)]
    r = l.copy()
    mergeSort(r)
    print(l, "-->", r)    
    if testStability(r): print("pass")
    else: 
        print("fail: [a:1, a:2, a:3, a:4, b:5, b:6] expected")    
        correct = False
    print()
    
    n = 100
    print(f"Stability test with {n*26} elements and duplicate keys")    
    l = []
    idx = 0
    for c in range(65,91):
        element = chr(c)
        for i in range(n):
            l.insert(i,Record(element, idx))
            idx += 1    
    r = l.copy()
    mergeSort(r)    
    if testStability(r): print("pass")
    else: 
        print("fail")    
        correct = False
    print()

    print("Speed test with random input")
    if correct:
        n=250
        a = [i for i in range(n)]
        random.shuffle(a)
        tMergeSort = timeit.timeit(lambda: mergeSort(a.copy()), number=n)/n
        tSelectionSort = timeit.timeit(lambda: selectionSort(a.copy()), number=n)/n
        print(f"For {len(a)} elements,\n mergeSort takes {tMergeSort:.10f} sec and selectionSort takes {tSelectionSort:.10f} sec on average")
        if tSelectionSort > tMergeSort*1.5: print("pass for timing")
        else: print("fail for timing")
    else:
        print("fail for timing (since the algorithm is not correct)")
    print()

