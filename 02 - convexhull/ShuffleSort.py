import random

def shuffleSort(a):
    # Generate a random real number for each entry in a[]
    r = []
    for _ in range(len(a)):
        r.append(random.random())

    # Sort according to the random real number
    return [i for i, _ in sorted(zip(a,r), key=lambda x: x[1])]
    
if __name__ == "__main__":
    print(shuffleSort([5,1,3,2]))
    print(shuffleSort(["b", "f", "z", "d", "i", "k", "p", "v"]))
    