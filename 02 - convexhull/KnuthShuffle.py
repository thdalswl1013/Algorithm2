import random

def knuthShuffle(a):
    for i in range(1, len(a)):
        j = random.randint(0,i) # Randomly select a position among 0 ~ i
        a[i], a[j] = a[j], a[i] # Swap a[i], a[j]

    return a
    
if __name__ == "__main__":
    print(knuthShuffle([5,1,3,2]))
    print(knuthShuffle(["b", "f", "z", "d", "i", "k", "p", "v"]))
    