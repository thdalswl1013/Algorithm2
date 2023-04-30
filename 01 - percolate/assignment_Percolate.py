import statistics
import math
import random
import timeit

def root(i): # root(i)=ids[ids[...ids[i]...]]
    while i != ids[i]:
        i = ids[i]
    return i

def connected(p, q): # 연결이 되어있으면(=p와 q의 루트 값이 같으면) True, 연결이 안되어있으면 False 출력
        return root(p)==root(q)

def union(p, q):   
    id1, id2 = root(p), root(q)
    if id1 == id2:
        return

    if size[id1] <= size[id2]: 
        ids[id1] = id2
        size[id2] += size[id1]
        
    else:
        ids[id2] = id1
        size[id1] += size[id2]

def simulate(n, t):

    global ids
    global size
    global flow_data
    
    
    open_ratio=[]
    for x in range(t):
        size=[] # 1
        ids=[] # 0~24
        flow_data=[] # 0은 닫힌 상태, 1은 열린 상태
        

        for i in range(n*n):
            ids.append(i)
            size.append(1)
            flow_data.append(0)

        ids.append(n*n)
        size.append(1)
        ids.append(n*n+1)
        size.append(1)

        for i in range(n):
            union(i,n*n)
            union(i+n*(n-1), n*n+1)

        while not connected(n*n,n*n+1):
            open_n=random.randrange(0, n*n)
            if flow_data[open_n]==0: # open_n번째가 닫혀있다면
                flow_data[open_n]=1 # open_n번째를 열어줘

                if (open_n%n != 0) and (open_n%n != n-1) and not (0 < open_n < n) and not (n*n-n < open_n < n*n-1) :

                    if flow_data[open_n-1]==1:
                        union(open_n, open_n-1)

                    if flow_data[open_n+1]==1:
                        union(open_n, open_n+1)
                    
                    if flow_data[open_n-n]==1:
                        union(open_n, open_n-n)
                
                    if flow_data[open_n+n]==1:
                        union(open_n, open_n+n)

                elif open_n == 0: # 제일 왼쪽 상단인 경우

                    if flow_data[open_n+1]==1: # 오른쪽이 열려있다면    
                        union(open_n,open_n+1)

                    if flow_data[open_n+n]==1: # 아래가 열려있다면 
                        union(open_n, open_n+n)

                elif open_n == n-1: # 제일 오른쪽 상단인 경우

                    if flow_data[open_n-1]==1: # 왼쪽이 열려있다면  
                        union(open_n, open_n-1)
                    if flow_data[open_n+n]==1: # 아래가 열려있다면  
                        union(open_n, open_n+n)

                elif open_n == n*(n-1): # 제일 왼쪽 하단인 경우

                    if flow_data[open_n-n]==1: # 위쪽이 열려있다면
                        union(open_n, open_n-n)
                    if flow_data[open_n+1]==1: # 오른쪽이 열려있다면    
                        union(open_n, open_n+1)
            
                elif open_n == n*n-1: # 제일 오른쪽 하단인 경우

                    if flow_data[open_n-n]==1: # 위쪽이 열려있다면
                        union(open_n, open_n-n)
                    if flow_data[open_n-1]==1: # 왼쪽이 열려있다면
                        union(open_n, open_n-1)

                elif open_n>0 and open_n<n: # 제일 위쪽인 경우
                    
                    if flow_data[open_n+n]==1: # 아래가 열려있다면  
                        union(open_n, open_n+n)

                elif open_n>(n-1)*n and open_n<n*n-1: # 제일 아래쪽인 경우

                    if flow_data[open_n-n]==1: # 위쪽이 열려있다면
                        union(open_n, open_n-n)

                elif open_n%n==0: # 제일 왼쪽인 경우

                    if flow_data[open_n+1]==1: # 오른쪽이 열려있다면    
                        union(open_n, open_n+1)

                elif open_n%n==n-1: # 제일 오른쪽인 경우
            
                    if flow_data[open_n-1]==1: # 왼쪽이 열려있다면
                        union(open_n, open_n-1)
    
        open_number=0        
        open_number = flow_data.count(1) # flow_data 리스트 중 열려있는 객체의 개수를 구한 것
        
        ratio=(open_number)/(n*n) # 열려있는 객체의 비율
        open_ratio.append(ratio) # 열려있는 객체의 비율 값을 open_ratio 리스트에 추가를 해줌
    
    mean=statistics.mean(open_ratio)
    stdev=statistics.stdev(open_ratio)
    confidence_intereval_left=mean-1.96*stdev/math.sqrt(t)
    confidence_intereval_right=mean+1.96*stdev/math.sqrt(t)
    print('mean                          = {:.10f}'.format(mean))
    print('stdev                         = {:.10f}'.format(stdev))
    print("95% confident interval        = {:.10f}, {:.10f}".format(confidence_intereval_left, confidence_intereval_right))

    return mean, stdev


#print(timeit.timeit(lambda: simulate(200,100), number=1))
#print(timeit.timeit(lambda: simulate(2,10000), number=1))

simulate(200,100)
print()

simulate(2,10000)
print()

simulate(2,100000)
print()
