import math

def ccw(i,j,k): # 오목한지 볼록한지 판단하는 함수
    area2=(j[0]-i[0]) * (k[1]-i[1]) - (j[1]-i[1]) * (k[0]-i[0])

    if area2>0: # 볼록하면 true
        return True
    else: # 오목하면 false
        return False


# y 값이 가장 작은 점 중 x 값이 가장 큰 점
def grahamScan(points):
    sorted_xy=sorted(points, key = lambda p: (p[1],-p[0])) # y값 다음 x값으로 정렬
    first=sorted_xy[0]
    stack=[]
    
    for i in range(len(points)):
        x=math.atan2(sorted_xy[i][1]-first[1], sorted_xy[i][0]-first[0])
        angle=(x,)
        sorted_xy[i]+=angle

    sorted_angle=sorted(sorted_xy, key = lambda p: p[2]) # angle 순으로 정렬 
    print(sorted_angle)
    stack.append(first)

    for i in sorted_angle:
        if len(stack)>=2:
            while ccw(stack[-2],stack[-1], i)==False:
                stack.pop()
        
                if len(stack) < 2:
                    break
        
        stack.append(i[0:2])
    print(stack)    


if __name__=="__main__":
    print(grahamScan([(0,0),(-2,-1),(-1,1),(1,-1),(3,-1),(-3,-1)]))
    print(grahamScan([(4,2),(3,-1),(2,-2),(1,0),(0,2),(0,-2),(-1,1),(-2,-1),(-2,-3),(-3,3),(-4,0),(-4,-2),(-4,-4)]))
