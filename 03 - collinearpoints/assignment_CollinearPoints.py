
def collinearPoints(points):
    sorted_xy=sorted(points, key = lambda p: (p[0],p[1])) # x값 다음 y값으로 오름차순 정렬 
    print(sorted_xy)

    incline_arr=[0] * len(sorted_xy)
    count=0

    for k in range(len(sorted_xy)):
        center=sorted_xy[k] #중심 점을 바꿔가면서..

        for i in range(len(sorted_xy)):
            if i==k:
                incline_arr[i]="no"
            else:
                incline_arr[i]=incline(center[0], center[1], sorted_xy[i][0], sorted_xy[i][1])

        print("center", center)
        print(incline_arr)      
        print()


            
        """

        dictionary_point_incline = { name:value for name, value in zip(sorted_xy, incline_arr) }  # (점, 기울기) 딕셔너리
        word_list=sorted(dictionary_point_incline.items(),key=lambda x:x[1])

        #print(word_list)

        print("딕셔너리: ", dictionary_point_incline)        
        """




def incline(x1,y1, x2,y2):
    if x1-x2==0:
        return 100 
    else:
        return (y1-y2)/(x1-x2)


if __name__ == "__main__":
    #collinearPoints([(19000,10000), (18000,10000), (32000,10000), (21000,10000), (1234,5678), (14000,10000)])

    #collinearPoints([(10000,0), (3000,7000), (7000,3000), (2000,21000), (3000,4000), (14000,15000), (6000,7000)])

    collinearPoints([(0,0), (1,1), (3,3), (4,4), (6,6), (7,7), (9,9)])

    #collinearPoints([(7,0), (14,0), (22,0), (27,0), (31,0), (42,0)])

    #collinearPoints([(12446,18993), (12798, 19345), (12834,19381), (12870, 19417), (12906, 19453), (12942,19489)])

    #collinearPoints([(1,1), (2,2), (3,3), (4,4), (2,0), (3,-1), (4,-2), (0,1), (-1,1), (-2,1), (-3,1), (2,1), (3,1), (4,1), (5,1)])
