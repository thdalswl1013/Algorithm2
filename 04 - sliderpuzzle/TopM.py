from queue import PriorityQueue
import random

def topM(input, m):
    pq = PriorityQueue()
    for i in input:
        pq.put(i)
        if pq.qsize() > m:
            pq.get()
    
    return [pq.get() for _ in range(m)]

if __name__ == "__main__":    
    print(topM([4,2,6,-100,-24,-6,2,26,42,77,89,0,-44],4))
    print(topM([4,2,6,-100,-24,77],3))
    print(topM(['Q','G','B','P','F'],2))

    quiz = []
    for _ in range(100):
        quiz.append('A')
        quiz.append('B')
        quiz.append('C')
    random.shuffle(quiz)
    print("quiz", topM(quiz,3))

    print(topM([('A','1/1/2022',10), ('B','7/13/2022',5), ('C','3/23/2021',12)],2))
    print(topM([('g',77), ('a',500), ('d',12), ('f',24), ('g',-100)],3))

    print(topM([(10,'A','1/1/2022'), (5,'B','7/13/2022'), (12,'C','3/23/2021')],2))
    print(topM([(77,'g'), (500,'a'), (12,'d'), (24,'f'), (-100,'g')],3))

    input = [('A','1/1/2022',10), ('B','7/13/2022',5), ('C','3/23/2021',12)]
    inputRearranged = [(i[2],i[0],i[1]) for i in input]
    result = topM(inputRearranged, 2)    
    resultRearranged = [(i[1],i[2],i[0]) for i in result]
    print(resultRearranged)

    print([(i[1],i[2],i[0]) for i in topM([(i[2],i[0],i[1]) for i in input], 2)])

    input = [('Turing', '6/17/1990', 644.08),\
        ('vonNeumann', '3/26/2002', 4121.85),\
        ('Dijkstra', '8/22/2007', 2678.40),\
        ('vonNeumann', '1/11/1999', 4409.74),\
        ('Dijkstra', '11/18/1995', 837.42),\
        ('Hoare',	'5/10/1993', 3229.27),\
        ('vonNeumann', '2/12/1994', 4732.35),\
        ('Hoare',	'8/18/1992', 4381.21),\
        ('Turing', '1/11/2002', 66.10),\
        ('Thompson', '2/27/2000', 4747.08)]

    inputRearranged = []
    for i in input:
        inputRearranged.append((i[2],i[0],i[1]))

    result = topM(inputRearranged, 5)

    resultRearranged = []
    for i in result:
        resultRearranged.append((i[1],i[2],i[0]))
    
    print(resultRearranged)


   
    