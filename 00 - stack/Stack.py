class Stack:
    def __init__(self): 
        self.list=[]

    def push(self, x):
        self.list.append(x)

    def pop(self):
        if len(self.list)==0:
           return None
        else:
            return self.list.pop()


s=Stack()

s.push(1)
s.push(2)
s.push(3)
s.push(4)

print(s.pop())
print(s.pop())
print(s.pop())
print(s.pop())
print(s.pop())
