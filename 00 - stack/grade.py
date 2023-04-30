import random
import sys
import inspect

countPass = 0
countFail = 0
TotalCases = 107
def go():
    global countPass, countFail
    countPass += 1
    #print(f"pass {(countPass+countFail)}")
def fail(msg=""):
    global countPass, countFail
    countFail += 1
    print(f"Fail {(countPass+countFail)}", msg)
def failStop(msg=""):
    global countPass, countFail
    countFail += 1
    print(f"Fail {(countPass+countFail)}", msg)
    finish()
    quit()
    
def finish():
    print()
    if countPass + countFail != TotalCases: print("Test did not proceed to the end")
    print(f"Total test cases: {(countPass+countFail)}")
    print(f"    Fail: {countFail} cases")
    print(f"    Pass: {countPass} cases")

if __name__ == "__main__":
    modules = list(sys.modules.keys())
    import Stack as st
    modules.append("Stack")
    if list(sys.modules.keys()) == modules: go()
    else: fail("Import is not allowed")
        
    if hasattr(st,"Stack") and inspect.isclass(st.Stack): go()
    else: failStop("Class Stack is not defined")

    s = st.Stack()
    
    if hasattr(s, "push") and inspect.ismethod(s.push): go()
    else: failStop("Method push is not defined")
    if len(inspect.signature(s.push).parameters) == 1: go()
    else: failStop("Method push does not have 1 parameter")
    
    if hasattr(s, "pop") and inspect.ismethod(s.pop): go()
    else: failStop("Method pop is not defined")
    if len(inspect.signature(s.pop).parameters) == 0: go()
    else: failStop("Method pop does not have 0 parameter")
    
    v = [] # List to verify the stack with
    n = 100 # Number of random integers to push and pop
    for _ in range(n):
        i = random.randint(0,100)
        s.push(i)
        v.append(i)

    idx = len(v)-1
    for _ in range(n):
        if s.pop() == v[idx]: go()
        else: fail("pop() failed for random elements")
        idx -= 1    

    try:
        if s.pop() == None: go()
        else: fail("pop() does not return None when it is supposed to be empty")
    except:
        fail("pop() does not return None when it is supposed to be empty")

    finish()