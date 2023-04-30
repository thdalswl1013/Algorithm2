st = {} # Define a symbol table (dictionary)

# Insert (key,value) pairs
st["www.knu.ac.kr"] = "155.230.11.1"
st["www.naver.com"] = "223.130.200.107"
st["dns.google"] = "8.8.8.8"

# Given a key, get the corresponding value
print(st["dns.google"])
print(len(st)) # Print the size of the symbol table

# Check to see whether a key is in the symbol table
if "dns.google" in st: print("dns.google is in st")
else: print("dns.google is NOT in st")

del st["dns.google"] # Delete a key and the corresponding value

# Check to see whether a key is in the symbol table
if "dns.google" in st: print("dns.google is in st")
else: print("dns.google is NOT in st")
print(len(st)) # Print the size of the symbol table

def frequencyCounter(sentences, minFrequency):
    st = {}
    for s in sentences:
        words = s.split()
        for w in words:
            if w not in st: st[w] = 1
            else: st[w] += 1
    
    for k, v in sorted(st.items(), key=lambda item: item[1], reverse=True):
        if (v >= minFrequency): print(k, v)

frequencyCounter(["it was the best of times",\
    "it was the worst of times",\
    "it was the age of wisdom",\
    "it was the age of foolishness",\
    "it was the epoch of belief",\
    "it was the epoch of incredulity",\
    "it was the season of light",\
    "it was the season of darkness",\
    "it was the spring of hope",\
    "it was the winter of despair",\
    ], 5)

st2 = {}
st2["a"] = 1
st2["a"] = 2
st2["a"] = 3
st2["a"] = 4
print(st2["a"], len(st2))

