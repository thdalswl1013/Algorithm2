stPokemon = {} # Create an empty dictionary

# Insert (key,value) pairs
stPokemon["리자몽"] = ["불꽃","비행"]
stPokemon["메가리자몽X"] = ["불꽃","비행"]
stPokemon["메가리자몽Y"] = ["불꽃","비행"]

# Given a key, find the corresponding value
print(stPokemon["메가리자몽X"])
stPokemon["메가리자몽X"].remove("비행")
stPokemon["메가리자몽X"].append("드래곤")
print(stPokemon["메가리자몽X"])

if "이상해꽃" not in stPokemon: stPokemon["이상해꽃"] = []
print(stPokemon["이상해꽃"])
types = stPokemon["이상해꽃"]
types.append("풀")
types.append("독")
print(stPokemon["이상해꽃"])


st = {"blue":[1,3,5], "red":[2,4,6], "purple":[7,8]}
if "white" in st: print("white in st")
else: print("white NOT in st")
print(st["blue"])
st["red"].append(9)
print(st["red"])


spamWords = set()
spamWords.add("buy")
spamWords.add("promotion")
print(spamWords)
spamWords.update(["sale", "urgent", "warning", "thankyou"])
print(spamWords)
spamWords.remove("thankyou")
print(spamWords)
if "urgent" in spamWords: print("urgent in spamWords")
else: print("urgent NOT in spamWords")
if "hope" not in spamWords: print("hope not in spamwords")
else: print("hope in spamWords")

colorWords = {'red','black','white','blue','yellow'}
for w1 in colorWords:
    for w2 in colorWords:
        if w1 != w2: print(w1,w2)
