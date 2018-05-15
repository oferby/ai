from nltk.corpus import wordnet

syns = wordnet.synsets("program")

print(syns)

print(syns[0].lemmas()[0].name())

print(syns[0].definition())

print(syns[0].examples())


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2))


