from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for i in range(10):
    print(tok[i])

words = word_tokenize(sample)

for i in range(10):
    print(words[i])

