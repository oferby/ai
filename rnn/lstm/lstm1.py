import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import text
import nltk
from nltk.tokenize import RegexpTokenizer
import re

f = open('yelp.txt', 'r')

lines = []
labels = []
words = set()
for line in f:
    split = line.split('\t')
    lines.append(split[0])
    labels.append(int(split[1]))

assert len(lines) > 0

tokenizer = RegexpTokenizer('[a-zA-Z]\w+')
for line in lines:
    tokens = tokenizer.tokenize(line)
    for t in tokens:
        words.add(t.lower())

print(len(words))

list_of_words = text.one_hot(' '.join(words), len(words))

model = Sequential()
model.add(Embedding(len(words) + 1, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# One-hot encoding (CountVectorizing)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

# learn the vocabulary and store CountVectorizer sparse matrix in X
X = vectorizer.fit_transform(lines)
X.toarray()
# transforming a new document according to learn vocabulary
vectorizer.transform(['A new document.']).toarray()
