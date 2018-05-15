from nltk.corpus import stopwords, state_union
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer

text = "this is an example of a text that uses nlp python package"

stop_words = set(stopwords.words("english"))

print(stop_words)

words = word_tokenize(text)

filtered = [w for w in words if w not in stop_words]

print(filtered)

ps = PorterStemmer()
print(ps.stem('riding'))



