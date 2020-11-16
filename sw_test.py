# %%
from pymystem3 import Mystem
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

# %%NLP Processing
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('russian'))
sw1 = set(", . ' : ; ! ? № % * ( ) [ ] { | } # $ ^ & - + < = > ` ~ 1 2 3 4 5 6 7 8 9 0 | @ · \' - `")
sw2 = set("· • — ❗️ ✪ \\ / 😁 😊 😉 ∙ ✔ ► ₽ ″ « » … ✅ ☑️ 🤦 ● 🔰 ° 📌 📢 ☎ ▼ ➥ ☛ 。 🔝 ⬇️ ▶")
stop_words = stop_words.union(sw1)
stop_words = stop_words.union(sw2)
print(stop_words)

m = Mystem()

def remove_sw_lemma(data, stop_words=stop_words, ms=m):
  words = word_tokenize(data)
  wordsFiltered = ''
  for w in words:
    if w not in stop_words:
      wordsFiltered += w+' '
  return [x for x in ms.lemmatize(wordsFiltered) if x != ' ']

print(remove_sw_lemma('я вася дурак и как бы тут эта все такое №  % 0 7 %'))
