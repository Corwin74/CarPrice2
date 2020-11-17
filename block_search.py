import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# NLP
from pymystem3 import Mystem
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

DATA_DIR = '~/CarPrice2/input/'
train = pd.read_csv(DATA_DIR + 'train.csv')
test = pd.read_csv(DATA_DIR + 'test.csv')
sample_submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')

stop_words = set(stopwords.words('russian'))
sw1 = set(", . ' : ; ! ? № % * ( ) [ ] { | } # $ ^ & - + < = > ` ~ 1 2 3 4 5 6 7 8 9 0 | @ · \' - ` , -  ― ")
sw2 = set("· • — ❗️ ✪ \\ / 😁 😊 😉 ∙ ✔ ► ₽ ″ « » … ✅ ☑️ 🤦 ● 🔰 ° 📌 📢 ☎ ▼ ➥ ☛ 。 🔝 ⬇️ ▶ 🥇 😀 🤗 ")
stop_words = stop_words.union(sw1)
stop_words = stop_words.union(sw2)
stop_words.update(['т.к', '..'])
stop_words.update([x * '▼' for x in range(1, 33)])
m = Mystem()

dict_descr = defaultdict(int)

def remove_sw_lemma(data, stop_word=stop_words, ms=m):
    words = word_tokenize(data)
    words_filtered = ''
    for w in words:
        if w not in stop_word:
            words_filtered += w + ' '
    return [x for x in ms.lemmatize(words_filtered) if x != ' ']


def dict_create(descr_elem):
    global dict_descr
    MIN_LEN_SEQ = 3
    MAX_LEN_SEQ = 18

    descr_elem_r = remove_sw_lemma(descr_elem)

    if (len_elem := len(descr_elem_r)) > MAX_LEN_SEQ:
        for ii in range(len_elem - MAX_LEN_SEQ + 1):
            for jj in range(MAX_LEN_SEQ - MIN_LEN_SEQ):
                dict_descr[' '.join(descr_elem_r[ii:ii + MIN_LEN_SEQ + jj])] += 1
    elif len_elem > MIN_LEN_SEQ:
        for ii in range(len_elem - MIN_LEN_SEQ):
            dict_descr[' '.join(descr_elem_r[:ii + MIN_LEN_SEQ])] += 1

train.description.apply(dict_create)

new_list = []

sorted_keys = sorted(dict_descr, key=dict_descr.get, reverse=True)
for i, item in enumerate(sorted_keys):
    dirty = False
    for subitem in sorted_keys[i + 1:i + 20] + sorted_keys[i - 20:i]:
        if item in subitem:
            dirty = True
            break
    if not dirty:
        new_list.append(item)

for item in new_list[:20]:
    print(f"Подстрока:, \"{item}\" встречается в описаниях {dict_descr.get(item)} раз.")
