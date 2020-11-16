import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import PIL
import cv2
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# # keras
#import tensorflow as tf
#import tensorflow.keras.layers as L
#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing import sequence
#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#import albumentations

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
stop_words.update(['т.к', '..','▼', '▼ ▼ ▼', '▼▼▼▼', '▼ ▼ ▼ ▼ ▼', '▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼', '▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼', '▼ ▼ ▼ ▼ ▼ ▼ ▼', '▼ ▼ ▼ ▼ ▼ ▼'])
stop_words.update([8*'▼', 9*'▼', 10*'▼', 11*'▼', 12*'▼', 13*'▼', 14*'▼', 15*'▼', 16*'▼', 17*'▼', 18*'▼', 19*'▼'])
stop_words.update([20*'▼', 21*'▼', 22*'▼', 23*'▼', 24*'▼', 25*'▼', 26*'▼', 27*'▼', 28*'▼', 29*'▼', 30*'▼', 31*'▼', 32*'▼', 33*'▼', 34*'▼', 35*'▼'])
m = Mystem()

dict_descr = defaultdict(int)
popular_seq = {}

def remove_sw_lemma(data, stop_words=stop_words, ms=m):
  words = word_tokenize(data)
  wordsFiltered = ''
  for w in words:
    if w not in stop_words:
      wordsFiltered += w+' '
  return [x for x in ms.lemmatize(wordsFiltered) if x != ' ']


def dict_create(descr_elem):
  global dict_descr
  MIN_LEN_SEQ = 3
  MAX_LEN_SEQ = 10

  descr_elem_r = remove_sw_lemma(descr_elem)
  

  if(len_elem := len(descr_elem_r)) > MAX_LEN_SEQ:
    for ii in range(len_elem - MAX_LEN_SEQ+1):
      for jj in range(7):
        dict_descr[' '.join(descr_elem_r[ii:ii+MIN_LEN_SEQ+jj])] += 1
  elif len_elem > MIN_LEN_SEQ:
    for ii in range(len_elem-MIN_LEN_SEQ):
      dict_descr[' '.join(descr_elem_r[:ii+MIN_LEN_SEQ])]



train.description.apply(dict_create)

new_list = []

sorted_keys = sorted(dict_descr, key=dict_descr.get, reverse=True)
for i, item in enumerate(sorted_keys):
  for subitem in sorted_keys[i+1:i+4]:
    if item in subitem:
      sorted_keys.remove(item)
      #print('Break!!!')
      break
      
      

for item in sorted_keys[:40]:
  print(f"Подстрока:, \"{item}\" встречается в описаниях {dict_descr.get(item)} раз.")
