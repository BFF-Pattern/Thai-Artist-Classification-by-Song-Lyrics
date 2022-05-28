# import libraries
import glob, os
import pandas as pd
import re
import shutil
import time
import nltk
import pickle
import numpy as np

import pythainlp
from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words

from tqdm import tqdm

# import util fuctions here

from config import cfg


nltk.download('words')
th_stop = tuple(thai_stopwords())
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()
# from pythainlp.ulmfit import process_thai


def clean(text, verbose=0, special_marks=cfg.SPECIAL_MARK_NAME):

    mark_count = {}

    for name, mark in tqdm(special_marks.items()):
        mark_count[name] = sum(list(text.map(lambda s: re.findall(mark, s))), [])
        text = text.map(lambda s: re.sub(mark, '', s))
    
    if (verbose==1):
        for name, count in mark_count.items():
            print(f'Number of {name}: {count}')
        
    return text


def split_word(text, allow_stemming=['english']):

    tokens = word_tokenize(text, engine='newmm')
    # Remove stop words ภาษาไทย และภาษาอังกฤษ
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop]

    # หารากศัพท์ภาษาไทย และภาษาอังกฤษ

    # English
    if 'english' in allow_stemming:
        tokens = [p_stemmer.stem(i) for i in tokens]

    # Thai
    if 'thai' in allow_stemming:
        tokens_temp = []
        for i in tokens:
            w_syn = wordnet.synsets(i)
            if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
                tokens_temp.append(w_syn[0].lemma_names('tha')[0])
            else:
                tokens_temp.append(i)
        tokens = tokens_temp

    # ลบตัวเลข
    tokens = [i for i in tokens if not i.isnumeric()]

    # ลบช่องว่าง
    tokens = [i for i in tokens if not ' ' in i]

    return tokens