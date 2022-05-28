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
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from pythainlp.corpus import get_corpus_path
from pythainlp.tokenize import THAI2FIT_TOKENIZER, word_tokenize

from typing import List, Tuple
import warnings

# import util fuctions here

from config import cfg
from utils.tokenizer import split_word


def get_model() -> Word2VecKeyedVectors:
    path = get_corpus_path(cfg.WORD_MODEL._MODEL_NAME)
    return KeyedVectors.load_word2vec_format(path, binary=True)


def sentence_vectorizer(text: str, use_mean: bool = True) -> np.ndarray:
    model = get_model()

    vec = np.zeros((1, cfg.WORD_MODEL.WORD_VECTOR_DIMENSION))

    #words = THAI2FIT_TOKENIZER.word_tokenize(text)

    words = split_word(text)
    print(words)
    len_words = len(words)

    if not len_words:
        return vec

    for word in words:
        if word == " ":
            word = cfg.WORD_MODEL._TK_SP
        elif word == "\n":
            word = cfg.WORD_MODEL._TK_EOL

        if word in model.index_to_key:
            vec += model.get_vector(word)

    if use_mean:
        vec /= len_words

    return vec
