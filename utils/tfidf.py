# import libraries
import glob, os
import pandas as pd
import re
import shutil
import time
import pickle
import numpy as np

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, vstack

# import util fuctions here

from config import cfg
from utils.ordered_easydict import OrderedEasyDict as edict



# calculate similarity of artist tf idf vector and song vector
def tf_idf_vector_similarity(artist_vector, song_vector, songs, same_artist):
    # check if song is from same artist
    if same_artist:
        # deduct song vector from artist vector
        artist_vector = (songs * artist_vector - song_vector) / (songs - 1)
    # calculate similarity
    return cosine_similarity(artist_vector, song_vector)[0][0]

class TFIDF():
    def __init__(self, song_df, song_filter_df):
        self.song_df = None
        self.song_filter_df = None
        self.update_df(song_df, song_filter_df)
        self.cv = None
        self.word_count_vector = None
        self.tfidf_transformer = None
    
    def init_tfidf(self, song_df=None):
        if song_df is None:
            song_df = self.song_df
        
        # initialise count vectorizer
        self.cv = CountVectorizer(analyzer=lambda x:x.split())
        self.word_count_vector = self.cv.fit_transform(song_df['words_str'])

        # compute idf
        self.tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        self.tfidf_transformer.fit(self.word_count_vector)

        self.tfidf_df = pd.DataFrame({'word': self.cv.get_feature_names_out(), 'weight': self.tfidf_transformer.idf_})

        return self.tfidf_df
    
    def update_df(self, song_df, song_filter_df):
        self.song_df = song_df
        self.song_filter_df = song_filter_df

    def add_tfidf_vector_lst(self, song_df=None, song_filter_df=None):
        if song_df is None:
            song_df = self.song_df
        if song_filter_df is None:
            song_filter_df = self.song_filter_df
        
        # assign tf idf scores to each song
        tf_idf_vector = self.tfidf_transformer.transform(self.word_count_vector)

        # attach count vectors to dataframe
        tf_idf_vector_lst = [-1] * len(song_df)
        for i in range(len(song_df)):
            tf_idf_vector_lst[i] = tf_idf_vector[i]
        song_df['tf_idf_vector'] = tf_idf_vector_lst    

        song_df['tf_idf_score'] = song_df['tf_idf_vector'].map(lambda vec: np.sum(vec.todense()))

        # join valus to selected artists
        song_filter_df = song_filter_df.join(song_df[['tf_idf_vector', 'tf_idf_score']])

        self.update_df(song_df, song_filter_df)

        return song_df, song_filter_df


# calculate similarity of artist tf idf vector and song vector
def tf_idf_vector_similarity(artist_vector, song_vector, songs, same_artist):
    # check if song is from same artist
    if same_artist:
        # deduct song vector from artist vector
        artist_vector = (songs * artist_vector - song_vector) / (songs - 1)
    # calculate similarity
    return cosine_similarity(artist_vector, song_vector)[0][0]