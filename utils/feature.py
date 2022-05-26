# import libraries
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, vstack

# import util fuctions here

from config import cfg
from utils.ordered_easydict import OrderedEasyDict as edict


def get_mean_vector(vec_lst):
    # calculate mean vector
    return csr_matrix(vstack(vec_lst).mean(axis=0))

def select_artist_song_create_feature(song_df, n_set, n_artist, n_song_min, n_song_artist_max, columns, stat_columns):
    song_count_group_df = song_df.groupby('artist')[['artist']].count().rename(columns={'artist': 'count'})
    artist_lst = list(song_count_group_df.loc[song_count_group_df['count'] >= n_song_min].index.values)

    n_set_total = sum(n_set.values())

    artist_set = []
    while len(artist_set) < n_set_total:
        new_artist = tuple(np.random.choice(artist_lst, size=n_artist, replace=False))
        if new_artist not in artist_set:
            artist_set.append(new_artist)

    # split artist sets
    artist_select = {}
    for field, n in n_set.items():
        i_select = np.random.choice(range(len(artist_set)), size=n, replace=False)
        artist_list = list(artist_set)
        artist_select[field] = [artist_list[i] for i in i_select]
        artist_set = [s for s in artist_set if s not in artist_select[field]]
    # create dataframe with all features
    feature_dict = {}
    # dictionary to map artist set id to list of artists
    set_id_to_artist_tp = {}

    i = 0
    for field, artist_set in artist_select.items():
        df_lst = []
        for artist_tp in artist_set:
            i += 1
            df = song_df.loc[song_df['artist'].isin(artist_tp), columns]
            # check if number of songs is too high
            if len(df) * n_artist > n_song_artist_max:
                df = df.sample(int(n_song_artist_max / n_artist), random_state=0)
                
            df['artist_set_id'] = i
            set_id_to_artist_tp[i] = artist_tp
            df_lst.append(df)
        feature_dict[field] = pd.concat(df_lst)  
        print('Number of songs in {}: {}'.format(field, len(feature_dict[field])))

    # get all selected artists
    artist_select_set = set.union(*[set(sum(tp_lst, ())) for tp_lst in artist_select.values()])

    # create artist dataframe from training data
    df_lst = []
    for artist, df in song_df.loc[song_df['artist'].isin(artist_select_set)].groupby('artist'):
        dic = {'artist': artist}
        # calculate averages and standard diviations
        for field in stat_columns:
            dic[field + '_mean'] = df[field].mean()
            dic[field + '_std'] = df[field].std()

        # number of songs
        dic['songs'] = len(df)

        # calculate average tf idf vector
        dic['tf_idf_vector_mean'] = get_mean_vector(df['tf_idf_vector'])

        df_lst.append(pd.DataFrame(dic, index=[0]))
    artist_feature_df = pd.concat(df_lst)

    def get_features(df):
        # get artist set id
        artist_set_id = df['artist_set_id'].iloc[0]
        
        # get all artists
        artist_feature_select_df = artist_feature_df.loc[artist_feature_df['artist']\
                                                         .isin(set_id_to_artist_tp[artist_set_id])]

        # merge dataframes
        artist_song_feature_df = pd.merge(artist_feature_select_df.assign(key=0), df.assign(key=0), on='key', 
                                          suffixes=['_artist', '_song']).drop('key', axis=1)    
        artist_song_feature_df['same_artist'] = \
            artist_song_feature_df['artist_artist'] == artist_song_feature_df['artist_song']

        # calculate features
        # add feature polarity
        for feature in stat_columns:
            artist_song_feature_df[feature + '_diff'] = \
                artist_song_feature_df[feature] - artist_song_feature_df[feature + '_mean']
            artist_song_feature_df[feature + '_diff_std'] = \
                artist_song_feature_df[feature + '_diff'] / artist_song_feature_df[feature + '_std']
        
        # calculate similarity of artist tf idf vector and song vector
        def tf_idf_vector_similarity(artist_vector, song_vector, songs, same_artist):
            # check if song is from same artist
            if same_artist:
                # deduct song vector from artist vector
                artist_vector = (songs * artist_vector - song_vector) / (songs - 1)
            # calculate similarity
            return cosine_similarity(artist_vector, song_vector)[0][0]

        # calculate vector similarity between artist and song
        artist_song_feature_df['vector_similarity'] = \
            artist_song_feature_df.apply(lambda row: tf_idf_vector_similarity(row['tf_idf_vector_mean'], 
                                                      row['tf_idf_vector'], row['songs'], row['same_artist']), axis=1)    
        return artist_song_feature_df

    artist_song_feature = {}
    for field in feature_dict:
        artist_song_feature[field] = feature_dict[field].groupby('artist_set_id').apply(get_features)\
                                                        .reset_index(drop=True)
        
    return artist_song_feature

# class FEATURE():
#     def __init__(self, song_df):
#         self.song_df = None
#         self.song_filter_df = None
#         self.update_df(song_df)

#     def update_df(self, song_df):
#             self.song_df = song_df
    
#     def get_artist_list(self, song_df, n_song_min=0):
#         if song_df is None:
#             song_df = self.song_df
        
#         assert n_song_min != 0

#         song_count_group_df = song_df.groupby('artist')[['artist']].count().rename(columns={'artist': 'count'})
#         artist_lst = list(song_count_group_df.loc[song_count_group_df['count'] >= n_song_min].index.values)
        
#         return artist_lst
    
#     def get_artist_set(self, artist_lst, n_set_total, n_artist):
#         artist_set = []
#         while len(artist_set) < n_set_total:
#             new_artist = tuple(np.random.choice(artist_lst, size=n_artist, replace=False))
#             if new_artist not in artist_set:
#                 artist_set.append(new_artist)
        
#         return artist_set
    
#     def sample_artist(self, artist_lst, n_set, n_artist):
#         n_set_total = sum(n_set.values())

#         artist_set = self.get_artist_set(artist_lst, n_set_total, n_artist)

#         # split artist sets
#         artist_select = {}
#         for field, n in n_set.items():
#             i_select = np.random.choice(range(len(artist_set)), size=n, replace=False)
#             artist_list = list(artist_set)
#             artist_select[field] = [artist_list[i] for i in i_select]
#             artist_set = [s for s in artist_set if s not in artist_select[field]]
        
#         return artist_select
    
#     def create_feature_dataframe(self, artist_select, n_artist, n_song_artist_max):
#         # create dataframe with all features
#         feature_dict = {}
#         # dictionary to map artist set id to list of artists
#         set_id_to_artist_tp = {}

#         song_df = self.song_df

#         i = 0
#         for field, artist_set in artist_select.items():
#             df_lst = []
#             for artist_tp in artist_set:
#                 i += 1
#                 columns = ['artist', 'song_name', 'n_words', 'unique_words_ratio', 'words_per_line', 'tf_idf_vector', 'tf_idf_score']

#                 df = song_df.loc[song_df['artist'].isin(artist_tp), columns]

#                 # check if number of songs is too high
#                 if len(df) * n_artist > n_song_artist_max:
#                     df = df.sample(int(n_song_artist_max / n_artist), random_state=0)
                    
#                 df['artist_set_id'] = i
#                 set_id_to_artist_tp[i] = artist_tp
#                 df_lst.append(df)
#             feature_dict[field] = pd.concat(df_lst)  
#             print('Number of songs in {}: {}'.format(field, len(feature_dict[field])))
        
#         return feature_dict, set_id_to_artist_tp

#     def select_artist_song_create_feature(self, n_set, n_artist, n_song_min, n_song_artist_max):
#         song_df = self.song_df

#         artist_lst = self.get_artist_list(n_song_min=n_song_min)
#         artist_select = self.sample_artist(artist_lst, n_set, n_artist)

        

#         i = 0
#         for field, artist_set in artist_select.items():
#             df_lst = []
#             for artist_tp in artist_set:
#                 i += 1
#                 columns = ['artist', 'song_name', 'n_words', 'unique_words_ratio', 'words_per_line', 'tf_idf_vector', 'tf_idf_score']

#                 df = song_df.loc[song_df['artist'].isin(artist_tp), columns]

#                 # check if number of songs is too high
#                 if len(df) * n_artist > n_song_artist_max:
#                     df = df.sample(int(n_song_artist_max / n_artist), random_state=0)
                    
#                 df['artist_set_id'] = i
#                 set_id_to_artist_tp[i] = artist_tp
#                 df_lst.append(df)
#             feature_dict[field] = pd.concat(df_lst)  
#             print('Number of songs in {}: {}'.format(field, len(feature_dict[field])))

#         # get all selected artists
#         artist_select_set = set.union(*[set(sum(tp_lst, ())) for tp_lst in artist_select.values()])

#         # create artist dataframe from training data
#         df_lst = []
#         for artist, df in song_df.loc[song_df['artist'].isin(artist_select_set)].groupby('artist'):
#             dic = {'artist': artist}
#             # calculate averages and standard diviations
#             for field in ['n_words', 'unique_words_ratio', 'words_per_line', 'tf_idf_score']:
#                 dic[field + '_mean'] = df[field].mean()
#                 dic[field + '_std'] = df[field].std()

#             # number of songs
#             dic['songs'] = len(df)

#             # calculate average tf idf vector
#             dic['tf_idf_vector_mean'] = get_mean_vector(df['tf_idf_vector'])

#             df_lst.append(pd.DataFrame(dic, index=[0]))
#         artist_feature_df = pd.concat(df_lst)

#         def get_features(df):
#             # get artist set id
#             artist_set_id = df['artist_set_id'].iloc[0]
            
#             # get all artists
#             artist_feature_select_df = artist_feature_df.loc[artist_feature_df['artist']\
#                                                             .isin(set_id_to_artist_tp[artist_set_id])]

#             # merge dataframes
#             artist_song_feature_df = pd.merge(artist_feature_select_df.assign(key=0), df.assign(key=0), on='key', 
#                                             suffixes=['_artist', '_song']).drop('key', axis=1)    
#             artist_song_feature_df['same_artist'] = \
#                 artist_song_feature_df['artist_artist'] == artist_song_feature_df['artist_song']

#             # calculate features
#             # add feature polarity
#             for feature in ['n_words', 'unique_words_ratio', 'words_per_line', 'tf_idf_score']:
#                 artist_song_feature_df[feature + '_diff'] = \
#                     artist_song_feature_df[feature] - artist_song_feature_df[feature + '_mean']
#                 artist_song_feature_df[feature + '_diff_std'] = \
#                     artist_song_feature_df[feature + '_diff'] / artist_song_feature_df[feature + '_std']
            
#             # calculate similarity of artist tf idf vector and song vector
#             def tf_idf_vector_similarity(artist_vector, song_vector, songs, same_artist):
#                 # check if song is from same artist
#                 if same_artist:
#                     # deduct song vector from artist vector
#                     artist_vector = (songs * artist_vector - song_vector) / (songs - 1)
#                 # calculate similarity
#                 return cosine_similarity(artist_vector, song_vector)[0][0]

#             # calculate vector similarity between artist and song
#             artist_song_feature_df['vector_similarity'] = \
#                 artist_song_feature_df.apply(lambda row: tf_idf_vector_similarity(row['tf_idf_vector_mean'], 
#                                                         row['tf_idf_vector'], row['songs'], row['same_artist']), axis=1)    
#             return artist_song_feature_df

#         artist_song_feature = {}
#         for field in feature_dict:
#             artist_song_feature[field] = feature_dict[field].groupby('artist_set_id').apply(get_features)\
#                                                             .reset_index(drop=True)
            
#         return artist_song_feature





