# import library
import pandas as pd
import numpy as np
import glob, os

# import util fuctions here

from models.src.config import cfg

def read_lyrics(lyrics_path):
    # TH LYRICS PATH : cfg.TH_DATA_LYRICS
    # ENG LYRICS PATH : cfg.ENG_DATA_LYRICS
    all_files = glob.glob(lyrics_path)

    song_list = []

    for filename in all_files:
        song_df = pd.read_csv(filename, index_col=None, header=0)
        song_list.append(song_df)

    song_df = pd.concat(song_list, axis=0, ignore_index=True)