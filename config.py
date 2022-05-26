import os
import numpy as np
import pandas as pd
import torch
from utils.ordered_easydict import OrderedEasyDict as edict


__C = edict()
cfg = __C


__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.MAX_WORKERS = 8
__C.GLOBAL.BATCH_SIZE = 4


__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))


__C.DATA = edict()
__C.DATA.BASE_PATH = os.path.join(__C.ROOT_DIR, 'data_lyrics')

__C.DATA.Artist_URL = os.path.join(__C.DATA.BASE_PATH, 'ArtistURL.csv')
__C.DATA.FOLDER = edict()
__C.DATA.FOLDER.ENG_SONGS = os.path.join(__C.DATA.BASE_PATH, 'engsongs')
__C.DATA.FOLDER.THAI_SONGS = os.path.join(__C.DATA.BASE_PATH, 'thaisongs')


# pandas data
__C.PD_BASE_PATH = os.path.join(__C.DATA.BASE_PATH, 'pd')
if not os.path.exists(__C.PD_BASE_PATH):
    os.makedirs(__C.PD_BASE_PATH)

__C.SPECIAL_MARK = {'round_brackets' : r'\((.*?)\)',
                    'square_brackets' : r'\[(.*?)\]',
                    'curly_brackets' : r'\{(.*?)\}',
                    'dot' : r'\.',
                    'star' : r'\*',
                    'semi_colon' : r'\;',
                    'colon' : r'\:',
                    'exclamation_mark' : r'\!',
                    'slash' : r'\/',
                    'backslash' : r'\\',
                    'question_mark' : r'\?',
                    'hashtag' : r'\#',
                    'percent' : r'\%',
                    'plus' : r'\+',
                    'minus' : r'\-',
                    'comma' :  r'\,',
                    'single_quote' : r"\'",
                    'double_quote' :  r'\"',
                    'dollar_sign' : r'\$',
                    'ampersand' : r'\&',
                    'underscore' : r'\_',
                    'r"\(|\)"' :  r'\(|\)',
                    'r\[' :  r'\[',
                    'r\]' :  r'\]'}


