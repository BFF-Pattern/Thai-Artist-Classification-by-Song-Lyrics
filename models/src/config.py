import os, sys
import numpy as np
import torch

# import util fuctions here

from models.src.ordered_easydict import OrderedEasyDict as edict

__C = edict()
cfg = __C

__C.GLOBAL = edict()
__C.GLOBAL = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.MAX_WORKERS = 8
__C.GLOBAL.BATCH_SIZE = 1

__C.WORD_MODEL = edict()
__C.WORD_MODEL.WORD_VECTOR_DIMENSION = 300 # WV_DIM
__C.WORD_MODEL._MODEL_NAME = "thai2fit_wv"
__C.WORD_MODEL._TK_SP = "xxspace"
__C.WORD_MODEL._TK_EOL = "xxeol"


__C.SPECIAL_MARK = {'round_brackets' : r'\((.*?)\)',
                    'square_brackets' : r'\[(.*?)\]',
                    'curly_brackets' : r'\{(.*?)\}',
                    'dot' : r'\.',
                    'star' : r'\*',
                    'semi_colon' : r'\;',
                    'colon' : r'\:',
                    'exclamation_mark' : r'\!',
                    'slash' : r'\\',
                    'slashR' : r'\/',
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
                    'r"\(|\)"' :  r'\(|\)'}


__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..', '..'))
__C.DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'data_lyrics')

# Song Lyrics Data

dir_th_lyrics = os.path.join(__C.DATA_BASE_PATH, 'thaisongs', '*.csv')
if os.path.exists(dir_th_lyrics):
    __C.TH_DATA_LYRICS = dir_th_lyrics

dir_eng_lyrics = os.path.join(__C.DATA_BASE_PATH, 'engsongs', '*.csv')
if os.path.exists(dir_eng_lyrics):
    __C.TH_DATA_LYRICS = dir_eng_lyrics


# Pandas Data

__C.PD_BASE_PATH = os.path.join(__C.DATA_BASE_PATH, 'pd')
if not os.path.exists(__C.PD_BASE_PATH):
    os.makedirs(__C.PD_BASE_PATH)