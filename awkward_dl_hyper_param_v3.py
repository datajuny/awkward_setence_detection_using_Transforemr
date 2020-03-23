
import sys
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import simplejson as json

model = "v1_checkpoint_em512_hi512_40epoch_len60_batch1024/"

DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'

TRAIN_Q1_DATA_FILE = 'train_q1.npy'
TRAIN_Q2_DATA_FILE = 'train_q2.npy'
TRAIN_LABEL_DATA_FILE = 'train_label.npy'
NB_WORDS_DATA_FILE = 'data_configs.json'

BATCH_SIZE = 1024
EPOCH = 40
HIDDEN = 512

NUM_LAYERS = 2
DROPOUT_RATIO = 0.5

TEST_SPLIT = 0.05
RNG_SEED = 13371447
EMBEDDING_DIM = 512
MAX_SEQ_LEN = 60

## 데이터를 불러오는 부분이다. 효과적인 데이터 불러오기를 위해, 미리 넘파이 형태로 저장시킨 데이터를 로드한다.
q1_data = np.load(open(DATA_IN_PATH + TRAIN_Q1_DATA_FILE, 'rb'))
q2_data = np.load(open(DATA_IN_PATH + TRAIN_Q2_DATA_FILE, 'rb'))
labels = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA_FILE, 'rb'))
prepro_configs = None


with open(DATA_IN_PATH + NB_WORDS_DATA_FILE, 'r') as f:
    prepro_configs = json.load(f)
    
VOCAB_SIZE = prepro_configs['vocab_size']
BUFFER_SIZE = len(labels)
