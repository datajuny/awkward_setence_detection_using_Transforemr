
import re
import json

from nltk.tokenize import sent_tokenize
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from awkward_dl_model_v3 import *
from awkward_dl_hyper_param_v3 import *

def awkward_dl_inference_test(RESUME):

    
    # 전처리
    FILTERS = "([~,!?\"':;)(])"
    change_filter = re.compile(FILTERS)
    
    result = re.sub(change_filter, "", RESUME).lower()
    result = sent_tokenize(result)
    
    # 문장 1개이하로 들어왔을때 예외처리
    if len(result) < 2:
        print("예외")
        return np.array([1], dtype=np.float32)
    
    FILTERS_DOT = "[.]"
    change_filter_dot = re.compile(FILTERS_DOT)
    
    # 문장분리 후 dot(점) 제거
    for idx, i in enumerate(result):
        result[idx] = re.sub(change_filter_dot, "", i)
               
    tokenizer = Tokenizer(num_words = VOCAB_SIZE)
    
    with open('./data_in/wordIndex.json') as json_file:
        word_index = json.load(json_file)
        tokenizer.word_index = word_index
    
    result = tokenizer.texts_to_sequences(result)
    result = pad_sequences(result, maxlen=MAX_SEQ_LEN, padding='pre')
    
    SENTENCE_LEN = len(result)
    
    q1_pad_data = result[:-1]
    q2_pad_data = result[1:]
    
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"base":q1_pad_data, "hypothesis":q2_pad_data}, shuffle=False)
    predictions = np.array([p['is_duplicate'] for p in lstm_est.predict(input_fn=predict_input_fn)])
    
    return predictions
