
from awkward_dl_hyper_param_v3 import *

def rearrange(base, hypothesis, labels):
    features = {"base": base, "hypothesis": hypothesis}
    return features, labels


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((train_Q1, train_Q2, train_y))
    dataset = dataset.shuffle(buffer_size=len(train_Q1))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(rearrange)
    dataset = dataset.repeat(EPOCH)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((test_Q1, test_Q2, test_y))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(rearrange)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()


def Malstm(features, labels, mode):
        
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    def basic_bilstm_network(inputs, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            lstm_fw = [
                    tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(HIDDEN), output_keep_prob=DROPOUT_RATIO)
                    for layer in range(NUM_LAYERS)
                    ]
            lstm_bw = [
                    tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(HIDDEN), output_keep_prob=DROPOUT_RATIO)
                    for layer in range(NUM_LAYERS)
                    ]
            
            multi_lstm_fw = tf.nn.rnn_cell.MultiRNNCell(lstm_fw)
            multi_lstm_bw = tf.nn.rnn_cell.MultiRNNCell(lstm_bw)

            (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_lstm_fw,
                                                cell_bw=multi_lstm_bw,
                                                inputs=inputs,
                                                dtype=tf.float32)
            
            outputs = tf.concat([fw_outputs, bw_outputs], 2)
            
            return outputs[:,-1,:]
            
    embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    
    base_embedded_matrix = embedding(features['base'])
    hypothesis_embedded_matrix = embedding(features['hypothesis'])
    
    base_sementic_matrix = basic_bilstm_network(base_embedded_matrix, 'base')
    hypothesis_sementic_matrix = basic_bilstm_network(hypothesis_embedded_matrix, 'hypothesis')

    base_sementic_matrix = tf.keras.layers.Dropout(DROPOUT_RATIO)(base_sementic_matrix)
    hypothesis_sementic_matrix = tf.keras.layers.Dropout(DROPOUT_RATIO)(hypothesis_sementic_matrix)
    
#     merged_matrix = tf.concat([base_sementic_matrix, hypothesis_sementic_matrix], -1)
#     logit_layer = tf.keras.layers.dot([base_sementic_matrix, hypothesis_sementic_matrix], axes=1, normalize=True)    
#     logit_layer = K.exp(-K.sum(K.abs(base_sementic_matrix - hypothesis_sementic_matrix), axis=1, keepdims=True))
    
    logit_layer = tf.exp(-tf.reduce_sum(tf.abs(base_sementic_matrix - hypothesis_sementic_matrix), axis=1, keepdims=True))
    logit_layer = tf.squeeze(logit_layer, axis=-1)
                
    if PREDICT:
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  predictions={
                      'is_duplicate':logit_layer
                  })
    
    #prediction 진행 시, None
    if labels is not None:
        labels = tf.to_float(labels)
    
#     loss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=labels, y_pred=logit_layer))
    loss = tf.losses.mean_squared_error(labels=labels, predictions=logit_layer)
#     loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels, logit_layer))
    
    if EVAL:
        accuracy = tf.metrics.accuracy(labels, tf.round(logit_layer))
        eval_metric_ops = {'acc': accuracy}
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  eval_metric_ops= eval_metric_ops,
                  loss=loss)

    elif TRAIN:

        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(
                  mode = mode,
                  train_op = train_op,
                  loss = loss)
    
    
os.environ["CUDA_VISIBLE_DEVICES"]="0" #For GPU

model_dir = os.path.join(os.getcwd(), DATA_OUT_PATH + model)
os.makedirs(model_dir, exist_ok=True)

config_tf = tf.estimator.RunConfig()

lstm_est = tf.estimator.Estimator(Malstm, model_dir=model_dir)
