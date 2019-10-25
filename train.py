class FLAGS:
  def __init__(self):
    # embedding
    self.w2v_file = ""
    self.embedding_dim = 256
 
    # input struct
    self.sen_len = 30
    self.doc_len = 75
 
    # model struct
    self.n_hidden = 100
    self.n_class = 2
 
    # log
    self.log_file_name = ""
 
    # train
    self.training_iter = 15
    self.scope = 'rnn'
 
    # tune
    self.batch_size = 32
    self.learning_rate = 0.005
    self.keep_prob = 0.8
    self.l2_reg = 0.00001
 
    # lambda1 and lambda2, i don't know what's this
    self.cause = 1.000
    self.pos = 1.00
 
flags = FLAGS()



class model(Model):
  # build model
  def __init__(self, word_embedding):
    super(model, self).__init__()
    # embedding matrix 
    self.word_embedding = word_embedding
    # base function
    self.self_attention = layers.Attention()
    self.softmax = layers.Softmax()
 
    # sentence feature fetch
    self.base_lstm_forward = layers.LSTM(flags.n_hidden, return_sequences=True, return_state=True)
    self.base_lstm_backward = layers.LSTM(flags.n_hidden, return_sequences=True, return_state=True, go_backwards=True)
    self.base_bilstm = layers.Bidirectional(layer=self.base_lstm_forward, merge_mode='concat', backward_layer=self.base_lstm_backward)
 
    # cause
    self.cause_lstm_forward = layers.LSTM(flags.n_hidden, return_sequences=True, return_state=True)
    self.cause_lstm_backward = layers.LSTM(flags.n_hidden, return_sequences=True, return_state=True, go_backwards=True)
    self.cause_bilstm = layers.Bidirectional(layer=self.cause_lstm_forward, merge_mode='concat', backward_layer=self.cause_lstm_backward)
    self.cause_dense_attention = layers.Dense(flags.n_hidden * 2)
    self.cause_dense = layers.Dense(flags.n_class)
 
    # effect
    self.effect_lstm_forward = layers.LSTM(flags.n_hidden, return_sequences=True, return_state=True)
    self.effect_lstm_backward = layers.LSTM(flags.n_hidden, return_sequences=True, return_state=True, go_backwards=True)
    self.effect_bilstm = layers.Bidirectional(layer=self.effect_lstm_forward, merge_mode='concat', backward_layer=self.effect_lstm_backward)
    self.effect_dense_attention = layers.Dense(flags.n_hidden * 2)
    self.effect_dense = layers.Dense(flags.n_class)
  
  # forward pass
  def call(self, x, is_training=False):
    # x = [batch_size, doc_len, sen_len]
    # embedding -> x = [batch_size, doc_len, sen_len, 256]
    print(x.shape)
    x = tf.nn.embedding_lookup(params=self.word_embedding, ids=tf.reshape(x, [-1, flags.sen_len]))
    print(x.shape)
    # x = [batch_size * doc_len, sen_len, 256]
    x = tf.reshape(x, [-1, flags.sen_len, flags.embedding_dim])
    x = tf.nn.dropout(x, rate=flags.keep_prob)
 
    # for casue
    # fetch sentence feature
    print("fetch sentence feature")
    print(x.shape)
    x_cause_len_output = self.base_bilstm(x)
    # x_cause_len = [batch_size * doc_len, sen_len, 100 * 2]
    x_cause_len = x_cause_len_output[0]
    print(x_cause_len.shape)
    # x_attention_cause = [batch_size * doc_len, sen_len, 100 * 2]
    x_attention_cause = self.self_attention([x_cause_len, x_cause_len])
    print(x_attention_cause.shape)
    # x_attention_cause = [batch_size * doc_len, sen_len * 100 * 2]
    x_attention_cause = tf.reshape(x_attention_cause, [-1, flags.sen_len * flags.n_hidden * 2])
    print(x_attention_cause.shape)
    # x_casue_sen_feature = []
    x_casue_sen_feature = self.cause_dense_attention(x_attention_cause)
    print(x_casue_sen_feature.shape)
    # x_casue_sen_feature = [batch_size, doc_len, 100 * 2]
    x_casue_sen_feature = tf.reshape(x_casue_sen_feature, [-1, flags.doc_len, flags.n_hidden * 2])
    print(x_casue_sen_feature.shape)
 
    # fetch cause
    print("fetch cause")
    x_cause_doc_output = self.effect_bilstm(x_casue_sen_feature)
    # x_cause_doc = [batch_size, doc_len, 100 * 2]
    x_cause_doc = x_cause_doc_output[0]
    print(x_cause_doc.shape)
    # x_cause_doc = [batch_size * doc_len, 100 * 2]
    x_cause_doc = tf.reshape(x_cause_doc, [-1, flags.n_hidden * 2])
    print(x_cause_doc.shape)
    # y_cause = [batch_size * doc_len, 2]
    y_cause = self.cause_dense(x_cause_doc)
    y_cause = self.softmax(y_cause)
    # y_cause = [bathc_size, doc_len, n_class]
    y_cause = tf.reshape(y_cause, [-1, flags.doc_len, flags.n_class])
    print(y_cause.shape)
 
    # for effect
    # fetch sentence feature
    print("fetch sentence feature")
    print(x.shape)
    x_effect_len_output = self.base_bilstm(x)
    # x_effect_len = [batch_size * doc_len, sen_len, 100 * 2]
    x_effect_len = x_effect_len_output[0]
    print(x_effect_len.shape)
    # x_attention_effect = [batch_size * doc_len, sen_len, 100 * 2]
    x_attention_effect = self.self_attention([x_effect_len, x_effect_len])
    print(x_attention_effect.shape)
    # x_attention_effect = [batch_size * doc_len, sen_len * 100 * 2]
    x_attention_effect = tf.reshape(x_attention_effect, [-1, flags.sen_len * flags.n_hidden * 2])
    print(x_attention_effect.shape)
    # x_effect_sen_feature = []
    x_effect_sen_feature = self.effect_dense_attention(x_attention_effect)
    print(x_effect_sen_feature.shape)
    # x_effect_sen_feature = [batch_size, doc_len, 100 * 2]
    x_effect_sen_feature = tf.reshape(x_effect_sen_feature, [-1, flags.doc_len, flags.n_hidden * 2])
    print(x_effect_sen_feature.shape)
 
    # fetch effect
    print("fetch effect")
    x_effect_doc_output = self.effect_bilstm(x_casue_sen_feature)
    # x_effect_doc = [batch_size, doc_len, 100 * 2]
    x_effect_doc = x_effect_doc_output[0]
    print(x_effect_doc.shape)
    # x_effect_doc = [batch_size * doc_len, 100 * 2]
    x_effect_doc = tf.reshape(x_effect_doc, [-1, flags.n_hidden * 2])
    print(x_effect_doc.shape)
    # y_effect = [batch_size * doc_len, 2]
    y_effect = self.effect_dense(x_effect_doc)
    y_effect = self.softmax(y_effect)
    # y_effect = [bathc_size, doc_len, n_class]
    y_effect = tf.reshape(y_effect, [-1, flags.doc_len, flags.n_class])
    print(y_effect.shape)
 
    return y_cause, y_effect




sample_input = tf.ones([flags.batch_size, flags.doc_len, flags.sen_len], tf.int32)
sample_word_embedding = tf.ones([30000, 256], tf.float32)
print(sample_input.shape, sample_word_embedding.shape)
test_model = model(sample_word_embedding)
result = test_model(sample_input)
