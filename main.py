import numpy as np
import tensorflow as tf
import os
from vocab import Vocab
from class_dict import CategoryDict
from text_data_set import TextDataSet
# Input
train_file = './text_classification_data/cnews.train.txt'
test_file = './text_classification_data/cnews.test.txt'
val_file = './text_classification_data/cnews.val.txt'

# Output

# 分詞文件
seg_train_file = './text_classification_data/cnews.train.seg.txt'
seg_test_file = './text_classification_data/cnews.test.seg.txt'
seg_val_file = './text_classification_data/cnews.val.seg.txt'

# 詞語->ID的映射，詞表文件
vocab_file = './text_classification_data/cnews.vocab.txt'
# Label -> ID 的映射
category_file = './text_classification_data/cnews.category.txt'


num_embedding_size = 32
num_timesteps = 600
# LSTM步長，MiniBatch同等長度，
#但其實也有方法可以不同長度
num_lstm_node = [64, 64] #LSTM 每一層的 Size 
#layer = 2，所以才兩個64
num_lstm_layer = 2
num_fc_nodes = 64
batch_size = 100
clip_lstm_grads = 1.0 
# 控制LSTM的梯度大小，對每個梯度設定一個上限
# 如果超過上限就把梯度設成這個值
learning_rate = 0.001
num_word_threshold = 10 #詞語的頻率，太少的話就忽略


vocab = Vocab(
    vocab_file,
    num_word_threshold
)
print(
    f"vocab_size {vocab.size()}"
)
test_str = "的 在 了 是"
print(vocab.sentence_to_id(test_str))

vocab_size = vocab.size()


category_vocab = CategoryDict(category_file)
num_classes = category_vocab.size()

train_dataset = TextDataSet(
    seg_train_file, vocab, category_vocab, num_timesteps
)
val_dataset = TextDataSet(
    seg_val_file, vocab, category_vocab, num_timesteps
)
test_dataset = TextDataSet(
    seg_test_file, vocab, category_vocab, num_timesteps
)
import tensorflow as tf
from tensorflow import keras

#from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN, LSTM

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=num_embedding_size,
        input_length=num_timesteps
    )
)
model.add(LSTM(num_lstm_node[0]))
model.add(Dropout(0.2))

model.add(
    Dense(num_fc_nodes, activation="relu")
)

model.add(Dropout(0.2))
model.add(Dense(num_classes, activation="softmax"))
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(
    loss=loss_fn,
    optimizer="adam",
    metrics=["accuracy"]
)
print(model.summary())

X_train = train_dataset.get_all_data()
y_train = train_dataset.get_all_label()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystop = EarlyStopping(monitor='loss', patience=3, verbose=1)


model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    verbose=1,
    #callbacks=[earlystop]
)
#loss: 0.0420 - accuracy: 0.9924 - val_loss: 0.5311 - val_accuracy: 0.9338

model.save("./news.h5")
