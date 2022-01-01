import tensorflow as tf

from text_data_set import TextDataSet
from vocab import Vocab
from class_dict import CategoryDict


seg_test_file = './text_classification_data/cnews.test.seg.txt'
vocab_file = './text_classification_data/cnews.vocab.txt'
category_file = './text_classification_data/cnews.category.txt'


num_word_threshold = 10 #詞語的頻率，太少的話就忽略
num_timesteps = 600

vocab = Vocab(
    vocab_file,
    num_word_threshold
)

category_vocab = CategoryDict(category_file)

test_dataset = TextDataSet(
    seg_test_file, vocab, category_vocab, num_timesteps
)
from tensorflow.keras.models import load_model
model = load_model('news.h5')

X_test = test_dataset.get_all_data()
y_test = test_dataset.get_all_label()

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

import numpy as np



def get_category_name(category_file, id):
    with open(category_file, 'r', encoding="utf-8") as f:
      lines = f.readlines()
    for i in range(len(lines)):
        if (i == id):
            return lines[i].strip('\r\n')

def get_sentence(vocab_file, id):
    with open(vocab_file, 'r', encoding="utf-8") as f:
      lines = f.readlines()
    for i in range(len(lines)):
        if (i == id):
            return lines[i].strip('\r\n').split("\t")[0]

            
for i in range(1, 20):
    print("Predict", np.argmax(model.predict(X_test)[i]), y_test[i])
    #Predict Cateogory Name
    print( get_category_name(category_file, np.argmax(model.predict(X_test)[i])) )
    #Label
    print( get_category_name(category_file, y_test[i]) )
    
    s = ""
    for j in range( len(X_test[i]) ):
        s+=get_sentence(vocab_file, X_test[i][j])
    print(s)

#category_vocab = get_category_name(category_file, 10)
