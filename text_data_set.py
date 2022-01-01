import numpy as np

class TextDataSet:
  def __init__(self, filename, vocab, category_vocab, num_timesteps):
    self._vocab = vocab
    self._category_vocab = category_vocab
    self._num_timesteps = num_timesteps #在每個MiniBatch對齊
    # Matrix
    self._inputs = []
    # Vector
    self._outputs = []
    self._indicator = 0
    self._parse_file(filename)
  
  def _parse_file(self, filename):
    print(
        f"Loading data from {filename}"
    )

    with open(filename, "r", encoding="utf-8") as f:
      lines = f.readlines()
    
    for line in lines:
      label, content = line.strip("\r\n").split("\t")
      #Label 轉換為ID
      id_label = self._category_vocab.category_to_id(label)
      #內容 轉換為ID
      id_words = self._vocab.sentence_to_id(content)
      
      #太多，截斷
      
      id_words = id_words[0: self._num_timesteps]
      #id_words = id_words[0: 600]

      #太少, 填充
      padding_num = self._num_timesteps - len(id_words)
      id_words = id_words + [
        self._vocab.unk() for i in range(padding_num)
      ]

      self._inputs.append(id_words)
      self._outputs.append(id_label)

    self._inputs = np.asarray(self._inputs, dtype=np.int32)
    self._outputs = np.asarray(self._outputs, dtype=np.int32)
    self._random_shuffle()
  
  def _random_shuffle(self):
    p = np.random.permutation(len(self._inputs))
    self._inputs = self._inputs[p]
    self._outputs = self._outputs[p]

  def next_batch(self, batch_size):
    end_indicator = self._indicator + batch_size
    if end_indicator > len(self._inputs):
      # 大於的話，打亂後，回去頭
      self._random_shuffle()
      self._indicator = 0
      end_indicator = batch_size
    if end_indicator > len(self._inputs):
      #Batch size lager than size of sample
      raise Exception('Batch size lager than size of sample')
    
    batch_inputs = self._inputs[self._indicator: end_indicator]
    batch_outputs = self._outputs[self._indicator: end_indicator]
    self._indicator = end_indicator
    return batch_inputs, batch_outputs
  def get_all_data(self):
    return self._inputs
  
  def get_all_label(self):
    return self._outputs