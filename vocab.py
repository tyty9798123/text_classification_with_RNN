class Vocab:
  def __init__(self, filename, num_word_threshold):
    self._word_to_id = {}
    self._unk = -1
    self._num_word_threshold = num_word_threshold
    self._read_dict(filename)

  def _read_dict(self, filename):
    with open(filename, 'r', encoding="utf-8") as f:
      lines = f.readlines();
    
    for line in lines:
      word, frequency = line.strip("\r\n").split("\t")
      frequency = int(frequency)

      if frequency < self._num_word_threshold: #過濾freq未達標準的詞語
        continue

      idx = len(self._word_to_id)
      
      if word == '<UNK>':
        self._unk = idx

      self._word_to_id[word] = idx #設定ID,不設定頻率
  
  def word_to_id(self, word):
    return self._word_to_id.get(word, self._unk) #不存在的話返回 _unk

  def unk(self):
    return self._unk
  
  def size(self):
    return len(self._word_to_id)

  def sentence_to_id(self, sentence):
    #把整個句子換成ID
    word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
    return word_ids