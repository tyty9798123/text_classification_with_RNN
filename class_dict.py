class CategoryDict:
  def __init__(self, filename):
    self._category_to_id = {}
    with open(filename, 'r', encoding="utf-8") as f:
      lines = f.readlines()
    for line in lines:
      category = line.strip('\r\n')
      idx = len(self._category_to_id)
      self._category_to_id[category] = idx
  
  def size(self):
    return len(self._category_to_id)

  def category_to_id(self, category):
    if category not in self._category_to_id:
      raise Exception(f"{category} is not in our category list")
    return self._category_to_id[category]
