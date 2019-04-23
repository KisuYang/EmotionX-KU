import joblib
from datetime import datetime
from pytorch_pretrained_bert import BertTokenizer

def load_data(hparams, path):
  tokenizer = BertTokenizer.from_pretrained(hparams.bert_type)

  global_tokenized_dialog, global_string_label = [], []  
  for dialog in joblib.load(path):
    tokenized_dialog, string_label = [], []
    for utter in dialog:
      tokenized_utter = tokenizer.tokenize(utter['utterance'])
      tokenized_dialog += tokenized_utter + [hparams.sep_token]
      string_label.append(utter['emotion'])
    global_tokenized_dialog.append(tokenized_dialog)
    global_string_label.append(string_label)
  '''
  For examples,
  global_tokenized_dialog[0] = ['hello', 'world', '[SEP]', 'bye', '[SEP]']
  global_string_label[0] = ['joy', 'sadness']
  '''
  global_ids_dialog = [tokenizer.convert_tokens_to_ids(tokenized_dialog)
      for tokenized_dialog in global_tokenized_dialog]
  global_onehot_label = []
  for string_label in global_string_label:
    onehot_label = []
    for str_label in string_label:
      if str_label == 'neutral':
        onehot_label.append(0)
      elif str_label == 'joy':
        onehot_label.append(1)
      elif str_label == 'sadness':
        onehot_label.append(2)
      elif str_label == 'anger':
        onehot_label.append(3)
      else:
        onehot_label.append(4)
    global_onehot_label.append(onehot_label)
  '''
  For examples,
  global_ids_dialog[0] = [6006, 16006, 102, 33222, 102]
  global_onehot_label[0] = [2, 3]
  '''

  return global_ids_dialog, global_onehot_label

def get_batch(global_data, batch_size, i_step):
  return global_data[i_step * batch_size : (i_step + 1) * batch_size]

def get_time():
  return str(datetime.now()).replace(':','')

def make_dir():
  pass