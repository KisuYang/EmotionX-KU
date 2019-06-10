import os
import json
import random
random.seed(0)

from pytorch_pretrained_bert import BertTokenizer


def load_data(hparams, path):
  tokenizer = BertTokenizer.from_pretrained(hparams.bert_type)

  global_tokenized_dialog, global_string_label = [], []
  for dialog in json.loads(open(path, 'r', encoding='utf-8').read()):
    tokenized_dialog, string_label = [], []
    for utter in dialog:      
      tokenized_utter = tokenizer.tokenize(utter['utterance'].lower())
      if len(tokenized_dialog + tokenized_utter) + 1 > hparams.max_input_len:
        print('[CAUTION] over max_input_len: ', utter['utterance'])
        continue
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

  global_ids_label = []
  for string_label in global_string_label:
    ids_label = []
    for str_label in string_label:
      if str_label == 'neutral':
        ids_label.append(0)
      elif str_label == 'joy':
        ids_label.append(1)
      elif str_label == 'sadness':
        ids_label.append(2)
      elif str_label == 'anger':
        ids_label.append(3)
      else:
        ids_label.append(4)
    global_ids_label.append(ids_label)
  '''
  For examples,
  global_ids_dialog[0] = [6006, 16006, 102, 33222, 102]
  global_ids_label[0] = [2, 3]
  '''

  return global_ids_dialog, global_ids_label

def shuffle_trainset(train_dialogs, train_labels):
  random_idx_list = list(range(len(train_dialogs)))
  random.shuffle(random_idx_list)
  return ([train_dialogs[idx] for idx in random_idx_list],
      [train_labels[idx] for idx in random_idx_list])

def get_batch(global_data, batch_size, i_step):
  return global_data[i_step * batch_size : (i_step + 1) * batch_size]

def make_dir(directory):
  #CAUTION: overwrite
  if not os.path.exists(directory):
    os.makedirs(directory)

def print_params(model):
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name, param.size()) # print grad params in a model