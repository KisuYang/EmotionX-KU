import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert import BertModel

class YksModel_BERT_FC1(nn.Module):
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    self.bert_model = BertModel.from_pretrained(self.hparams.bert_type)
    self.linear_h = nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)
    self.selu = nn.SELU()
    self.dropout = nn.Dropout(p=self.hparams.dropout)
    self.linear_o = nn.Linear(self.hparams.hidden_size, self.hparams.n_class)
    self.softmax = nn.Softmax(dim=1)
    self.loss = self._define_weighted_cross_entropy_loss()    

  def _define_weighted_cross_entropy_loss(self):
    #TODO: auto-generate the below weights list
    n_appear = [22120, 5580, 1624, 2420, 16172] # hand-counted n_appear in train
    weights = [1.0 / n_appear[i] for i in range(len(n_appear))]
    return nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).cuda())

  def _get_sep_pos(self, batch_dialogs):
    sep_pos = []
    for dialog in batch_dialogs:
      sep_pos.append([0] # added for computational efficiency
          + [i for i, x in enumerate(dialog) if x == self.hparams.sep_id])
    return sep_pos # return initial 0 and the positions of [SEP] tokens

  def _2dlist_padding(self, batch_dialogs):
    return torch.nn.utils.rnn.pad_sequence(
      tuple([torch.tensor(dialog).cuda()
          for dialog in batch_dialogs]), batch_first=True)

  def forward(self, batch_dialogs):
    '''
    For examples,
    batch_dialogs[0] = [6006, 16006, 102, 33222, 102]
    '''
    sep_pos = self._get_sep_pos(batch_dialogs)
    batch_dialogs = self._2dlist_padding(batch_dialogs) # list to padded tensor
    segment_tensors = torch.zeros(batch_dialogs.size(), dtype=torch.long).cuda() # dummy tensor
    '''
    For examples,
    sep_pos = [[0, 2, 14, 20], [0, 26, 38, 66, 94, 104]]
        Exceptively, first elements 0 doesn't mean the position of [SEP] token.
    batch_dialogs = tensor([[6006, 16006, 102, 33222, 102, 0, 0, 0], ...])
    '''
    output_layers, _ = self.bert_model(batch_dialogs, segment_tensors)
    output_layers = torch.stack(output_layers, dim=0) # list of tensors to tensor
    last_layers = output_layers[-1] # [n_batchs, n_tokens, hidden_size]

    ### Dialog Embeddings to Utterance Embeddings
    utter_embeddings = torch.zeros([1, self.hparams.hidden_size]).cuda() # initial dummy tensor
    for i_batch, last_layer in enumerate(last_layers):
      for i_utter in range(len(sep_pos[i_batch]) - 1):
        # tokens embeddings of a utterance
        tokens_embedding = last_layers[i_batch,
            sep_pos[i_batch][i_utter] : sep_pos[i_batch][i_utter+1]]
        # mean = convert tokens embeddings to utterance embedding
        utter_embedding = torch.mean(tokens_embedding, dim=0)
        # concat
        utter_embedding = utter_embedding.view(-1, self.hparams.hidden_size)
        utter_embeddings = torch.cat([utter_embeddings, utter_embedding])
    utter_embeddings = utter_embeddings[1:] # remove the initial dummy tensor
    # utter_embeddings.size(): [n_utteraces in a batch, hidden_size]

    ### Linear
    utter_embeddings = self.dropout(self.selu(self.linear_h(utter_embeddings)))
    utter_softmax = self.softmax(self.linear_o(utter_embeddings))
    return utter_softmax # [n_utteraces in a batch, n_class]

  def cal_loss(self, batch_labels, pred_labels):
    '''
    For examples,
    <list> batch_labels[0] = [0, 1, 0, 4]
           batch_labels: [batch_size, each n_utterances]
    <tensor> pred_labels[0] = tensor([0.24, 0.19, 0.19, 0.19, 0.19])
             pred_labels.size(): [n_utterances in a batch, n_class]
    '''
    batch_labels = sum(batch_labels, []) # flatten to 1d list
    target_class = torch.tensor(batch_labels).cuda()

    return self.loss(pred_labels, target_class)

  def count_appear_and_correct(self, batch_labels, pred_labels):
    '''
    class: [neutral, joy, sadness, anger, OOD]
    <list> batch_labels[0] = [0, 1, 0, 4]
           batch_labels: [batch_size, each n_utterances]
    <tensor> pred_labels[0] = tensor([0.24, 0.19, 0.19, 0.19, 0.19])
             pred_labels.size(): [n_utterances in a batch, n_class]    
    '''
    batch_labels = sum(batch_labels, []) # flatten to 1d list
    pred_labels = pred_labels[:, :-1] # trim the OOD column
    max_values, max_indices = torch.max(pred_labels, dim=1)
    max_indices = max_indices.tolist()

    # counting
    n_appear = [0] * (self.hparams.n_class - 1)
    n_correct = [0] * (self.hparams.n_class - 1)
    for target, pred in zip(batch_labels, max_indices):
      if target >= self.hparams.n_class - 1: # exclude OOD class
        continue
      elif target == pred:
        n_appear[target] += 1
        n_correct[target] += 1
      else:
        n_appear[target] += 1

    return [n_appear, n_correct]
  
  def get_uwa_and_wa(self, n_appear, n_correct):
    '''
    For examples,
    n_appear = [5271, 1343, 413, 629]
    n_correct = [5180, 19, 0, 1]
    '''
    # accuracy per a class
    class_accuracy = [0] * (self.hparams.n_class - 1)
    for i in range(self.hparams.n_class - 1):
      if n_appear[i] > 0:
        class_accuracy[i] = n_correct[i] / n_appear[i]
    # un-weighted accuracy(UWA)
    uwa = sum(class_accuracy) / len(class_accuracy)
    # weighted accuracy(WA)
    wa = sum([acc * n_appear[i] / sum(n_appear)
        for i, acc in enumerate(class_accuracy)])

    return uwa, wa

class SomeonesModel_Something_Else(nn.Module):
  def __init__(self, hparams):
    super().__init__()
  def forward(self):
    return None