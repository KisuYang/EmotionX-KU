# Kisu Yang, willow4@korea.ac.kr
# Dongyub Lee, jude.lee@kakaocorp.com, dongyub63@gmail.com
# Taesun Whang, hts920928@korea.ac.kr
# SeolHwa Lee, whiteldark@korea.ac.kr

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datetime import datetime
from tqdm import tqdm
from tqdm import trange
import torch
import torch.nn as nn

from hparams import YKSMODEL_BERT_FC1_HPARAMS
from utils import load_data, shuffle_trainset, get_batch, get_time
from models import YksModel_BERT_FC1
from tensorboardX import SummaryWriter

def main():
  if not torch.cuda.is_available():
    raise NotImplementedError
  hparams = type('', (object,), YKSMODEL_BERT_FC1_HPARAMS)() # dict to class
  print_params = type('', (object,), {'step':0, 'high_uwa':0., 'high_wa':0.})()
  print_params.cur_model_id = hparams.model_name+'.'+get_time()

  train_dialogs, train_labels = load_data(hparams, hparams.friends_train)
  test_dialogs, test_labels = load_data(hparams, hparams.friends_test)
  
  model = YksModel_BERT_FC1(hparams)
  #if True: model.load_state_dict(torch.load(PATH))
  model.cuda()
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=len(train_dialogs)//hparams.batch_size, )
  writer = SummaryWriter(
      log_dir=os.path.join(hparams.log_dir, print_params.cur_model_id))

  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name, param.size()) # print grad params in model

  for i_epoch in range(hparams.n_epoch):
    scheduler.step()
    train_dialogs, train_labels = shuffle_trainset(train_dialogs, train_labels)
    tqdm_range = trange(0, len(train_dialogs)//hparams.batch_size, desc='')
    for i_step in tqdm_range:
      model.train()
      batch_dialogs = get_batch(train_dialogs, hparams.batch_size, i_step)
      batch_labels = get_batch(train_labels, hparams.batch_size, i_step)
      optimizer.zero_grad()
      pred_labels = model(batch_dialogs)
      loss = model.cal_loss(batch_labels, pred_labels)
      loss.backward()
      optimizer.step()
      tqdm_range.set_description('weighted_cross_entropy: %.4f' % loss.item())

      if i_step % 40 == 0:
        model.eval()
        n_appear = [0] * (hparams.n_class - 1)
        n_correct = [0] * (hparams.n_class - 1)
        n_step = len(test_dialogs) // hparams.batch_size
        for i_step in range(n_step):
          batch_dialogs = get_batch(test_dialogs, hparams.batch_size, i_step)
          batch_labels = get_batch(test_labels, hparams.batch_size, i_step)
          pred_labels = model(batch_dialogs)
          count = model.count_appear_and_correct(batch_labels, pred_labels)
          n_appear = [x + y for x, y in zip(n_appear, count[0])]
          n_correct = [x + y for x, y in zip(n_correct, count[1])]
        uwa, wa = model.get_uwa_and_wa(n_appear, n_correct)

        # print
        print('i_epoch: ', i_epoch)
        print('n_appear:\t', n_appear)
        print('n_correct:\t', n_correct)
        each_accuracy = [x / X * 100 for x, X in zip(n_correct, n_appear)]
        print('each_accuracy:\t[%.2f(%%), %.2f(%%), %.2f(%%), %.2f(%%)]' %
          (each_accuracy[0], each_accuracy[1], each_accuracy[2], each_accuracy[3]))
        if uwa > print_params.high_uwa:
          print_params.high_uwa = uwa          
        if wa > print_params.high_wa:
          print_params.high_wa = wa
          # if True: torch.save(model.state_dict(),PATH)
        print('Highest UWA: %.2f(%%), Highest WA: %.2f(%%)'
            % (print_params.high_uwa*100, print_params.high_wa*100))
        print('Current UWA: %.2f(%%), Current WA: %.2f(%%)' % (uwa*100, wa*100))
        print()
        writer.add_scalar(hparams.wce_log, loss, print_params.step)
        writer.add_scalar(hparams.uwa_log, uwa, print_params.step)
        writer.add_scalar(hparams.wa_log, wa, print_params.step)
        print_params.step += 1

if __name__ == '__main__':
  main()