# Kisu Yang, willow4@korea.ac.kr
# Dongyub Lee, jude.lee@kakaocorp.com, dongyub63@gmail.com
# Taesun Whang, hts920928@korea.ac.kr
# Seolhwa Lee, whiteldark@korea.ac.kr

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
  tparams = type('', (object,), {
      'model_name':hparams.model_name+'.'+get_time(),
      'step':0, 'high_uwa':0., 'high_wa':0.,
      'high_micro_f1':0., 'high_macro_f1':0.})()

  train_dialogs, train_labels = load_data(hparams, hparams.friends_train)
  test_dialogs, test_labels = load_data(hparams, hparams.friends_test)
  
  model = YksModel_BERT_FC1(hparams)
  # checkpoint = torch.load(PATH)
  # model.load_state_dict(checkpoint['model_state_dict'])
  model.cuda()
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=len(train_dialogs)//hparams.batch_size)
  writer = SummaryWriter(
      log_dir=os.path.join(hparams.log_dir, ))

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
      torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.clip)
      optimizer.step()
      tqdm_range.set_description('weighted_cross_entropy: %.4f' % loss.item())

      if i_step % hparams.print_per == 0:
        model.eval()
        n_appear = [0] * (hparams.n_class - 1)
        n_correct = [0] * (hparams.n_class - 1)
        n_positive = [0] * (hparams.n_class - 1)
        n_step = len(test_dialogs) // hparams.batch_size
        for i_step in range(n_step):
          batch_dialogs = get_batch(test_dialogs, hparams.batch_size, i_step)
          batch_labels = get_batch(test_labels, hparams.batch_size, i_step)
          pred_labels = model(batch_dialogs)
          counts = model.count_for_eval(batch_labels, pred_labels)
          n_appear = [x + y for x, y in zip(n_appear, counts[0])]
          n_correct = [x + y for x, y in zip(n_correct, counts[1])]
          n_positive = [x + y for x, y in zip(n_positive, counts[2])]
        uwa, wa = model.get_uwa_and_wa(n_appear, n_correct)
        precision, recall, f1, micro_f1, macro_f1 = model.get_f1_scores(
            n_appear, n_correct, n_positive)

        # print
        print('i_epoch: ', i_epoch)
        print('n_true:\t\t\t', n_appear)
        print('n_positive:\t\t', n_positive)
        print('n_true_positive:\t', n_correct)
        print('precision:\t[%.4f, %.4f, %.4f, %.4f]' % (
            precision[0], precision[1], precision[2], precision[3]))
        print('recall:\t\t[%.4f, %.4f, %.4f, %.4f]' % (
            recall[0], recall[1], recall[2], recall[3]))
        print('f1:\t\t[%.4f, %.4f, %.4f, %.4f]' % (
            f1[0], f1[1], f1[2], f1[3]))
        if micro_f1 > tparams.high_micro_f1:
          tparams.high_micro_f1 = micro_f1
        print('Micro F1: %.4f (<=%.4f)' % (micro_f1, tparams.high_micro_f1))
        print()
        writer.add_scalar(hparams.micro_f1_log, micro_f1, tparams.step)
        writer.add_scalar(hparams.wce_log, loss, tparams.step)
        tparams.step += 1

        #TODO: checkpoint ensembles
        # torch.save({
        #     'epoch':i_epoch
        #     'model_state_dict':model.state_dict(),
        #     'optimizer_state_dict':optimizer.state_dict(),
        #     'loss':loss,
        #   }, PATH)
        
if __name__ == '__main__':
  main()