# Kisu Yang, willow4@korea.ac.kr
# Dongyub Lee, jude.lee@kakaocorp.com
# Taesun Whang, hts920928@korea.ac.kr
# Seolhwa Lee, whiteldark@korea.ac.kr


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
from tqdm import trange
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
torch.manual_seed(0)

from hparams import EMOTIONX_MODEL_HPARAMS
from models import EmotionX_Model
from utils import load_data, shuffle_trainset, \
    get_batch, get_time, make_dir, print_params


def main():
  if not torch.cuda.is_available():
    raise NotImplementedError()
  hparams = type('', (object,), EMOTIONX_MODEL_HPARAMS)() # dict to class
  tparams = type('', (object,), {
      'model_name':hparams.model_name+'.'+get_time(),
      'step':0, 'print_per':2000, 'highest_micro_f1':0.})()

  train_dialogs, train_labels = load_data(hparams, hparams.friends_train)
  test_dialogs, test_labels = load_data(hparams, hparams.friends_test)
  em_train_dialogs, em_train_labels = load_data(hparams, hparams.empush_train)
  train_dialogs += em_train_dialogs # merge
  train_labels += em_train_labels
  hparams.n_appear = [sum(train_labels, []).count(i) for i in range(5)]
  n_step = len(train_dialogs) // hparams.batch_size

  model = EmotionX_Model(hparams)
  model.cuda()
  model.train()
  print_params(model)
  optimizer = torch.optim.Adam(model.parameters(), hparams.learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_step)
  writer = SummaryWriter(log_dir=os.path.join(hparams.log_dir, tparams.model_name))

  for i_epoch in range(hparams.n_epoch):
    train_dialogs, train_labels = shuffle_trainset(train_dialogs, train_labels)
    tqdm_range = trange(0, n_step, desc='')
    scheduler.step()
    
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

      if i_step % tparams.print_per == 0:

        # eval
        model.eval()
        n_appear = [0] * (hparams.n_class - 1)
        n_correct = [0] * (hparams.n_class - 1)
        n_positive = [0] * (hparams.n_class - 1)
        for i_test in range(len(test_dialogs) // hparams.batch_size):
          batch_dialogs = get_batch(test_dialogs, hparams.batch_size, i_test)
          batch_labels = get_batch(test_labels, hparams.batch_size, i_test)
          pred_labels = model(batch_dialogs)
          counts = model.count_for_eval(batch_labels, pred_labels)
          n_appear = [x + y for x, y in zip(n_appear, counts[0])]
          n_correct = [x + y for x, y in zip(n_correct, counts[1])]
          n_positive = [x + y for x, y in zip(n_positive, counts[2])]
        uwa, wa = model.get_uwa_and_wa(n_appear, n_correct)
        precision, recall, f1, micro_f1, macro_f1 = model.get_f1_scores(
            n_appear, n_correct, n_positive)

        # save
        if micro_f1 > tparams.highest_micro_f1 - 0.01:
          make_dir(os.path.join(hparams.save_dir, tparams.model_name))
          torch.save({
            'epoch':i_epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss,
          }, os.path.join(hparams.save_dir, tparams.model_name) + '/'
              + tparams.model_name + '.' + str(tparams.step) + '.pt')

        # print
        print('i_epoch: ', i_epoch)
        print('i_total_step: ', tparams.step)
        print('n_true:\t\t\t', n_appear)
        print('n_positive:\t\t', n_positive)
        print('n_true_positive:\t', n_correct)
        print('precision:\t[%.4f, %.4f, %.4f, %.4f]' % (
            precision[0], precision[1], precision[2], precision[3]))
        print('recall:\t\t[%.4f, %.4f, %.4f, %.4f]' % (
            recall[0], recall[1], recall[2], recall[3]))
        print('f1:\t\t[%.4f, %.4f, %.4f, %.4f]' % (
            f1[0], f1[1], f1[2], f1[3]))
        if micro_f1 > tparams.highest_micro_f1:
          tparams.highest_micro_f1 = micro_f1
        print('Micro F1: %.4f (<=%.4f)' % (micro_f1, tparams.highest_micro_f1))
        print()

        # write
        writer.add_scalar(hparams.log_micro_f1, micro_f1, tparams.step)
        writer.add_scalar(hparams.log_wce, loss, tparams.step)
        for name, param in model.named_parameters():
          writer.add_histogram(name, param.clone().cpu().data.numpy(), tparams.step)
        tparams.step += 1
        
if __name__ == '__main__':
  main()