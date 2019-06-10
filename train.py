import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
torch.manual_seed(0)

from hparams import EMOTIONX_MODEL_HPARAMS
from models import EmotionX_Model
from utils import load_data, shuffle_trainset, get_batch, make_dir, print_params


def main():
  if not torch.cuda.is_available():
    raise NotImplementedError()
  hparams = type('', (object,), EMOTIONX_MODEL_HPARAMS)() # dict to class

  # data
  fr_train_dialogs, fr_train_labels = load_data(hparams, hparams.fr_train_path)
  em_train_dialogs, em_train_labels = load_data(hparams, hparams.em_train_path)
  train_dialogs = fr_train_dialogs + em_train_dialogs
  train_labels = fr_train_labels + em_train_labels
  test_dialogs, test_labels = load_data(hparams, hparams.em_test_path)
  assert len(train_dialogs) == len(train_labels)
  assert len(test_dialogs) == len(test_labels)

  # hyper-parameter
  hparams.n_appear = [sum(train_labels, []).count(i) for i in range(5)]
  max_i = len(train_dialogs) // hparams.batch_size
  total_step = 0
  print_per = len(train_dialogs) // 4
  highest_micro_f1 = 0.

  # model
  model = EmotionX_Model(hparams)
  model.cuda()
  model.train()
  print_params(model)
  optimizer = torch.optim.Adam(model.parameters(), hparams.learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_i)
  writer = SummaryWriter(log_dir=hparams.log_dir)

  # train
  for i_epoch in range(hparams.n_epoch):
    train_dialogs, train_labels = shuffle_trainset(train_dialogs, train_labels)
    scheduler.step()

    for i_step in tqdm(range(max_i)):
      batch_dialogs = get_batch(train_dialogs, hparams.batch_size, i_step)
      batch_labels = get_batch(train_labels, hparams.batch_size, i_step)
      optimizer.zero_grad()
      pred_labels = model(batch_dialogs)
      loss = model.cal_loss(batch_labels, pred_labels)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.clip)
      optimizer.step()

      # print
      if i_step % print_per == 0:
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

        print('i_epoch: ', i_epoch)
        print('i_total_step: ', total_step)
        print('n_true:\t\t\t', n_appear)
        print('n_positive:\t\t', n_positive)
        print('n_true_positive:\t', n_correct)
        print('precision:\t[%.4f, %.4f, %.4f, %.4f]' % (
            precision[0], precision[1], precision[2], precision[3]))
        print('recall:\t\t[%.4f, %.4f, %.4f, %.4f]' % (
            recall[0], recall[1], recall[2], recall[3]))
        print('f1:\t\t[%.4f, %.4f, %.4f, %.4f]' % (
            f1[0], f1[1], f1[2], f1[3]))
        if micro_f1 > highest_micro_f1:
          highest_micro_f1 = micro_f1
          friend_high_step = total_step
        print('Micro F1: %.4f (<=%.4f at %d-th total_step)'
            % (micro_f1, highest_micro_f1, friend_high_step))
        print()

        # write
        writer.add_scalar(hparams.log_micro_f1+'fr', micro_f1, total_step)
        writer.add_scalar(hparams.log_wce_loss+'fr', loss, total_step)
        total_step += 1

        model.train()

if __name__ == '__main__':
  main()