from collections import defaultdict

BASE_HPARAMS = defaultdict(
  description='base',
  friends_train='./data/friends_train_dialogs',
  friends_test='./data/friends_test_dialogs',
  
  save_dir='/mnt/raid5/yks/EmotionX/saves',
  saving=False,
  log_dir='/mnt/raid5/yks/EmotionX/logs',
  model_name='',
  wce_log='train_loss',
  uwa_log='test_UWA',
  wa_log='test_WA',

  bert_type = 'bert-base-cased',
  cls_token='[CLS]',
  sep_token='[SEP]',
  pad_token='[PAD]',
  sep_id=102,
  pad_id=0,

  hidden_size=768,
  n_class=4+1, # neutral, joy, sadness, anger + OOD

  n_epoch=100,
  batch_size=4,
  learning_rate=1e-4,
  negative_slope=0.2, # leaky_relu
)

YKSMODEL_BERT_FC1_HPARAMS = BASE_HPARAMS.copy()
YKSMODEL_BERT_FC1_HPARAMS.update(
  model_name='bert_fc1',
  saving=True,
)