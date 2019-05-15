from collections import defaultdict


EMOTIONX_MODEL_HPARAMS = defaultdict(
  description='base',
  model_name='bert_max_selu',

  friends_train='./data/friend4000.joblib',
  friends_test='./data/friend800.joblib', # CAUTION: duplicated with trainset 
  empush_train='./data/empush4000.joblib',  
  save_dir='/mnt/raid5/yks/EmotionX/saves',
  log_dir='/mnt/raid5/yks/EmotionX/logs',
  log_micro_f1='micro_f1',
  log_wce='train_loss',

  # bert
  bert_type='bert-base-uncased',
  max_input_len=512,
  cls_token='[CLS]',
  sep_token='[SEP]',
  pad_token='[PAD]',
  cls_id=101,
  sep_id=102,
  pad_id=0,

  # classifier
  hidden_size=768,
  inter_hidden_size=384,
  n_class=4+1, # neutral, joy, sadness, anger + OOD

  n_epoch=30,
  batch_size=1,
  learning_rate=2e-5,
  dropout=0.1,
  clip=5,
)