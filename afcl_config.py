# -*- coding: utf-8 -*-
from box import Box

cfg = Box()
cfg.task_name = 'img_metric'

# data
cfg.data = Box()
cfg.data.train_dir = '' # replace with train set dir
cfg.data.train_file = '' # replace with train set dir

cfg.data.valid_file = '' # replace with your final annotation-file's path

cfg.data.img_size = 224
cfg.data.t_crop = 0.8
cfg.data.c_crop = 0.7
cfg.data.random = True  

# model
cfg.model = Box()
cfg.model.output_embed_dim = 128
cfg.model.backbone = 'vit'
cfg.model.pretrained_path = ''


# train
cfg.train = Box()
# cfg.train.use_board = True
cfg.train.use_board = False
cfg.train.do_train = True
cfg.train.do_eval = False
cfg.train.do_trace = False
cfg.train.init_lr = 0.002
cfg.train.momentum = 0.9
cfg.train.weight_decay = 1e-4
cfg.train.batch_size = 128
cfg.train.epochs = 40
cfg.train.max_grad_norm = 1.0
cfg.train.output_dir_root = ''
cfg.train.gradient_accumulation_steps = 5
cfg.train.save_steps = 2000 * cfg.train.gradient_accumulation_steps
cfg.train.logging_steps = 20 * cfg.train.gradient_accumulation_steps
cfg.train.warmup_ratio = 1e-4
cfg.train.same_prod_neg_filter = False
cfg.train.loss_lambda = 0.75
cfg.train.loss_type = 'circle'
cfg.train.use_gather = True
cfg.train.img_margin = 0.2
cfg.train.prod_margin = 0.1
