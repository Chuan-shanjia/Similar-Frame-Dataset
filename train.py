# -*- coding: utf-8 -*-
import os
import time
import json
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
from afcl_config import cfg
from collections import defaultdict
from model import ResNet50, Swin_B_4_7, ViT_B_16
from dataset_sfd import SFDataset, collate_fn
from evalrecall_dataset import RecallDataset, evaluate_recall

from timm.scheduler import MultiStepLRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.losses import MetricLoss

def seed_everything(seed=2048):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    batch_size = cfg.train.batch_size
    train_epochs = cfg.train.epochs
    init_lr = cfg.train.init_lr
    warmup_ratio = cfg.train.warmup_ratio
    momentum = cfg.train.momentum
    weight_decay = cfg.train.weight_decay
    max_grad_norm = cfg.train.max_grad_norm
    save_steps = cfg.train.save_steps
    logging_steps = cfg.train.logging_steps
    gradient_accumulation_steps = cfg.train.gradient_accumulation_steps
    output_embed_dim = cfg.model.output_embed_dim
    backbone = cfg.model.backbone
    ps = cfg.train.ps
    output_dir = "{}_{}_{}".format(cfg.train.output_dir_root, backbone, ps)
    log_output_dir = output_dir+'/logs'
    model_output_dir = output_dir+'/models'

    if args.local_rank == 0:
        exp_config = "backbone{}_bs{}_lr{}_{}".format(
            backbone,
            batch_size,
            init_lr,
            ps)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(log_output_dir):
            os.mkdir(log_output_dir)
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)

    dist.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()
    
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device("cuda:{}".format(device_ids[args.local_rank]))
    torch.cuda.set_device(device)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_output_name = log_output_dir + f'/{args.local_rank}.log'
    f_h = logging.FileHandler(logger_output_name)
    c_h = logging.StreamHandler()
    logger.addHandler(f_h)
    logger.addHandler(c_h)
    logger.warning(f"Process device: {device}, local_rank: {args.local_rank}")
    
    seed_everything()

    # dataset
    train_dataset = SFDataset(cfg=cfg.data)
    train_sampler = DistributedSampler(train_dataset, drop_last=True, shuffle=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

    eval_recall_dataset = RecallDataset(cfg.data.valid_file)
    eval_recall_dataloader = DataLoader(dataset=eval_recall_dataset, batch_size=512, shuffle=False, num_workers=4)

    logging_steps = logging_steps if logging_steps != -1 else len(train_dataloader)
    save_steps = save_steps if save_steps != -1 else len(train_dataloader)

    # Load pretrained model and tokenizer
    model_dict = {
        'res':ResNet50,
        'vit':ViT_B_16,
        'swin':Swin_B_4_7,
    }

    model = model_dict[backbone](embed_size=output_embed_dim)

    # init loss and optimizer
    loss_fn = MetricLoss(cfg.train)
    if cfg.train.adam:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader), eta_min=0, last_epoch=-1)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = MultiStepLRScheduler(optimizer, decay_t=[3, 6], decay_rate=0.1, warmup_t=1, warmup_lr_init=warmup_ratio)

    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[device_ids[args.local_rank]], output_device=device_ids[args.local_rank])

    # log cfg
    logger.warning(f"TASK CONFIG: \n {cfg}")
    logger.warning(f"train_dataloader length: {len(train_dataloader)}, train_dataset length: {len(train_dataset)}")

    scaler = GradScaler()
    # Train
    num_steps = len(train_dataloader)
    model.zero_grad()
    
    if rank == 0:
        step_loss = []
        epoch_loss = []
    train_st = time.time()
    step_st = time.time()
    
    for epoch in range(train_epochs):
        train_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_img_batch, label_batch = batch[0], batch[1]
            optimizer.zero_grad()
            
            if cfg.train.use_amp:   # True
                with autocast():
                    output_embeds = model(input_img_batch)
                    loss = loss_fn.cal_loss(output_embeds, label_batch)
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
                        
                if rank == 0:
                    step_loss.append(loss.item())
                    epoch_loss.append(loss.item())
                
                if (step + 1) % gradient_accumulation_steps == 0:    
                    scaler.unscale_(optimizer)                          
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output_embeds = model(input_img_batch)
                loss = loss_fn.cal_loss(output_embeds, label_batch)
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward() 
                
                if rank == 0:
                    step_loss.append(loss.item())
                    epoch_loss.append(loss.item())
                
                if (step + 1) % gradient_accumulation_steps == 0:                         
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            if rank == 0 and (step + 1) % logging_steps == 0:
                # Log metrics
                logger.warning(
                    "Local Rank: {0}, Train, Epoch_{1}_Step_{2}/{3}, TrainLoss: {4}, Learning Rate: {5}, Cost Time: {6}".format(
                                                                                            args.local_rank, epoch,
                                                                                            step, num_steps,
                                                                                            sum(step_loss)/len(step_loss),
                                                                                            optimizer.state_dict()['param_groups'][0]['lr'],
                                                                                            time.time()-step_st))
                step_st = time.time()
                step_loss = []

        if rank == 0:
            logger.warning("Epoch Loss: {}".format(sum(epoch_loss)/len(epoch_loss)))
        
            # Save model checkpoint
            eval_st = time.time()
            model.eval()

            # eval recall online
            recall_list = evaluate_recall(eval_recall_dataset, eval_recall_dataloader, model, device)
            logger.warning("R1, R5, R10 on arid : {} ".format(recall_list))

            logger.warning("Epoch Loss: {}".format(sum(epoch_loss)/len(epoch_loss)))
            
            epoch_loss = []
            checkpoint_dir = os.path.join(model_output_dir, "checkpoint-{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            logger.warning("Saving model checkpoint to {}".format(checkpoint_dir))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(),
                        os.path.join(checkpoint_dir, f"saved_model_{epoch}.pth"))
            logger.warning("Saving optimizer, scheduler and model states to %s", checkpoint_dir)

        scheduler.step(epoch+1)  # Update learning each epoch
        logger.info("\n")
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Metric DDP Training")
    parser.add_argument('--device_ids', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--init_lr', type=float, default=0.002)
    parser.add_argument('--loss_lambda_pair', type=float, default=0.75)
    parser.add_argument('--loss_type', type=str, default='ms')
    parser.add_argument('--use_mae', action='store_true')
    parser.add_argument('--img_margin', type=float, default=0.2)
    parser.add_argument('--prod_margin', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--backbone', type=str, default='vit')
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_gather', action='store_true')
    parser.add_argument('--use_mlp', action='store_true')
    parser.add_argument('--use_proxy', action='store_true')
    parser.add_argument('--use_2proxy', action='store_true')
    parser.add_argument('--use_level', action='store_true')
    parser.add_argument('--use_neg_filter', action='store_true')
    parser.add_argument('--use_pos_filter', action='store_true')
    parser.add_argument('--same_prod_method', type=str, default='all')
    parser.add_argument('--data_src', type=str, default='ecom')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--ps', type=str, default='')

    args = parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    cfg.train.use_amp = args.use_amp
    cfg.train.init_lr = args.init_lr
    cfg.train.loss_lambda_pair = args.loss_lambda_pair
    cfg.train.loss_type = args.loss_type
    cfg.train.use_mae = args.use_mae
    cfg.train.use_gather = args.use_gather
    cfg.train.img_margin = args.img_margin
    cfg.train.prod_margin = args.prod_margin
    cfg.train.use_neg_filter = args.use_neg_filter
    cfg.train.use_pos_filter = args.use_pos_filter
    cfg.train.same_prod_method = args.same_prod_method
    cfg.train.use_level = args.use_level
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epoch
    cfg.train.adam = args.adam
    cfg.model.output_embed_dim = args.embed_dim
    cfg.model.backbone = args.backbone
    cfg.model.pretrained_path = args.pretrained_path
    cfg.model.use_mlp = args.use_mlp
    cfg.model.use_proxy = args.use_proxy
    cfg.model.use_2proxy = args.use_2proxy
    cfg.data.img_size = args.img_size
    cfg.data.data_src = args.data_src
    cfg.data.random = args.random
    cfg.train.ps = 'lr_' + str(args.init_lr) + '_' + args.ps

    cfg.train.load_online_model = args.load_online_model
    main(args)
