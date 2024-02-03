import torch
from pytorch_metric_learning import losses, distances, reducers, miners
from pytorch_metric_learning.utils import distributed as pml_dist
import math 

class Loss(object):
    def __init__(self, cfg):

        loss_type = cfg.loss_type
        use_gather = cfg.use_gather
        img_margin = cfg.img_margin
        prod_margin = cfg.prod_margin

        self.use_level = cfg.use_level
        self.loss_lambda = cfg.loss_lambda_pair
        self.use_mae = cfg.use_mae

        mining_func_img = None
        loss_fn_img = None
        mining_func_prod = None
        loss_fn_prod = None

        if loss_type == 'triplet':
            # <------ Triplet loss ------->
            mining_func_img = miners.TripletMarginMiner(margin=img_margin, distance=distances.CosineSimilarity(), type_of_triplets="all")
            loss_fn_img = losses.TripletMarginLoss(margin=img_margin, distance=distances.CosineSimilarity(), reducer=reducers.ThresholdReducer(low=0))
            mining_func_prod = miners.TripletMarginMiner(margin=prod_margin, distance=distances.CosineSimilarity(), type_of_triplets="all")
            loss_fn_prod = losses.TripletMarginLoss(margin=prod_margin, distance=distances.CosineSimilarity(), reducer=reducers.ThresholdReducer(low=0))
        elif loss_type == 'info':
            # <------ infoNCE loss ------->
            loss_fn_img = losses.NTXentLoss()
            loss_fn_prod = losses.NTXentLoss()
        elif loss_type == 'ms':
            # <------ MS loss ------->
            if self.use_level:
                self.mining_func_sprod = miners.MultiSimilarityMiner(epsilon=img_margin/4)
                self.mining_func_dprod = miners.MultiSimilarityMiner(epsilon=img_margin)
            else:
                mining_func_img = miners.MultiSimilarityMiner(epsilon=img_margin)
            loss_fn_img = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
            mining_func_prod = miners.MultiSimilarityMiner(epsilon=prod_margin)
            loss_fn_prod = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

        if use_gather:
            if self.use_level:
                self.mining_func_sprod = pml_dist.DistributedMinerWrapper(miner=self.mining_func_sprod, efficient=True)
                self.mining_func_dprod = pml_dist.DistributedMinerWrapper(miner=self.mining_func_dprod, efficient=True)
            if mining_func_img:
                mining_func_img = pml_dist.DistributedMinerWrapper(miner=mining_func_img, efficient=True)
            if loss_fn_img:
                loss_fn_img = pml_dist.DistributedLossWrapper(loss=loss_fn_img, efficient=True)
            if mining_func_prod:
                mining_func_prod = pml_dist.DistributedMinerWrapper(miner=mining_func_prod, efficient=True)
            if loss_fn_prod:
                loss_fn_prod = pml_dist.DistributedLossWrapper(loss=loss_fn_prod, efficient=True)
        
        self.mining_func_img = mining_func_img
        self.loss_fn_img = loss_fn_img
        self.mining_func_prod = mining_func_prod
        self.loss_fn_prod = loss_fn_prod

        if self.use_mae:
            loss_fn_img = losses.VICRegLoss()
            self.loss_fn_img = loss_fn_img

    def cal_loss(self, embedding1, label_1, embedding2, label_2, same_prod_method='all', use_neg_filter=False, use_pos_filter=False):
        
        if not self.use_mae:
            indices_tuple_img = None
            if self.use_level:
                label_2_all = self.gather_labels(label_2)
                indices_tuple_sprod = self.mining_func_sprod(embedding1, label_1)
                indices_tuple_sprod = self.get_sprod_neg_pairs(indices_tuple_sprod, label_2, label_2_all)
                loss_sprod = self.loss_fn_img(embedding1, label_1, indices_tuple_sprod)
                indices_tuple_dprod = self.mining_func_dprod(embedding1, label_1)
                indices_tuple_dprod = self.get_dprod_neg_pairs(indices_tuple_dprod, label_2, label_2_all)
                loss_dprod = self.loss_fn_img(embedding1, label_1, indices_tuple_dprod)
                loss = loss_sprod*0.5 + loss_dprod*0.5
            else:
                if self.mining_func_img:
                    indices_tuple_img = self.mining_func_img(embedding1, label_1)
                    if use_neg_filter:
                        # filtering the neg pairs by removing the same product samples
                        label_2_all = self.gather_labels(label_2)
                        indices_tuple_img = self.filter_neg_pairs(indices_tuple_img, label_2, label_2_all)
                loss = self.loss_fn_img(embedding1, label_1, indices_tuple_img)
        else:
            embedding_x = embedding1[::2]
            embedding_y = embedding1[1::2]
            loss = self.loss_fn_img(embedding_x, embedding_y)

        if same_prod_method == 'all':
            indices_tuple_prod = None
            if self.mining_func_prod:
                indices_tuple_prod = self.mining_func_prod(embedding2, label_2)
                if use_pos_filter:
                    # filtering the pos pairs by removing the same img samples
                    label_1_all = self.gather_labels(label_1)
                    indices_tuple_prod = self.filter_pos_pairs(indices_tuple_prod, label_1, label_1_all)
            loss_prod = self.loss_fn_prod(embedding2, label_2, indices_tuple_prod)
            loss = loss*self.loss_lambda + loss_prod*(1-self.loss_lambda)
        elif same_prod_method == 'raw':
            embedding2 = embedding2[::2]
            label_2 = label_2[::2]
            indices_tuple_prod = None
            if self.mining_func_prod:
                indices_tuple_prod = self.mining_func_prod(embedding2, label_2)
            loss_prod = self.loss_fn_prod(embedding2, label_2, indices_tuple_prod)
            loss = loss*self.loss_lambda + loss_prod*(1-self.loss_lambda)
        
        return loss

    def cal_loss_selsup(self, embedding, label):
        indices_tuple_img = None
        if self.mining_func_img:
            indices_tuple_img = self.mining_func_img(embedding, label)
        loss = self.loss_fn_img(embedding, label, indices_tuple_img)
        return loss

    def gather_labels(self, labels):
        dist_ref_labels = self.all_gather(labels)
        all_labels = torch.cat([labels, dist_ref_labels], dim=0)
        return all_labels

    def all_gather(self, x):
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            rank = torch.distributed.get_rank()
            x_list = [torch.ones_like(x) for _ in range(world_size)]
            torch.distributed.all_gather(x_list, x.contiguous())
            # remove curr rank
            x_list.pop(rank)
            return torch.cat(x_list, dim=0)
        return None

    def filter_neg_pairs(self, indices_tuple, label_batch, label_batch_all):
        filrered_neg_pair_idxs = (label_batch[indices_tuple[2]] != label_batch_all[indices_tuple[3]]).view(-1)
        return (indices_tuple[0],
                indices_tuple[1],
                indices_tuple[2][filrered_neg_pair_idxs],
                indices_tuple[3][filrered_neg_pair_idxs])
    
    def filter_pos_pairs(self, indices_tuple, label_batch, label_batch_all):
        filrered_pos_pair_idxs = (label_batch[indices_tuple[0]] != label_batch_all[indices_tuple[1]]).view(-1)
        return (indices_tuple[0][filrered_pos_pair_idxs],
                indices_tuple[1][filrered_pos_pair_idxs],
                indices_tuple[2],
                indices_tuple[3])
    
    def get_sprod_neg_pairs(self, indices_tuple, label_batch, label_batch_all):
        filrered_pair_idxs = (label_batch[indices_tuple[2]] == label_batch_all[indices_tuple[3]]).view(-1)
        return (indices_tuple[0],
                indices_tuple[1],
                indices_tuple[2][filrered_pair_idxs],
                indices_tuple[3][filrered_pair_idxs])
    
    def get_dprod_neg_pairs(self, indices_tuple, label_batch, label_batch_all):
        filrered_pair_idxs = (label_batch[indices_tuple[2]] != label_batch_all[indices_tuple[3]]).view(-1)
        return (indices_tuple[0],
                indices_tuple[1],
                indices_tuple[2][filrered_pair_idxs],
                indices_tuple[3][filrered_pair_idxs])

class MetricLoss(object):
    def __init__(self, cfg):

        loss_type = cfg.loss_type
        use_gather = cfg.use_gather
        img_margin = cfg.img_margin

        self.loss_lambda = cfg.loss_lambda_pair
        self.use_mae = cfg.use_mae

        mining_func_img = None
        loss_fn_img = None

        if loss_type == 'triplet':
            # <------ Triplet loss ------->
            mining_func_img = miners.TripletMarginMiner(margin=img_margin, distance=distances.CosineSimilarity(), type_of_triplets="all")
            loss_fn_img = losses.TripletMarginLoss(margin=img_margin, distance=distances.CosineSimilarity(), reducer=reducers.ThresholdReducer(low=0))
        elif loss_type == 'info':
            # <------ infoNCE loss ------->
            loss_fn_img = losses.NTXentLoss()
        elif loss_type == 'ms':
            # <------ MS loss ------->
            mining_func_img = miners.MultiSimilarityMiner(epsilon=img_margin)
            loss_fn_img = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        elif loss_type == 'circle':
            loss_fn_img = losses.CircleLoss(m=0.25, gamma=80)
            
        if use_gather:
            if mining_func_img:
                mining_func_img = pml_dist.DistributedMinerWrapper(miner=mining_func_img, efficient=True)
            if loss_fn_img:
                loss_fn_img = pml_dist.DistributedLossWrapper(loss=loss_fn_img, efficient=True)
        
        self.mining_func_img = mining_func_img
        self.loss_fn_img = loss_fn_img

        if self.use_mae:
            loss_fn_img = losses.VICRegLoss()
            self.loss_fn_img = loss_fn_img

    def cal_loss(self, embedding1, label_1):
        
        if not self.use_mae:
            indices_tuple_img = None
            if self.mining_func_img:
                indices_tuple_img = self.mining_func_img(embedding1, label_1)
            loss = self.loss_fn_img(embedding1, label_1, indices_tuple_img)
        else:
            embedding_x = embedding1[::2]
            embedding_y = embedding1[1::2]
            loss = self.loss_fn_img(embedding_x, embedding_y)
        
        return loss

    def cal_loss_selsup(self, embedding, label):
        indices_tuple_img = None
        if self.mining_func_img:
            indices_tuple_img = self.mining_func_img(embedding, label)
        loss = self.loss_fn_img(embedding, label, indices_tuple_img)
        return loss

    def gather_labels(self, labels):
        dist_ref_labels = self.all_gather(labels)
        all_labels = torch.cat([labels, dist_ref_labels], dim=0)
        return all_labels

    def all_gather(self, x):
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            rank = torch.distributed.get_rank()
            x_list = [torch.ones_like(x) for _ in range(world_size)]
            torch.distributed.all_gather(x_list, x.contiguous())
            # remove curr rank
            x_list.pop(rank)
            return torch.cat(x_list, dim=0)
        return None

    def filter_neg_pairs(self, indices_tuple, label_batch, label_batch_all):
        filrered_neg_pair_idxs = (label_batch[indices_tuple[2]] != label_batch_all[indices_tuple[3]]).view(-1)
        return (indices_tuple[0],
                indices_tuple[1],
                indices_tuple[2][filrered_neg_pair_idxs],
                indices_tuple[3][filrered_neg_pair_idxs])
    
    def filter_pos_pairs(self, indices_tuple, label_batch, label_batch_all):
        filrered_pos_pair_idxs = (label_batch[indices_tuple[0]] != label_batch_all[indices_tuple[1]]).view(-1)
        return (indices_tuple[0][filrered_pos_pair_idxs],
                indices_tuple[1][filrered_pos_pair_idxs],
                indices_tuple[2],
                indices_tuple[3])
    
    def get_sprod_neg_pairs(self, indices_tuple, label_batch, label_batch_all):
        filrered_pair_idxs = (label_batch[indices_tuple[2]] == label_batch_all[indices_tuple[3]]).view(-1)
        return (indices_tuple[0],
                indices_tuple[1],
                indices_tuple[2][filrered_pair_idxs],
                indices_tuple[3][filrered_pair_idxs])
    
    def get_dprod_neg_pairs(self, indices_tuple, label_batch, label_batch_all):
        filrered_pair_idxs = (label_batch[indices_tuple[2]] != label_batch_all[indices_tuple[3]]).view(-1)
        return (indices_tuple[0],
                indices_tuple[1],
                indices_tuple[2][filrered_pair_idxs],
                indices_tuple[3][filrered_pair_idxs])
