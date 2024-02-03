import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


def cosine_similarity(embed1: torch.tensor, embed2: torch.tensor):
    """
        embed1: N * C
        embeds: M * C
        return: N * M
    """
    return torch.matmul(F.normalize(embed1, dim=-1), F.normalize(embed2, dim=-1).t())

def metric_topk_recall_one_v3(query_embeds: torch.tensor, query_labels: torch.tensor, gallery_embeds: torch.tensor, gallery_labels: torch.tensor, topk=(1, 5, 10 )):

    device = "cpu"
    query_embeds, query_labels, gallery_embeds, gallery_labels = query_embeds.to(device), query_labels.to(device), gallery_embeds.to(device), gallery_labels.to(device)

    sim_matrix = 0.5 + 0.5 * cosine_similarity(query_embeds, gallery_embeds)
    topk_scores, topk_indices = torch.topk(sim_matrix, max(topk), dim=1, largest=True, sorted=True)
    topk_labels = gallery_labels[topk_indices]
    recall_list = []

    for k in topk:
        correct_recall = (topk_labels[:,:k] == query_labels.unsqueeze(1)).any(1).float().mean().item()
        recall_list.append(correct_recall)


    return recall_list, topk_scores, topk_indices



def evaluate_recall(dataset, dataloader, model, device, save_tensor_dir=None):

    query_embs = []
    query_labels = []

    gallery_embs = []
    gallery_labels = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            imgs, img_labels, img_types = batch[0], batch[1], batch[2]
            imgs, img_labels = imgs.to(device), img_labels.to(device)

            features = model(imgs).detach().cpu()

            for feature, img_label, img_type in zip(features, img_labels, img_types):
                if 'query' in img_type:
                    query_embs.append(feature)
                    query_labels.append(img_label)
                else:
                    gallery_embs.append(feature)
                    gallery_labels.append(img_label)

        if len(query_embs) > 0:
            query_embs = torch.stack(query_embs, dim=0)
            query_labels = torch.stack(query_labels, dim=0)

            if save_tensor_dir:
                torch.save(query_embs, os.path.join(save_tensor_dir, 'query_embs.pt'))
                torch.save(query_labels, os.path.join(save_tensor_dir, 'query_labels.pt'))

        if len(gallery_embs) > 0:
            gallery_embs = torch.stack(gallery_embs, dim=0)
            gallery_labels = torch.stack(gallery_labels, dim=0)

            if save_tensor_dir:
                torch.save(gallery_embs, os.path.join(save_tensor_dir, 'gallery_embs.pt'))
                torch.save(gallery_labels, os.path.join(save_tensor_dir, 'gallery_labels.pt'))

    recall_list, topk_scores, topk_indices = metric_topk_recall_one_v3(query_embeds=query_embs, query_labels=query_labels, gallery_embeds=gallery_embs, gallery_labels=gallery_labels, topk=(1, 5, 10))
    
    return recall_list
    


class RecallDataset(Dataset):
    def __init__(self, csv_path):
        self.samples = pd.read_csv(csv_path)

        self.img_size = 224

        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
            ToTensorV2()
        ])

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        row = self.samples.iloc[index]

        img_path = row['path']
        img_type = row['type']
        img_label = row['label']

        img = Image.open(img_path)
        img = self.transform(image=np.asarray(img))['image']

        return img, img_label, img_type

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    from model import ViT_B_16

    model = ViT_B_16(embed_size=128, cfg=None)

    # load weights
    tmp_dict = model.state_dict()
    state_dict = torch.load('', map_location=torch.device('cpu'))
    state_dict = {k:v for k,v in state_dict.items() if k in tmp_dict}
    tmp_dict.update(state_dict)
    model.load_state_dict(tmp_dict)

    eval_recall_file = ''
    save_tensor_dir =  ''
    os.makedirs(save_tensor_dir, exist_ok=True)

    eval_recall_dataset = RecallDataset(eval_recall_file)
    print('length of eval set: ', len(eval_recall_dataset))
    eval_recall_dataloader = DataLoader(dataset=eval_recall_dataset, batch_size=512, shuffle=False, num_workers=4)
    recall_list = evaluate_recall(eval_recall_dataset, eval_recall_dataloader, model, device, save_tensor_dir)
    print('R1 R5 R10 on arid:{}'.format(recall_list))
