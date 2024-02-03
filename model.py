# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import timm

class ResNet50(nn.Module):
    def __init__(self, embed_size):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, embed_size)

    def forward(self, input_imgs):
        output = self.model(input_imgs)
        return output

class ViT_B_16(nn.Module):
    def __init__(self, embed_size, cfg=None):
        super(ViT_B_16, self).__init__()
        self.use_proxy = cfg.model.use_proxy if cfg else None
        self.use_2proxy = cfg.model.use_2proxy if cfg else None
        self.same_prod_method = cfg.train.same_prod_method if cfg else None
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=embed_size)

        if self.use_proxy:
            self.proxy = self._build_mlp(2, embed_size, 128, 128)
        if self.use_2proxy:
            self.proxy1 = self._build_mlp(2, embed_size, 512, 1024)
            self.proxy2 = self._build_mlp(2, embed_size, 512, 1024)
            
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))

        return nn.Sequential(*mlp)

    def forward(self, input_imgs, labels=None):
        output = self.backbone(input_imgs)
        if self.training:
            if self.use_proxy:
                output = self.proxy(output)
                return output
            if self.use_2proxy:
                output1 = self.proxy1(output)
                output2 = self.proxy2(output)
                return (output1, output2)
        output = F.normalize(output)
        return output

class Swin_B_4_7(nn.Module):
    def __init__(self, embed_size, use_mlp=False, use_proxy=False, use_2proxy=False):
        super(Swin_B_4_7, self).__init__()
        self.use_proxy = use_proxy
        self.use_2proxy = use_2proxy
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.backbone.head = torch.nn.Linear(1024, embed_size)

        if use_mlp:
            hidden_dim = self.backbone.head.weight.shape[1]
            del self.backbone.head # remove original fc layer
            self.backbone.head = self._build_mlp(3, hidden_dim, 4096, embed_size)
        if use_proxy:
            self.proxy = self._build_mlp(2, embed_size, 512, 1024)
        if use_2proxy:
            self.proxy1 = self._build_mlp(2, embed_size, 512, 1024)
            self.proxy2 = self._build_mlp(2, embed_size, 512, 1024)
            
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))

        return nn.Sequential(*mlp)

    def forward(self, input_imgs, labels=None):
        output = self.backbone(input_imgs)
        if self.training:
            if self.use_proxy:
                output = self.proxy(output)
                return output
            if self.use_2proxy:
                output1 = self.proxy1(output)
                output2 = self.proxy2(output)
                return (output1, output2)

        output = output.mean(dim=[1, 2])
        output = F.normalize(output)

        return output

if __name__ == "__main__":
    model = ResNet50(embed_size=128)
    input_imgs = torch.rand(size=(4, 3, 224, 224))
    labels = torch.zeros(size=(4, 1)).long()
    print(model)
    model.train()
    feats = model(input_imgs )
    print(feats.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    