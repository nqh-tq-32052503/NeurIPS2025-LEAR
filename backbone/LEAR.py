# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import timm
import torchvision.transforms as transforms
import copy

from backbone import MammothBackbone, register_backbone


class LEAR(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Instantiates the layers of the network.
        """

        super(LEAR, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.num_classes = num_classes

        self.model_dim = 768
        self.fc_dim = 768

        self.fcArr = [nn.Linear(self.model_dim, self.fc_dim, device=self.device)]
        self.classifierArr = [nn.Linear(self.model_dim, self.num_classes, device=self.device)]
        self.distributions = []

        model_name_vit = 'vit_base_patch16_224'

        self.c_expert = 0

        self.vitProcess = transforms.Compose(
            [transforms.Resize(224)])

        self.local_vitmodel = timm.create_model(
            model_name_vit,
            pretrained=True,
            num_classes=self.num_classes
        )
        for param in self.local_vitmodel.parameters():
            param.requires_grad = False

        # partially activated
        num_unfrozen_layers = 3
        for block in self.local_vitmodel.blocks[-num_unfrozen_layers:]:
            for param in block.parameters():
                param.requires_grad = True


    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def forward_expert(self, cls_local_features, return_features=False):
        distributions = torch.stack(self.distributions, dim=0) if isinstance(self.distributions, list) else self.distributions
        X_normalized = F.normalize(cls_local_features, p=2, dim=1)  # Shape (M, E)
        Y_normalized = F.normalize(distributions, p=2, dim=1)  # Shape (N, E)
        similarity_matrix = torch.mm(X_normalized, Y_normalized.t())
        return F.softmax(similarity_matrix, dim=1)

    def forward(self, x: torch.Tensor, return_features=False) -> torch.Tensor:
        cls_local_features = self.forward_fusion(x)
        X_normalized = F.normalize(cls_local_features, p=2, dim=1)
        similarity_matrix = torch.mm(X_normalized, X_normalized.t())
        return similarity_matrix

    def forward_fusion(self, x, return_features=False):
        processX = self.vitProcess(x)
        if processX.size(1) == 1:
            processX = processX.expand(-1, 3, -1, -1)
        local_features = self.local_vitmodel.patch_embed(processX)
        local_cls_token = self.local_vitmodel.cls_token.expand(local_features.shape[0], -1, -1)
        local_features = torch.cat((local_cls_token, local_features), dim=1)
        local_features = local_features + self.local_vitmodel.pos_embed
        for block in self.local_vitmodel.blocks:
            local_features = block(local_features)
        local_features = self.local_vitmodel.norm(local_features)
        cls_local_features = local_features[:, 0, :]
        return cls_local_features

    def myprediction(self, x, index):
        with torch.no_grad():
            cls_local_features = self.forward_fusion(x)
            out = self.forward_expert(cls_local_features)
            return out


@register_backbone("lear")
def LEAR_backbone(num_classes):
    return LEAR(num_classes)





