# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import os
from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from backbone.utils.hsic import hsic
import torch.nn.functional as F
import random
import copy
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal
from backbone.bilora import BiLORA_MoE


class LEAR(ContinualModel):
    NAME = 'LEAR'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(LEAR, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.train_loader_size = None
        self.iter = 0
        self.use_bilora = True if args.use_bilora == 1 else False
        print("Use BiLORA: ", self.use_bilora)
        self.apply_bilora_for = args.apply_bilora_for
        self.bilora_mode = args.bilora_mode
        print("Apply BiLORA for: ", self.apply_bilora_for)
        if self.use_bilora:
            self.init_bilora()

    def end_task(self, dataset) -> None:
        self.net.global_vitmodel[9].end_task()

    def begin_task(self, dataset, threshold=0) -> None:
        self.net.CreateNewExper(-1, dataset.N_CLASSES)
        self.opt = self.get_optimizer()

    def myPrediction(self,x,k):
        if self._current_task > 0:
            self.apply_bilora(selected_index=k)
        with torch.no_grad():
            #Perform the prediction according to the seloeced expert
            out = self.net.myprediction(x,k)
        return out

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        l2_distance = torch.nn.MSELoss()
        # HSIC: Measure of dependence between global and local features
        # Put negative sign because we want to maximize the dependence between them
        self.opt.zero_grad()
        if len(self.net.fcArr) > 1:
            # NOTE: Task tiếp theo
            outputs, Freezed_global_features, Freezed_local_features, global_features, local_features = self.net(inputs, return_features=True)
            loss_kd = kl_loss(local_features, Freezed_local_features)
            loss_mi = l2_distance(global_features, Freezed_global_features) #Directly calculate the L2 distance between features is more efficient than calculate MI between prediction, and it's also effective
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_kd + loss_hsic + loss_mi
            loss_vis = [loss_ce.item(), loss_kd.item(), loss_hsic.item(), loss_mi.item()]
        else:
            # NOTE: Task đầu tiên
            outputs, global_features, local_features = self.net(inputs)
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_hsic
            loss_vis = [loss_ce.item(), loss_hsic.item()]

        loss_tot.backward()


        self.opt.step()

        return loss_vis

    def cal_expert_dist(self,x):
        processX = self.net.vitProcess(x)
        if processX.size(1) == 1:
            processX = processX.expand(-1, 3, -1, -1)

        features = self.net.global_vitmodel.patch_embed(processX)
        cls_token = self.net.global_vitmodel.cls_token.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_token, features), dim=1)
        features = features + self.net.global_vitmodel.pos_embed

        # forward pass till -3
        for block in self.net.global_vitmodel.blocks[:-3]:
            features = block(features)

        # features = self.net.Forever_freezed_blocks(features)

        # features = self.net.local_vitmodel.norm(features)
        f_norm = self.net.global_vitmodel.norm
        class_token = features[:, 0, :]
        distances = []
        for task in range(len(self.net.fcArr)):
            t_features = features.clone()
            self.net.global_vitmodel[9].attn.evaluate(task)
            for block in self.net.global_vitmodel.blocks[-3:]:
                t_features = block(t_features)
            t_features = f_norm(t_features)
            t_cls_token = t_features[:, 0, :]
            distances.append(kl_loss(t_cls_token, class_token))
        
        return distances
            

    def init_bilora(self):
        for i in range(3):
            self.net.local_vitmodel.blocks[9 + i].attn = BiLORA_MoE(dim=768, use_dom_enc_vect=False)
        for i in range(2):
            self.net.global_vitmodel.blocks[10 + i].attn = BiLORA_MoE(dim=768, use_dom_enc_vect=False)
        self.net.global_vitmodel.blocks[9].attn = BiLORA_MoE(dim=768, use_dom_enc_vect=True)

def kl_loss(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

    student_prob = (student_feat + 1) / 2
    teacher_prob = (teacher_feat.detach() + 1) / 2

    loss_kld = F.kl_div(
        torch.log(student_prob + 1e-10),
        teacher_prob,
        reduction='batchmean'
    )
    return loss_kld