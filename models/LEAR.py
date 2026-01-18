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
import time
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal
from backbone.bilora import BiLoRA_Global, BiLoRA_Local, BiLORA_MoE


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
        self.init_bilora()
        # self.init_indices()

    def extract_distribution(self, processX):
        features = self.net.local_vitmodel.patch_embed(processX)
        cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_token, features), dim=1)
        features = features + self.net.local_vitmodel.pos_embed
        for block in self.net.local_vitmodel.blocks:
            features = block(features)
        features = self.net.local_vitmodel.norm(features)
        class_token = features[:, 0, :]
        return class_token
    
    def end_task(self, dataset) -> None:
        #calculate distribution
        train_loader = dataset.train_loader
        with torch.no_grad():
            train_iter = iter(train_loader)

            pbar = tqdm(train_iter, total=len(train_iter),
                        desc=f"Calculate distribution for task {self.current_task + 1}",
                        disable=False, mininterval=0.5)

            count = 0
            while True:
                try:
                    data = next(train_iter)
                except StopIteration:
                    break

                x = data[0]
                y = data[1]
                x = x.to(self.device)
                y = y.to(self.device)

                processX = self.net.vitProcess(x)
                if processX.size(1) == 1:
                    processX = processX.expand(-1, 3, -1, -1)
                extracted_features = self.extract_distribution(processX)
                self.task_distributions.index_add_(0, y, extracted_features)

                count += 1
                pbar.update()

            pbar.close()
            if self.current_task == 0:
                self.net.distributions = self.task_distributions.clone()
            else:
                old_distributions = self.net.distributions.clone()
                self.net.distributions = torch.concat((old_distributions, self.task_distributions), dim=0)

    def begin_task(self, dataset, threshold=0) -> None:
        self.opt = self.get_optimizer()
        self.task_distributions = torch.zeros((dataset.N_CLASSES, 768), dtype=torch.float32, device=self.device)
        print(f"[INFO] Starting task {self.current_task} with {dataset.N_CLASSES} classes.")
        
        
    def myPrediction(self,x,k):
        with torch.no_grad():
            #Perform the prediction according to the seloeced expert
            out = self.net.myprediction(x,k)
        return out

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        l2_distance = torch.nn.MSELoss()
        # HSIC: Measure of dependence between global and local features
        # Put negative sign because we want to maximize the dependence between them
        self.opt.zero_grad()
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1))
        label_matrix = label_matrix.float()
        input_sim_matrix = self.net(inputs)
        loss_tot = l2_distance(input_sim_matrix, label_matrix)
        loss_vis = loss_tot.item()
        loss_tot.backward()
        self.opt.step()
        return loss_vis

    def cal_expert_dist(self,x):
        pass

    def init_indices(self):
        n_frq = 3000
        dim = 768
        n_tasks = 10
        full_permutation = torch.randperm(dim * dim, generator=torch.Generator().manual_seed(42)).tolist()
        list_indices = [full_permutation[t * n_frq : (t + 1) * n_frq] for t in range(n_tasks)]
        self.list_indices = list_indices
    
    def init_bilora(self):
        print("[INFO] Initializing BiLoRA MoE")
        for i in range(3):
            self.net.local_vitmodel.blocks[9 + i].attn = BiLORA_MoE(dim=768)
            
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