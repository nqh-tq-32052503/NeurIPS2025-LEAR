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
from backbone.bilora import BiLoRA_Global, BiLoRA_Local


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
        self.init_indices()

    def extract_distribution(self, processX, mode="global"):
        if mode == "global":
            features = self.net.global_vitmodel.patch_embed(processX)
            cls_token = self.net.global_vitmodel.cls_token.expand(features.shape[0], -1, -1)
            features = torch.cat((cls_token, features), dim=1)
            features = features + self.net.global_vitmodel.pos_embed
            for block in self.net.global_vitmodel.blocks:
                features = block(features)
            features = self.net.global_vitmodel.norm(features)
            class_token = features[:, 0, :]
            return class_token
        else:
            features = self.net.local_vitmodel.patch_embed(processX)
            cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
            features = torch.cat((cls_token, features), dim=1)
            features = features + self.net.local_vitmodel.pos_embed
            for block in self.net.local_vitmodel.blocks:
                features = block(features)
            features = self.net.local_vitmodel.norm(features)
            class_token = features[:, 0, :]
            return self.net.fcArr[self.current_task](class_token)
    def end_task(self, dataset) -> None:
        #calculate distribution
        train_loader = dataset.train_loader
        num_choose = 100
        with torch.no_grad():
            train_iter = iter(train_loader)

            pbar = tqdm(train_iter, total=num_choose,
                        desc=f"Calculate distribution for task {self.current_task + 1}",
                        disable=False, mininterval=0.5)

            fc_features_list = []

            count = 0
            while count < num_choose:
                try:
                    data = next(train_iter)
                except StopIteration:
                    break

                x = data[0]
                x = x.to(self.device)

                processX = self.net.vitProcess(x)
                if processX.size(1) == 1:
                    processX = processX.expand(-1, 3, -1, -1)
                extracted_features = self.extract_distribution(processX, mode="global")
                fc_features_list.append(extracted_features)

                count += 1
                pbar.update()

            pbar.close()
            fc_features = torch.cat(fc_features_list, dim=0)  # [num*b,fc_size]
            mu = torch.mean(fc_features, dim=0)
            sigma = torch.cov(fc_features.T)
            try:
                L = torch.linalg.cholesky(sigma)
            except RuntimeError:
                # Nếu ma trận vẫn chưa xác định dương, thêm jitter rồi thử lại
                eps = 1e-5
                sigma = sigma + eps * torch.eye(sigma.size(0), device=sigma.device)
                L = torch.linalg.cholesky(sigma)
            self.net.distributions.append(MultivariateNormal(mu, scale_tril=L))
            # self.net.distributions.append(MultivariateNormal(mu, sigma))

        #deal with grad and blocks
        for fc in self.net.fcArr:
            for param in fc.parameters():
                param.requires_grad = False

        for cls in self.net.classifierArr:
            for param in cls.parameters():
                param.requires_grad = False

        # self.net.Freezed_local_blocks = copy.deepcopy(torch.nn.Sequential(*list(self.net.local_vitmodel.blocks[-3:])))
        # self.net.Freezed_global_blocks = copy.deepcopy(torch.nn.Sequential(*list(self.net.global_vitmodel.blocks[-3:])))

        # for block in self.net.Freezed_local_blocks:
        #     for param in block.parameters():
        #         param.requires_grad = False

        # for block in self.net.Freezed_global_blocks:
        #     for param in block.parameters():
        #         param.requires_grad = False

    def begin_task(self, dataset, threshold=0) -> None:
        if self.current_task > 0:
            self.net.CreateNewExper(-1, dataset.N_CLASSES)

        self.opt = self.get_optimizer()
        for i in range(3):
            current_indices = self.list_indices[self.current_task]
            self.net.global_vitmodel.blocks[9 + i].attn.start_task(current_indices)
            self.net.local_vitmodel.blocks[9 + i].attn = BiLoRA_Local(dim=768, local_indices=current_indices).to(self.device)
        
        
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
        if len(self.net.fcArr) > 1:
            # NOTE: Task tiếp theo
            outputs, Freezed_global_features, Freezed_local_features, global_features, local_features = self.net(inputs, return_features=True)
            loss_kd = kl_loss(local_features, Freezed_local_features)
            loss_mi = kl_loss(global_features, Freezed_global_features) #Directly calculate the L2 distance between features is more efficient than calculate MI between prediction, and it's also effective
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_kd + loss_hsic + loss_mi
            loss_vis = [loss_ce.item(), loss_kd.item(), loss_hsic.item(), loss_mi.item()]
        else:
            # NOTE: Task đầu tiên
            t1 = time.time()
            outputs, global_features, local_features = self.net(inputs)
            t2 = time.time()
            loss_hsic = -hsic(global_features, local_features)
            t3 = time.time()
            loss_ce = self.loss(outputs, labels)
            t4 = time.time()
            print("Passing model: {0:.2f}|HSIC Loss: {1:.2f}|CELoss: {2:.2f}".format(t2 - t1, t3 - t2, t4 - t3))
            loss_tot = loss_ce + loss_hsic
            loss_vis = [loss_ce.item(), loss_hsic.item()]

        loss_tot.backward()


        self.opt.step()

        return loss_vis

    def cal_expert_dist(self,x):
        processX = self.net.vitProcess(x)
        if processX.size(1) == 1:
            processX = processX.expand(-1, 3, -1, -1)

        features = self.net.local_vitmodel.patch_embed(processX)
        cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_token, features), dim=1)
        features = features + self.net.local_vitmodel.pos_embed

        # forward pass till -3
        for block in self.net.local_vitmodel.blocks[:-3]:
            features = block(features)

        features = self.net.Forever_freezed_blocks(features)

        features = self.net.local_vitmodel.norm(features)

        class_token = features[:, 0, :]
        distances = [0] * len(self.net.fcArr)
        for t, (fc, dist) in enumerate(zip(self.net.fcArr, self.net.distributions)):
            fc_feature = fc(class_token)
            delta = fc_feature - dist.mean
            inv_cov = torch.linalg.inv(dist.covariance_matrix)
            mahalanobis = torch.sqrt(delta @ inv_cov @ delta.T).diagonal()
            distances[t] += mahalanobis.mean().item()
        return distances

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
            self.net.global_vitmodel.blocks[9 + i].attn = BiLoRA_Global(dim=768).to(self.device)
            
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