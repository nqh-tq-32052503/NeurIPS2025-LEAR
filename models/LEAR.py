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
from backbone.bilora import BiLORA_MoE, BiLoRA_InverseMoE
from models.tracing import TaskExpertTracing


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

    def end_task(self, dataset) -> None:
        #calculate distribution
        train_loader = dataset.train_loader
        num_choose = len(train_loader)
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

                fc_features_list.append(self.net.fcArr[self.current_task](class_token))

                count += 1
                pbar.update()

            pbar.close()
            fc_features = torch.cat(fc_features_list, dim=0)  # [num*b,fc_size]
            mu = torch.mean(fc_features, dim=0)
            sigma = torch.cov(fc_features.T)
            self.net.distributions.append(MultivariateNormal(mu, sigma))

        #deal with grad and blocks
        for fc in self.net.fcArr:
            for param in fc.parameters():
                param.requires_grad = False

        for cls in self.net.classifierArr:
            for param in cls.parameters():
                param.requires_grad = False
        self.reset_router_penalty()
        self.save_current_task()
        
    def begin_task(self, dataset, threshold=0) -> None:
        min_idx = 0
        if self.current_task > 0:
            self.net.CreateNewExper(min_idx, dataset.N_CLASSES)

        self.opt = self.get_optimizer()

    def save_current_task(self):
        index = self.current_task
        global_vitmodel = copy.deepcopy(self.net.global_vitmodel)
        local_vitmodel = copy.deepcopy(self.net.local_vitmodel)
        fc = copy.deepcopy(self.net.fcArr[self.current_task])
        classifier = copy.deepcopy(self.net.classifierArr[self.current_task])
        task_expert = TaskExpertTracing(global_vitmodel, local_vitmodel, fc, classifier)
        task_expert.eval().to(self.device)
        example_input = torch.rand(7, 3, 224, 224).to(self.device)
        traced_model = torch.jit.trace(task_expert, example_input)
        if not os.path.exists("./factory"):
            os.mkdir("./factory")
        traced_model.save(f"./factory/task_{index}.pt")
        print("[INFO] Save current task checkpoints")
    
    def myPrediction(self,x,k):
        with torch.no_grad():
            #Perform the prediction according to the seloeced expert
            # out = self.net.myprediction(x,k)
            out = self.net.myprediction(x,k, apply_task=True)
        return out

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        # HSIC: Measure of dependence between global and local features
        # Put negative sign because we want to maximize the dependence between them
        self.opt.zero_grad()
        outputs, global_features, local_features = self.net(inputs, return_features=True)
        loss_hsic = -hsic(global_features, local_features)
        loss_ce = self.loss(outputs, labels)
        p_loss = self.cal_router_penalty_loss()
        loss_tot = loss_ce + loss_hsic + p_loss
        loss_vis = [loss_ce.item(), loss_hsic.item(), p_loss.item(), 0]

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

    def reset_router_penalty(self):
        zero_tensor = torch.zeros(768, device=self.device)
        for i in range(3):
            self.net.global_vitmodel.blocks[9 + i].attn.moe_v.expert_weights = zero_tensor
            self.net.global_vitmodel.blocks[9 + i].attn.moe_k.expert_weights = zero_tensor
            self.net.local_vitmodel.blocks[9 + i].attn.moe_v.expert_weights = zero_tensor
            self.net.local_vitmodel.blocks[9 + i].attn.moe_k.expert_weights = zero_tensor
        print("[INFO] Reset router penalty")
    
    def init_bilora(self):
        print("[INFO] Initializing BiLoRA MoE")
        for i in range(3):
            self.net.global_vitmodel.blocks[9 + i].attn = BiLORA_MoE(dim=768, n_frq=self.args.n_frq, num_experts=self.args.n_experts, topk=self.args.topk)
        for i in range(3):
            self.net.local_vitmodel.blocks[9 + i].attn = BiLORA_MoE(dim=768, n_frq=self.args.n_frq, num_experts=self.args.n_experts, topk=self.args.topk)

    def cal_router_penalty_loss(self):
        router_penalty = 0
        for i in range(3):
            k_loss = penalty_loss(self.net.global_vitmodel.blocks[9 + i].attn.moe_k.expert_weights)
            v_loss = penalty_loss(self.net.global_vitmodel.blocks[9 + i].attn.moe_v.expert_weights)
            router_penalty += k_loss + v_loss
        for i in range(3):
            k_loss = penalty_loss(self.net.local_vitmodel.blocks[9 + i].attn.moe_k.expert_weights)
            v_loss = penalty_loss(self.net.local_vitmodel.blocks[9 + i].attn.moe_v.expert_weights)
            router_penalty += k_loss + v_loss
        return router_penalty

def penalty_loss(X, threshold=0.7):
    # Penalty loss: Chỉ phạt khi X < threshold
    # print("X (expert weights):", torch.mean(X))
    penalty = torch.mean(torch.clamp(threshold - X, min=0))
    return penalty

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