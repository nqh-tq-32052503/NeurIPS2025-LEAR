import math
import logging
from functools import partial
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0

    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def init_param_ada(self, t, r):
        self.lora_A_k[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_k[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)
        self.lora_A_v[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_v[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)

        nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k[t].weight)
        nn.init.zeros_(self.lora_B_v[t].weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        if task > -0.5:
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
        return matrix_k, matrix_v
    
    def get_pre_matrix(self, task):
        with torch.no_grad():
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v
    

def FFT_SHIFT(matrix):
        m_clone = matrix.clone()
        m,n = m_clone.shape
        m = int(m / 2)
        n = int(n / 2)

        for i in range(m):
            for j in range(n):
                m_clone[i][j] = matrix[m+i][n+j]
                m_clone[m+i][n+j] = matrix[i][j]
                m_clone[m+i][j] = matrix[i][j+n]
                m_clone[i][j+n] = matrix[m+i][j]
        return m_clone        

class Attention_LoRA_FFT_Separate(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=3000, indices=None):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)

        self.n_frq = n_frq
        self.coef_k = nn.Parameter(torch.randn(self.n_frq), requires_grad=True).to(self.qkv.weight.device)
        self.coef_v = nn.Parameter(torch.randn(self.n_frq), requires_grad=True).to(self.qkv.weight.device)
        self.indices = indices
    
    def select_pos(self, t, dim, seed=777):
        indices = torch.randperm(dim * dim, generator=torch.Generator().manual_seed(seed+t*10))[:self.n_frq]
        indices = torch.stack([indices // dim, indices % dim], dim=0)
        return indices
    
    def get_delta_w_k(self, alpha=300):
        indices = self.indices
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_k
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha
    
    def get_delta_w_v(self, alpha=300):
        indices = self.indices
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_v
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        weight_k = self.get_delta_w_k()
        weight_v = self.get_delta_w_v()
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_LoRA_FFT_Aggregate(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=3000, list_indices=None):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)

        self.n_frq = n_frq
        self.list_indices = list_indices
        self.current_task = len(list_indices)
        self.coef_k = nn.ParameterList([nn.Parameter(torch.randn(self.n_frq), requires_grad=True) for _ in range(self.current_task)]).to(self.qkv.weight.device)
        self.coef_v = nn.ParameterList([nn.Parameter(torch.randn(self.n_frq), requires_grad=True) for _ in range(self.current_task)]).to(self.qkv.weight.device)
        

    
    def get_delta_w_k(self, task, alpha=300):
        indices = self.list_indices[task]
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_k[task]
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha
    
    def get_delta_w_v(self, task, alpha=300):
        indices = self.list_indices[task]
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_v[task]
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        weight_k = torch.stack([self.get_delta_w_k(t) for t in range(self.current_task)], dim=0).sum(dim=0)
        weight_v = torch.stack([self.get_delta_w_v(t) for t in range(self.current_task)], dim=0).sum(dim=0)
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BiLoRA_Manager(object):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=3000, device="cuda", mode="separate"):
        self.dim = dim
        self.device = device
        self.n_tasks = n_tasks
        self.n_frq = n_frq
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.r = r
        self.full_permutation = torch.randperm(
            dim * dim, 
            generator=torch.Generator().manual_seed(42)
        )
        self.indices = [self.select_pos(t).to(self.device) for t in range(n_tasks)]
        self.weights = []
        self.mode = mode

    def get_bilora_attn_separate(self, task):
        indices = self.indices[task]
        return Attention_LoRA_FFT_Separate(dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
                                  qk_scale=self.qk_scale, attn_drop=self.attn_drop, proj_drop=self.proj_drop,
                                  r=self.r, n_tasks=self.n_tasks, n_frq=self.n_frq, indices=indices).to(self.device)
    
    def get_bilora_attn_aggregate(self, task):
        list_indices = self.indices[:(task + 1)]
        return Attention_LoRA_FFT_Aggregate(dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
                                  qk_scale=self.qk_scale, attn_drop=self.attn_drop, proj_drop=self.proj_drop,
                                  r=self.r, n_tasks=self.n_tasks, n_frq=self.n_frq, list_indices=list_indices).to(self.device)
        
    
    def get_bilora_attn(self, task):
        if self.mode == "separate":
            return self.get_bilora_attn_separate(task=task)
        elif self.mode == "aggregate":
            return self.get_bilora_attn_aggregate(task=task)
        
    def select_pos(self, t):
        """
        t: index của tập dữ liệu (ví dụ 0, 1, 2... 9)
        """
        start = t * self.n_frq
        end = (t + 1) * self.n_frq
        indices = self.full_permutation[start:end]
        indices_2d = torch.stack([indices // self.dim, indices % self.dim], dim=0)
        return indices_2d

    def save_bilora_attn(self, bilora_attn: Attention_LoRA):
        saved_bilora_attn = copy.deepcopy(bilora_attn)
        self.weights.append(saved_bilora_attn)
        
class MoE(nn.Module):
    def __init__(self, dim, num_experts, n_frq, n_tasks, topk, device="cuda", use_expert_weights=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_frq = n_frq
        self.n_tasks = n_tasks
        self.topk = topk
        self.num_experts = num_experts
        self.use_expert_weights = use_expert_weights
        self.full_permutation = torch.randperm(dim * dim, generator=torch.Generator().manual_seed(42)).tolist()
        self.router = nn.Linear(dim, num_experts)
        self.coeff = nn.Parameter(torch.randn(num_experts, n_frq), requires_grad=True)
        self.create_bilora_indices()
        
    def create_bilora_indices(self):
        list_indices = [torch.tensor(self.full_permutation[t * self.n_frq : (t + 1) * self.n_frq], device=self.device) for t in range(self.num_experts)]
        self.list_indices = torch.stack(list_indices, dim=0)

    def vectorized_forward(self, batch_expert_weights, batch_expert_indices, alpha=300):
        B, topk = batch_expert_indices.shape
        device = self.coeff.device
        dim = self.dim
        all_selected_indices = self.list_indices[batch_expert_indices]
        current_coeffs = self.coeff[batch_expert_indices]
        if self.use_expert_weights:
            weighted_coeffs = current_coeffs * batch_expert_weights.unsqueeze(-1)
            flat_values = weighted_coeffs.view(-1)
        else:
            flat_values = current_coeffs.view(-1)
        aggregated_freq_mask = torch.zeros(B, dim, dim, device=device)
        batch_offsets = torch.arange(B, device=device).view(B, 1, 1) * (dim * dim)
        flat_indices = (all_selected_indices + batch_offsets).view(-1)
        aggregated_freq_mask.view(-1).index_put_(
            (flat_indices,), 
            flat_values, 
            accumulate=True
        )
        final_output = torch.fft.ifft2(aggregated_freq_mask, dim=(-2, -1)).real * alpha
        return final_output
    
    def forward(self, cls_token, **kwargs):
        alpha = 300
        router_logits = self.router(cls_token)
        batch_expert_weights, batch_expert_indices = torch.topk(F.softmax(router_logits, dim=-1), self.topk, dim=-1)
        batch_expert_weights = batch_expert_weights / batch_expert_weights.sum(dim=-1, keepdim=True)
        outputs = self.vectorized_forward(batch_expert_weights, batch_expert_indices, alpha=alpha)
        return outputs

class InverseMoE(nn.Module):
    def __init__(self, dim, num_experts=20, n_frq=3000, topk=5, device="cuda"):
        super().__init__()
        self.num_experts = num_experts
        self.n_frq = n_frq
        self.device = device
        self.dim = dim
        self.topk = topk
        self.full_permutation = torch.randperm(dim * dim, generator=torch.Generator().manual_seed(42))
        self.create_bilora_indices()
        self.router = nn.Linear(dim, num_experts)
        
    def create_bilora_indices(self):
        list_indices = [torch.tensor(self.full_permutation[t * self.n_frq : (t + 1) * self.n_frq], device=self.device) for t in range(self.num_experts)]
        self.list_indices = torch.stack(list_indices, dim=0)
    
    def forward(self, cls_token: torch.Tensor, **kwargs):
        B = cls_token.size(0)
        N = self.dim
        router_logits = self.router(cls_token)
        _, batch_expert_indices = torch.topk(F.softmax(router_logits, dim=-1), self.topk, dim=-1)
        all_selected_indices = self.list_indices[batch_expert_indices]
        indices_flat = all_selected_indices.view(B, -1).long()
        binary_matrix_flat = torch.zeros(B, N * N, device=self.device)
        binary_matrix_flat.scatter_(dim=1, index=indices_flat, value=1.0)
        return binary_matrix_flat.view(B, N, N)

class BiLORA_MoE(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=3000, num_experts=20, topk=5, use_expert_weights=False):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)
        self.num_experts= num_experts
        self.n_tasks = n_tasks
        assert num_experts < (dim * dim) / n_frq, "Number of experts is too large"
        self.topk = topk
        self.n_frq = n_frq
        self.use_expert_weights = use_expert_weights
        self.moe_k = MoE(dim=dim, num_experts=num_experts, n_frq=n_frq, n_tasks=n_tasks, topk=topk, device=self.qkv.weight.device, use_expert_weights=self.use_expert_weights)
        self.moe_v = MoE(dim=dim, num_experts=num_experts, n_frq=n_frq, n_tasks=n_tasks, topk=topk, device=self.qkv.weight.device, use_expert_weights=self.use_expert_weights)
    
    def to(self, device):
        self.moe_k.to(device)
        self.moe_v.to(device)
    
    def forward(self, x: torch.Tensor, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        weight_k = self.moe_k(x[:, 0, :])
        weight_v = self.moe_v(x[:, 0, :])
        k = k + torch.bmm(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + torch.bmm(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BiLoRA_InverseMoE(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0, r=64, n_tasks=10, num_experts=20):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)
        
        self.moe_k = InverseMoE(dim=dim, num_experts=num_experts, n_frq=r, topk=5, device=self.qkv.weight.device)
        self.moe_v = InverseMoE(dim=dim, num_experts=num_experts, n_frq=r, topk=5, device=self.qkv.weight.device)
    
    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        weight_k = self.moe_k(x[:, 0, :])
        weight_v = self.moe_v(x[:, 0, :])
        k = k @ weight_k
        v = v @ weight_v

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x