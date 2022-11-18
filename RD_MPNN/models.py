# coding: utf-8
'''
**************************************************************************
The models here are designed to work with datasets having maximum arity 6.
**************************************************************************
'''


import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
import math

from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import MessagePassing
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils import get_param, softmax

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)


class MDistMult(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MDistMult, self).__init__()
        self.emb_dim = emb_dim
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data[1:])

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)
        e3 = self.E(e3_idx)
        e4 = self.E(e4_idx)
        e5 = self.E(e5_idx)
        e6 = self.E(e6_idx)

        x = r * e1 * e2 * e3 * e4 * e5 * e6
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x


class MCP(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MCP, self).__init__()
        self.emb_dim = emb_dim
        self.E1 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E2 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E3 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E4 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E5 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E6 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)

        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)


    def init(self):
        self.E1.weight.data[0] = torch.ones(self.emb_dim)
        self.E2.weight.data[0] = torch.ones(self.emb_dim)
        self.E3.weight.data[0] = torch.ones(self.emb_dim)
        self.E4.weight.data[0] = torch.ones(self.emb_dim)
        self.E5.weight.data[0] = torch.ones(self.emb_dim)
        self.E6.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E1.weight.data[1:])
        xavier_normal_(self.E2.weight.data[1:])
        xavier_normal_(self.E3.weight.data[1:])
        xavier_normal_(self.E4.weight.data[1:])
        xavier_normal_(self.E5.weight.data[1:])
        xavier_normal_(self.E6.weight.data[1:])
        xavier_normal_(self.R.weight.data)

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E1(e1_idx)
        e2 = self.E2(e2_idx)
        e3 = self.E3(e3_idx)
        e4 = self.E4(e4_idx)
        e5 = self.E5(e5_idx)
        e6 = self.E6(e6_idx)
        x = r * e1 * e2 * e3 * e4 * e5 * e6
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x


class HSimplE(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(HSimplE, self).__init__()
        self.emb_dim = emb_dim
        self.max_arity = 6
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)


    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data)


    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E(e1_idx)
        e2 = self.shift(self.E(e2_idx), int(1 * self.emb_dim/self.max_arity))
        e3 = self.shift(self.E(e3_idx), int(2 * self.emb_dim/self.max_arity))
        e4 = self.shift(self.E(e4_idx), int(3 * self.emb_dim/self.max_arity))
        e5 = self.shift(self.E(e5_idx), int(4 * self.emb_dim/self.max_arity))
        e6 = self.shift(self.E(e6_idx), int(5 * self.emb_dim/self.max_arity))
        x = r * e1 * e2 * e3 * e4 * e5 * e6
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x


class HypE(BaseClass):
    def __init__(self, d, emb_dim, **kwargs):
        super(HypE, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.in_channels = kwargs["in_channels"]    # 1
        self.out_channels = kwargs["out_channels"]  # 6
        self.filt_h = kwargs["filt_h"]  # 1
        self.filt_w = kwargs["filt_w"]  # 1
        self.stride = kwargs["stride"]  # 2
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.emb_dim = emb_dim
        self.max_arity = 6
        rel_emb_dim = emb_dim
        self.E = torch.nn.Embedding(d.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(d.num_rel(), rel_emb_dim, padding_idx=0)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.inp_drop = torch.nn.Dropout(0.2)

        # fc_length 是特征图拼接后向量的长度
        fc_length = (1-self.filt_h+1)*math.floor((emb_dim-self.filt_w)/self.stride + 1)*self.out_channels

        self.bn2 = torch.nn.BatchNorm1d(fc_length)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        # Projection network    fc为P
        self.fc = torch.nn.Linear(fc_length, emb_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # size of the convolution filters outputted by the hypernetwork
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        # Hypernetwork
        self.fc1 = torch.nn.Linear(rel_emb_dim + self.max_arity + 1, fc1_length)
        # 卷积核参数
        self.fc2 = torch.nn.Linear(self.max_arity + 1, fc1_length)


    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])

    def convolve(self, r_idx, e_idx, pos):
        # 将实体嵌入整理成 [batch,1,1,emb]
        e = self.E(e_idx).view(-1, 1, 1, self.E.weight.size(1))
        # 关系嵌入
        r = self.R(r_idx)

        x = e
        x = self.inp_drop(x)

        # x就是卷积输入
        # one_hot_target tensor([1., 0., 0., 0., 0., 0., 0.]) 7
        one_hot_target = (pos == torch.arange(self.max_arity + 1).reshape(self.max_arity + 1)).float().to(self.device)
        # poses [b,7]
        poses = one_hot_target.repeat(r.shape[0]).view(-1, self.max_arity + 1)
        one_hot_target.requires_grad = False
        poses.requires_grad = False

        # 得到batch个 卷积核组的参数数量的参数， 实际上就是取出fc2中的第i行， fc2中每一行的参数数量正好是卷积核组的参数数量
        k = self.fc2(poses)
        # 将k分成卷积核形式
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)

        # k ->
        k = k.view(e.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)
        x = x.permute(1, 0, 2, 3)

        # 卷积操作
        x = F.conv2d(x, k, stride=self.stride, groups=e.size(0))

        # x -> [b, 1, out:fm数, fm宽, fm长]
        x = x.view(e.size(0), 1, self.out_channels, 1-self.filt_h+1, -1)
        # x -> [b, fm宽, fm长, 1, out:fm数]
        x = x.permute(0, 3, 4, 1, 2)
        # x -> [b, fm宽, fm长, out:fm数]
        x = torch.sum(x, dim=3)
        # x -> [b, out:fm数, fm宽, fm长]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(e.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms, bs):
        r = self.R(r_idx)
        e1 = self.convolve(r_idx, e1_idx, 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
        e2 = self.convolve(r_idx, e2_idx, 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
        e3 = self.convolve(r_idx, e3_idx, 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
        e4 = self.convolve(r_idx, e4_idx, 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
        e5 = self.convolve(r_idx, e5_idx, 4) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
        e6 = self.convolve(r_idx, e6_idx, 5) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)

        x = e1 * e2 * e3 * e4 * e5 * e6 * r
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x


class MTransH(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MTransH, self).__init__()
        self.emb_dim = emb_dim
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R1 = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.R2 = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)

        self.b0 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b1 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b2 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b3 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b4 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b5 = torch.nn.Embedding(dataset.num_rel(), 1)

        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R1.weight.data[0] = torch.ones(self.emb_dim)
        self.R2.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R1.weight.data[1:])
        xavier_normal_(self.R2.weight.data[1:])
        normalize_entity_emb = F.normalize(self.E.weight.data[1:], p=2, dim=1)
        normalize_relation_emb = F.normalize(self.R1.weight.data[1:], p=2, dim=1)
        normalize_norm_emb = F.normalize(self.R2.weight.data[1:], p=2, dim=1)
        self.E.weight.data[1:] = normalize_entity_emb
        self.R1.weight.data[1:] = normalize_relation_emb
        self.R2.weight.data[1:] = normalize_norm_emb
        xavier_normal_(self.b0.weight.data)
        xavier_normal_(self.b1.weight.data)
        xavier_normal_(self.b2.weight.data)
        xavier_normal_(self.b3.weight.data)
        xavier_normal_(self.b4.weight.data)
        xavier_normal_(self.b5.weight.data)

    def pnr(self, e_idx, r_idx):
        original = self.E(e_idx)
        norm = self.R2(r_idx)
        return original - torch.sum(original * norm, dim=1, keepdim=True) * norm

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms):
        r = self.R1(r_idx)
        e1 = self.pnr(e1_idx, r_idx) * self.b0(r_idx)
        e1 = e1 * ms[:,0].unsqueeze(-1).expand_as(e1)
        e2 = self.pnr(e2_idx, r_idx) * self.b1(r_idx)
        e2 = e2 * ms[:,1].unsqueeze(-1).expand_as(e2)
        e3 = self.pnr(e3_idx, r_idx) * self.b2(r_idx)
        e3 = e3 * ms[:,2].unsqueeze(-1).expand_as(e3)
        e4 = self.pnr(e4_idx, r_idx) * self.b3(r_idx)
        e4 = e4 * ms[:,3].unsqueeze(-1).expand_as(e4)
        e5 = self.pnr(e5_idx, r_idx) * self.b4(r_idx)
        e5 = e5 * ms[:,4].unsqueeze(-1).expand_as(e5)
        e6 = self.pnr(e6_idx, r_idx) * self.b5(r_idx)
        e6 = e6 * ms[:,5].unsqueeze(-1).expand_as(e6)
        x = r + e1 + e2 + e3 + e4 + e5 + e6
        x = self.hidden_drop(x)
        x = -1 * torch.norm(x, p=2, dim=1)
        return x


# MPNN+Transformer
# class MPNN(BaseClass):
#     def __init__(self, hyperedge, edge_index, edge_type, device, dataset, emb_dim, **kwargs):
#         super(MPNN, self).__init__()
#         self.emb_dim = emb_dim
#         self.max_arity = 6
#         self.device = device
#         self.ent_num = torch.tensor(dataset.num_ent(), dtype=torch.long, device=self.device)
#
#         # self.P = torch.nn.Embedding(6, emb_dim)
#         self.hidden_drop_rate = kwargs["hidden_drop"]
#         self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
#
#         # self.E = get_param((dataset.num_ent() + 6, emb_dim))
#         # self.R = get_param((dataset.num_rel(), emb_dim))
#         self.E = torch.zeros(dataset.num_ent() + 6, emb_dim)
#         self.R = torch.zeros(dataset.num_rel(), emb_dim)
#
#         # add
#         self.hyperedge = torch.tensor(hyperedge, dtype=torch.long, device=self.device)
#         self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
#         self.edge_type = torch.tensor(edge_type, dtype=torch.long, device=self.device)
#
#         self.MPlayer = MP(dataset, self.device, emb_dim, **kwargs).to(self.device)
#
#         # transformer
#         self.hid_drop = 0.3
#         self.hid_drop2 = 0.1
#         self.feat_drop = 0.3
#         self.num_transformer_layers = 2
#         self.num_heads = 4
#         self.num_hidden = 512
#         self.d_model = 200
#         self.positional = True
#         self.pooling = "avg"
#
#         self.hidden_drop = torch.nn.Dropout(self.hid_drop)
#         self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
#         self.feature_drop = torch.nn.Dropout(self.feat_drop)
#
#         # 初始化一层
#         encoder_layers = TransformerEncoderLayer(self.emb_dim, self.num_heads, self.num_hidden, self.hid_drop2)
#         # 叠加多层
#         self.encoder = TransformerEncoder(encoder_layers, self.num_transformer_layers)
#         # transformer 中的位置嵌入
#         self.position_embeddings = torch.nn.Embedding(self.max_arity + 1, self.emb_dim)
#
#         self.layer_norm = torch.nn.LayerNorm(self.emb_dim)  # 200
#
#         if self.pooling == "concat":
#             self.flat_sz = self.emb_dim * (self.max_arity + 1)
#             self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
#         else:
#             self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)  # run
#
#         self.fc_out = torch.nn.Linear(self.emb_dim, 1)
#
#     def init(self):
#         pass
#
#         # self.E.weight.data[0] = torch.zeros(self.emb_dim)
#         # xavier_normal_(self.E.weight.data[1:])
#         # xavier_normal_(self.R.weight.data)
#
#     def concat(self, r, e1, e2, e3, e4, e5, e6):
#         # e1_embed [N,1,200]
#         r = r.view(-1, 1, self.emb_dim)
#         e1 = e1.view(-1, 1, self.emb_dim)
#         e2 = e2.view(-1, 1, self.emb_dim)
#         e3 = e3.view(-1, 1, self.emb_dim)
#         e4 = e4.view(-1, 1, self.emb_dim)
#         e5 = e5.view(-1, 1, self.emb_dim)
#         e6 = e6.view(-1, 1, self.emb_dim)
#
#         # 2,N,200
#         stack_inp = torch.cat([r, e1, e2, e3, e4, e5, e6], 1).transpose(1, 0)  # [7, N, emb_dim]
#         return stack_inp
#
#     def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
#
#         outE, outR = self.MPlayer(self.hyperedge, self.edge_index, self.edge_type)
#
#         r = outR[r_idx]
#         e1 = outE[e1_idx] * outE[self.ent_num]
#         e2 = outE[e2_idx] * outE[self.ent_num + 1]
#         e3 = outE[e3_idx] * outE[self.ent_num + 2]
#         e4 = outE[e4_idx] * outE[self.ent_num + 3]
#         e5 = outE[e5_idx] * outE[self.ent_num + 4]
#         e6 = outE[e6_idx] * outE[self.ent_num + 5]
#
#         stk_inp = self.concat(r, e1, e2, e3, e4, e5, e6)
#
#         if self.positional:
#             # N,7
#             positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
#             # N,7,200 -> 7,N,200
#             pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
#             stk_inp = stk_inp + pos_embeddings
#
#         mask = torch.zeros((r_idx.shape[0], self.max_arity + 1)).bool().to(self.device)
#         mask[:, 3] = e3_idx == 0
#         mask[:, 4] = e4_idx == 0
#         mask[:, 5] = e5_idx == 0
#         mask[:, 6] = e6_idx == 0
#
#         # put emb+pos in transformer
#         x = self.encoder(stk_inp, src_key_padding_mask=mask)
#
#         if self.pooling == 'concat':
#             x = x.transpose(1, 0).reshape(-1, self.flat_sz)
#         elif self.pooling == "avg":
#             x = torch.mean(x, dim=0)
#         elif self.pooling == "min":
#             x, _ = torch.min(x, dim=0)
#
#         x = self.fc(x)
#         x = self.fc_out(x)  # sum score
#         x = torch.squeeze(x)
#         self.E = outE
#         self.R = outR
#         return x
#
#     def test(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
#         #   得分函数
#
#         r = self.R[r_idx]
#         e1 = self.E[e1_idx] * self.E[self.ent_num]
#         e2 = self.E[e2_idx] * self.E[self.ent_num + 1]
#         e3 = self.E[e3_idx] * self.E[self.ent_num + 2]
#         e4 = self.E[e4_idx] * self.E[self.ent_num + 3]
#         e5 = self.E[e5_idx] * self.E[self.ent_num + 4]
#         e6 = self.E[e6_idx] * self.E[self.ent_num + 5]
#
#         stk_inp = self.concat(r, e1, e2, e3, e4, e5, e6)
#
#         if self.positional:
#             # N,7
#             positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
#             # N,7,200 -> 7,N,200
#             pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
#             stk_inp = stk_inp + pos_embeddings
#
#         mask = torch.zeros((r_idx.shape[0], self.max_arity + 1)).bool().to(self.device)
#         mask[:, 3] = e3_idx == 0
#         mask[:, 4] = e4_idx == 0
#         mask[:, 5] = e5_idx == 0
#         mask[:, 6] = e6_idx == 0
#
#         # put emb+pos in transformer
#         x = self.encoder(stk_inp, src_key_padding_mask=mask)
#
#         if self.pooling == 'concat':
#             x = x.transpose(1, 0).reshape(-1, self.flat_sz)
#         elif self.pooling == "avg":
#             x = torch.mean(x, dim=0)
#         elif self.pooling == "min":
#             x, _ = torch.min(x, dim=0)
#
#         x = self.fc(x)
#         x = self.fc_out(x)
#
#         return x
#
#

# MPNN+ConvKB


class MPNN(BaseClass):
    def __init__(self, hyperedge, edge_index, edge_type, device, dataset, emb_dim, **kwargs):
        super(MPNN, self).__init__()
        self.emb_dim = emb_dim
        self.max_arity = 6
        self.device = device
        self.ent_num = torch.tensor(dataset.num_ent(), dtype=torch.long, device=self.device)

        # self.P = torch.nn.Embedding(6, emb_dim)
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

        # self.E = get_param((dataset.num_ent() + 6, emb_dim))
        # self.R = get_param((dataset.num_rel(), emb_dim))
        # self.E = torch.zeros(dataset.num_ent() + 6, emb_dim)
        # self.R = torch.zeros(dataset.num_rel(), emb_dim)

        # add
        self.hyperedge = torch.tensor(hyperedge, dtype=torch.long, device=self.device)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        self.edge_type = torch.tensor(edge_type, dtype=torch.long, device=self.device)

        self.MPlayer = MP(dataset, self.device, emb_dim, **kwargs).to(self.device)

        # conv
        self.kernel_sz_h = 7
        self.kernel_sz_w = 3
        self.n_filters = 200
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.n_filters,
                                       kernel_size=(self.kernel_sz_h, self.kernel_sz_w), stride=1)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)

        self.hidden_drop = torch.nn.Dropout(0.3)
        self.hidden_drop2 = torch.nn.Dropout(0.1)
        self.feature_drop = torch.nn.Dropout(0.3)

        feature_size_h = 1
        feature_size_w = self.emb_dim - self.kernel_sz_w + 1
        self.feature_size = feature_size_h * feature_size_w * self.n_filters
        self.fc = torch.nn.Linear(self.feature_size, self.emb_dim)

        # self.mlp1 = torch.nn.Linear(self.emb_dim, self.emb_dim)
        # self.mlp2 = torch.nn.Linear(self.emb_dim, self.emb_dim)
        # self.mlp3 = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def init(self):
        pass

    def concat(self, r, e1, e2, e3, e4, e5, e6):
        # e1_embed [bs,1,100]
        r = r.view(-1, 1, self.emb_dim)
        e1 = e1.view(-1, 1, self.emb_dim)
        e2 = e2.view(-1, 1, self.emb_dim)
        e3 = e3.view(-1, 1, self.emb_dim)
        e4 = e4.view(-1, 1, self.emb_dim)
        e5 = e5.view(-1, 1, self.emb_dim)
        e6 = e6.view(-1, 1, self.emb_dim)

        # bs,7,200
        stack_inp = torch.cat([r, e1, e2, e3, e4, e5, e6], 1).reshape((-1, 1, self.max_arity + 1, self.emb_dim))
        return stack_inp


    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms, bs):

        outE, outR = self.MPlayer(self.hyperedge, self.edge_index, self.edge_type)

        r = outR[r_idx]
        e1 = outE[e1_idx] * outE[self.ent_num] * ms[:, 0].view(-1, 1)
        e2 = outE[e2_idx] * outE[self.ent_num + 1] * ms[:, 1].view(-1, 1)
        e3 = outE[e3_idx] * outE[self.ent_num + 2] * ms[:, 2].view(-1, 1)
        e4 = outE[e4_idx] * outE[self.ent_num + 3] * ms[:, 3].view(-1, 1)
        e5 = outE[e5_idx] * outE[self.ent_num + 4] * ms[:, 4].view(-1, 1)
        e6 = outE[e6_idx] * outE[self.ent_num + 5] * ms[:, 5].view(-1, 1)

        stk_inp = self.concat(r, e1, e2, e3, e4, e5, e6)

        # 卷积
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)  #  bs * n_filters * feature_size_h * feature_size_w     bs*200*1*98

        # 卷积后处理
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)

        # 打平
        x = x.view(-1, self.feature_size)

        # 矩阵转换
        x = self.fc(x)      # bs * 100
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # x = self.mlp1(x)
        # x = self.mlp2(x)
        # x = self.mlp3(x)

        # self.E = outE
        # self.R = outR

        x = torch.sum(x, dim=1)     # bs
        return x


    def test(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms, bs):

        #   得分函数
        outE, outR = self.MPlayer(self.hyperedge, self.edge_index, self.edge_type)

        r = outR[r_idx]
        e1 = outE[e1_idx] * outE[self.ent_num] * ms[:, 0].view(-1, 1)
        e2 = outE[e2_idx] * outE[self.ent_num + 1] * ms[:, 1].view(-1, 1)
        e3 = outE[e3_idx] * outE[self.ent_num + 2] * ms[:, 2].view(-1, 1)
        e4 = outE[e4_idx] * outE[self.ent_num + 3] * ms[:, 3].view(-1, 1)
        e5 = outE[e5_idx] * outE[self.ent_num + 4] * ms[:, 4].view(-1, 1)
        e6 = outE[e6_idx] * outE[self.ent_num + 5] * ms[:, 5].view(-1, 1)

        stk_inp = self.concat(r, e1, e2, e3, e4, e5, e6)

        # 卷积
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)  # bs * n_filters * feature_size_h * feature_size_w     bs*20*1*98

        # 卷积后处理
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)

        # 打平
        x = x.view(-1, self.feature_size)

        # 矩阵转换
        x = self.fc(x)  # bs * 100
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # x = self.mlp1(x)
        # x = self.mlp2(x)
        # x = self.mlp3(x)

        x = torch.sum(x, dim=1)  # bs
        return x





# # MPNN
# class MPNN(BaseClass):
#     def __init__(self, hyperedge, edge_index, edge_type, device, dataset, emb_dim, **kwargs):
#         super(MPNN, self).__init__()
#         self.emb_dim = emb_dim
#         self.max_arity = 6
#         self.device = device
#         self.ent_num = torch.tensor(dataset.num_ent(), dtype=torch.long, device=self.device)
#
#         # self.P = torch.nn.Embedding(6, emb_dim)
#         self.hidden_drop_rate = kwargs["hidden_drop"]
#         self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
#
#         # self.E = get_param((dataset.num_ent() + 6, emb_dim))
#         # self.R = get_param((dataset.num_rel(), emb_dim))
#         self.E = torch.zeros(dataset.num_ent() + 6, emb_dim)
#         self.R = torch.zeros(dataset.num_rel(), emb_dim)
#
#         # add
#         self.hyperedge = torch.tensor(hyperedge, dtype=torch.long, device=self.device)
#         self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
#         self.edge_type = torch.tensor(edge_type, dtype=torch.long, device=self.device)
#
#         self.MPlayer = MP(dataset, self.device, emb_dim, **kwargs).to(self.device)
#
#     def init(self):
#         pass
#
#
#     def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
#
#         outE, outR = self.MPlayer(self.hyperedge, self.edge_index, self.edge_type)
#
#         r = outR[r_idx]
#         e1 = outE[e1_idx] * outE[self.ent_num]
#         e2 = outE[e2_idx] * outE[self.ent_num + 1]
#         e3 = outE[e3_idx] * outE[self.ent_num + 2]
#         e4 = outE[e4_idx] * outE[self.ent_num + 3]
#         e5 = outE[e5_idx] * outE[self.ent_num + 4]
#         e6 = outE[e6_idx] * outE[self.ent_num + 5]
#
#         x = e1 * e2 * e3 * e4 * e5 * e6 * r
#
#         x = self.hidden_drop(x)
#         x = torch.sum(x, dim=1)
#
#         self.E = outE
#         self.R = outR
#         return x
#
#     def test(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
#         #   得分函数
#
#         r = self.R[r_idx]
#         e1 = self.E[e1_idx] * self.E[self.ent_num]
#         e2 = self.E[e2_idx] * self.E[self.ent_num + 1]
#         e3 = self.E[e3_idx] * self.E[self.ent_num + 2]
#         e4 = self.E[e4_idx] * self.E[self.ent_num + 3]
#         e5 = self.E[e5_idx] * self.E[self.ent_num + 4]
#         e6 = self.E[e6_idx] * self.E[self.ent_num + 5]
#
#         x = e1 * e2 * e3 * e4 * e5 * e6 * r
#
#         x = self.hidden_drop(x)
#         x = torch.sum(x, dim=1)
#         return x




class MP(MessagePassing):
    """
        MessagePassing
    """
    def __init__(self, dataset, device, emb_dim, **kwargs):
        super(self.__class__, self).__init__(flow='target_to_source', aggr='add')
        # step1
        # 定义网络参数等
        self.emb_dim = emb_dim
        self.drop = torch.nn.Dropout(0.2)
        self.bn = torch.nn.BatchNorm1d(emb_dim)
        self.act = torch.tanh
        self.ent_num = kwargs["ent_num"]
        self.attention = False  # 是否使用注意力
        self.mlp = False        # 是否使用mlp技巧
        self.multi = False      # 信息组合方式 True为相加相乘 False为concat

        self.E_in = torch.nn.Embedding(dataset.num_ent() + 6, emb_dim, padding_idx=0)
        self.R_in = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        xavier_normal_(self.E_in.weight.data)
        xavier_normal_(self.R_in.weight.data)

        if self.multi:
            self.w_alle = get_param((emb_dim, emb_dim))
        else:
            self.w_alle = get_param((emb_dim * 6, emb_dim))
            self.w_addpos = get_param((emb_dim * 2, emb_dim))

        self.w_alleandr = get_param((emb_dim, emb_dim))
        self.w_rel = get_param((emb_dim, emb_dim))  # (100,100)

        if self.mlp:
            self.num_mlp_layers = 3
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(torch.nn.Linear(self.emb_dim, self.emb_dim))
            for layer in range(self.num_mlp_layers - 2):
                self.linears.append(torch.nn.Linear(self.emb_dim, self.emb_dim))
            self.linears.append(torch.nn.Linear(self.emb_dim, self.emb_dim))

            for layer in range(self.num_mlp_layers - 1):
                self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))

        if self.attention:
            self.heads = 2
            self.attn_dim = self.emb_dim // self.heads  # 50
            self.negative_slope = 0.2
            self.attn_drop = 0.1
            self.att = get_param((1, self.heads, 2 * self.attn_dim))    # 1*2*100

    def MLP(self, x):
        # print("**** mlp is used ***")
        h = x
        for layer in range(self.num_mlp_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_mlp_layers - 1](h)

    def forward(self, hyperedge, edge_index, edge_type):

        # setp1
        # propagate信息传播过程，得到更新后的实体矩阵
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.in_norm = None
        x = self.propagate(self.edge_index, x=self.E_in.weight, edge_type=self.edge_type,
                                rel_embed=self.R_in.weight, edge_norm=self.in_norm, hyperedge=hyperedge, edge_indexs=edge_index,
                                ent_embed=self.E_in.weight, source_index=self.edge_index[0])

        out = self.drop(x) * (1 / 2) + self.drop(self.E_in.weight) * (1 / 2)

        if self.mlp:
            out = self.MLP(out)

        out = self.bn(out)
        out = self.act(out)


        # step3
        # 更新关系嵌入矩阵
        out_r = torch.matmul(self.R_in.weight, self.w_rel)

        return out, out_r

    def message(self, x_j, x_i, edge_type, rel_embed, edge_norm, hyperedge, edge_indexs, ent_embed, source_index):
        """
            信息传递函数
        :param x_j: 尾实体嵌入矩阵，实际上是位置。没用
        :param x_i: 头实体嵌入矩阵
        :param edge_type:
        :param rel_embed:
        :param edge_norm:
        :param hyperedge:
        :param ent_embed:
        :return:
        """
        hyper_emb = self.get_hyperedge_emb(ent_embed, rel_embed, edge_type, hyperedge, edge_indexs)
        statement = hyper_emb[edge_type]

        print('x_j.shape', x_j.shape, 'x_j', x_j[:10])
        # print('pos_emb', ent_embed[self.ent_num:])
        if self.multi:
            out = statement * x_j
        else:
            out = torch.cat((statement, x_j), dim=1)
            out = torch.matmul(out, self.w_addpos)

        if self.attention:
            # 结果嵌入 和 头实体嵌入 分成n个头
            out = out.view(-1, self.heads, self.attn_dim)
            x_i = x_i.view(-1, self.heads, self.attn_dim)

            alpha = torch.einsum('bij,kij -> bi', [torch.cat([x_i, out], dim=-1), self.att])
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, source_index, ent_embed.size(0))
            alpha = F.dropout(alpha, p=self.attn_drop)
            out = out * alpha.view(-1, self.heads, 1)
            out = out.view(-1, self.heads * self.attn_dim)


        return out

    # 更新聚合信息，得到最终输出
    def update(self, aggr_out):

        return aggr_out

    # @staticmethod   # 计算每个三元组的norm值,全部设置为1
    # def compute_norm():
    #     return 1

    def get_hyperedge_emb(self, ent_embed, rel_embed, edge_type, hyperedge, edge_indexs):

        r = rel_embed[hyperedge[:,0]]

        e1 = ent_embed[hyperedge[:, 1]] * ent_embed[self.ent_num]
        e2 = ent_embed[hyperedge[:, 2]] * ent_embed[self.ent_num + 1]
        e3 = ent_embed[hyperedge[:, 3]] * ent_embed[self.ent_num + 2]
        e4 = ent_embed[hyperedge[:, 4]] * ent_embed[self.ent_num + 3]
        e5 = ent_embed[hyperedge[:, 5]] * ent_embed[self.ent_num + 4]
        e6 = ent_embed[hyperedge[:, 6]] * ent_embed[self.ent_num + 5]

        if self.multi:
            e_all = e1 + e2 + e3 + e4 + e5 + e6
            e_all = torch.matmul(e_all, self.w_alle)
        else:
            e_all = torch.cat((e1, e2, e3, e4, e5, e6), dim=1)
            e_all = torch.matmul(e_all, self.w_alle)

        out = (1 / 2) * e_all + (1 / 2) * r
        out = torch.matmul(out, self.w_alleandr)

        return out

