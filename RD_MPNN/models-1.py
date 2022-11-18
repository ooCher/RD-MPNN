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

from utils import get_param

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
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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


class MPNN(BaseClass):
    def __init__(self, hyperedge, edge_index, edge_type, device, dataset, emb_dim, **kwargs):
        super(MPNN, self).__init__()
        self.emb_dim = emb_dim
        self.max_arity = 6
        self.device = device
        self.ent_num = dataset.num_ent()
        self.E = torch.nn.Embedding(dataset.num_ent() + 6, emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        # self.P = torch.nn.Embedding(6, emb_dim)
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

        # add
        self.hyperedge = torch.tensor(hyperedge, dtype=torch.long, device=self.device)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        self.edge_type = torch.tensor(edge_type, dtype=torch.long, device=self.device)

        self.MPlayer = MP(self.device, emb_dim, **kwargs).to(self.device)

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data)

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        E, R = self.MPlayer(self.hyperedge, self.edge_index, self.edge_type, self.E.weight, self.R.weight)

        r = R[r_idx]
        e1 = E[e1_idx] * E[self.ent_num]
        e2 = E[e2_idx] * E[self.ent_num + 1]
        e3 = E[e3_idx] * E[self.ent_num + 2]
        e4 = E[e4_idx] * E[self.ent_num + 3]
        e5 = E[e5_idx] * E[self.ent_num + 4]
        e6 = E[e6_idx] * E[self.ent_num + 5]

        x = e1 * e2 * e3 * e4 * e5 * e6 * r

        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x

    def test(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        #   得分函数
        num = torch.tensor(self.ent_num, dtype=torch.long, device=self.device)
        r = self.R(r_idx)
        e1 = self.E(e1_idx) * self.E(num)
        e2 = self.E(e2_idx) * self.E(num + 1)
        e3 = self.E(e3_idx) * self.E(num + 2)
        e4 = self.E(e4_idx) * self.E(num + 3)
        e5 = self.E(e5_idx) * self.E(num + 4)
        e6 = self.E(e6_idx) * self.E(num + 5)

        x = e1 * e2 * e3 * e4 * e5 * e6 * r

        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x


class MP(MessagePassing):
    """
        MessagePassing
    """
    def __init__(self, device, emb_dim, **kwargs):
        super(self.__class__, self).__init__(flow='target_to_source', aggr='add')
        # step1
        # 定义网络参数等
        self.emb_dim = emb_dim
        self.drop = torch.nn.Dropout(0.2)
        self.bn = torch.nn.BatchNorm1d(emb_dim)
        self.act = torch.tanh
        self.ent_num = kwargs["ent_num"]

        self.w_rel = get_param((emb_dim, emb_dim))  # (100,100)

    def forward(self, hyperedge, edge_index, edge_type, E, R):

        # setp1
        # propagate信息传播过程，得到更新后的实体矩阵
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.in_norm = None
        x = self.propagate(self.edge_index, x=E, edge_type=self.edge_type,
                                rel_embed=R, edge_norm=self.in_norm, hyperedge=hyperedge, edge_indexs=edge_index,
                                ent_embed=E, source_index=self.edge_index[0])

        out = self.drop(x) * (1 / 2) + self.drop(E) * (1 / 2)
        out = self.bn(out)
        out = self.act(out)

        # step3
        # 更新关系嵌入矩阵
        out_r = torch.matmul(R, self.w_rel)

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

        out = statement * x_j

        return out

    # 更新聚合信息，得到最终输出
    def update(self, aggr_out):
        return aggr_out

    # @staticmethod   # 计算每个三元组的norm值,全部设置为1
    # def compute_norm():
    #     return 1

    def get_hyperedge_emb(self, ent_embed, rel_embed, edge_type, hyperedge, edge_indexs):


        x = rel_embed[hyperedge[:,0]]

        for i in range(1, hyperedge.shape[1]):
            x *= ent_embed[hyperedge[:,i]] * ent_embed[self.ent_num+i-1]

        return x

