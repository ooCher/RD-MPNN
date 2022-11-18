# coding=utf-8
import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

from torch_scatter import scatter_add, scatter_max

def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def softmax(src, index, num_nodes=None):
    """

    :param src:     注意力值
    :param index:   头实体
    :param num_nodes:   实体个数
    :return:
    """
    # 得到实体个数
    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out

