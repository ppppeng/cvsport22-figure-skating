import numpy as np
import paddle
import paddle.nn as nn
from ..registry import BACKBONES

import sys

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


class Graph():
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'fsd10':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                             (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                             (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                             (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                             (21, 14), (19, 14), (20, 19)]
            self.edge = self_link + neighbor_link
            self.center = 8
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                              (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                              (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                              (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                              (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


class TCN(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1):
        super(TCN, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Temporal_Block(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branche1 = nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU(),
            TCN(branch_channels, branch_channels, kernel_size=kernel_size[0], stride=stride, dilation=dilations[0]),
        )
        self.branche2 = nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU(),
            TCN(branch_channels, branch_channels, kernel_size=kernel_size[1], stride=stride, dilation=dilations[1]),
        )


        # Additional Max & 1x1 branch
        self.branche3 = nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2D(branch_channels)  # 为什么还要加bn
        )

        self.branche4 = nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2D(branch_channels)
        )

        # Residual connection
        if not residual:
            self.res = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.res = lambda x: x
        else:
            self.res = TCN(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

            # initialize
            # self.apply(weights_init)#todo 初始化

    def forward(self, x):
        # Input dim: (N,C,T,V)
        branch_outs = []
        branch_outs.append(self.branche1(x))
        branch_outs.append(self.branche2(x))
        branch_outs.append(self.branche3(x))
        branch_outs.append(self.branche4(x))
        out = paddle.concat(branch_outs, axis=1)
        out += self.res(x)
        return out


class CTRGC(nn.Layer):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO in_channels
        if in_channels <= 4 or in_channels == 9:
            self.rel_channels = 8
            # self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            # self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2D(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2D(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2D(self.rel_channels, self.out_channels, kernel_size=1)
        self.shortcut = nn.Conv2D(self.in_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()


    def forward(self, x, A=None, alpha=1):
        x1 = self.conv1(x)
        x1 = x1.mean(-2)
        x2 = self.conv2(x).mean(-2)
        y = self.shortcut(x)
        x = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x = self.conv3(x) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x = paddle.einsum('ncuv,nctv->nctu', x, y)

        return x


class Spatial_Block(nn.Layer):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(Spatial_Block, self).__init__()

        self.adaptive = adaptive
        self.num_subset = A.shape[0]

        # three graph

        # self.convs = nn.Sequential(  # hardcode instead of num_subset
        #     CTRGC(in_channels, out_channels),
        #     CTRGC(in_channels, out_channels),
        #     CTRGC(in_channels, out_channels)
        # )
        # graphA = paddle.create_parameter([25, 25], dtype='float32',
        #                                  default_initializer=paddle.nn.initializer.Assign(A[0]))
        # graphB = paddle.create_parameter([25, 25], dtype='float32',
        #                                  default_initializer=paddle.nn.initializer.Assign(A[1]))
        # graphC = paddle.create_parameter([25, 25], dtype='float32',
        #                                  default_initializer=paddle.nn.initializer.Assign(A[2]))
        # self.add_parameter(name='graphA', parameter=graphA)
        # self.add_parameter(name='graphB', parameter=graphB)
        # self.add_parameter(name='graphC', parameter=graphC)
        # alpha = paddle.create_parameter([1], dtype='float32')
        # self.add_parameter(name='alpha', parameter=alpha)
        # end

        # one graph
        self.conv = CTRGC(in_channels, out_channels)
        A = A.sum(axis=0)
        graph = paddle.create_parameter([25, 25], dtype='float32',
                                        default_initializer=paddle.nn.initializer.Assign(A))
        self.add_parameter(name='graph', parameter=graph)
        self.alpha = 1
        # end

        # todo learn alpha or not

        self.bn = nn.BatchNorm2D(out_channels)
        # self.soft = nn.Softmax(-2)#todo unused
        self.relu = nn.ReLU()

        if residual:
            if in_channels != out_channels:
                self.res = nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 1),
                    nn.BatchNorm2D(out_channels)
                )
            else:
                self.res = lambda x: x
        else:
            self.res = lambda x: 0

    def forward(self, x):

        # three graph
        # y = self.convs[0](x, self.graphA, self.alpha)
        # y = y + self.convs[1](x, self.graphB, self.alpha)
        # y = y + self.convs[2](x, self.graphC, self.alpha)
        # end

        # one graph
        y = self.conv(x, self.graph, self.alpha)
        # end

        y = self.bn(y)
        y += self.res(x)
        y = self.relu(y)
        return y


class ST_Block(nn.Layer):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=[9, 9],
                 dilations=[1, 2]):
        super(ST_Block, self).__init__()
        self.spatial_model = Spatial_Block(in_channels, out_channels, A, adaptive=adaptive)
        self.temporal_model = Temporal_Block(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                             dilations=dilations,
                                             residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.res = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.res = lambda x: x
        else:
            self.res = TCN(in_channels, out_channels, kernel_size=1, stride=stride)  # todo

    def forward(self, x):
        y = self.relu(self.temporal_model(self.spatial_model(x)) + self.res(x))
        return y


@BACKBONES.register()
class CTRGCN(nn.Layer):
    '''
    Channel-wise Topology Refinement Graph Convolution Network
    '''

    # todo test code in_channels=2
    def __init__(self, in_channels=2, adaptive=True, **kwargs):
        super(CTRGCN, self).__init__()

        self.graph = Graph(
            layout='fsd10',
            strategy='spatial',
        )
        A = paddle.to_tensor(self.graph.A, dtype='float32')
        base_channel = 64

        self.ctrgcn = nn.Sequential(
            ST_Block(in_channels, base_channel, A, residual=False, adaptive=adaptive, **kwargs),
            ST_Block(base_channel, base_channel, A, adaptive=adaptive, **kwargs),
            ST_Block(base_channel, base_channel, A, adaptive=adaptive, **kwargs),
            ST_Block(base_channel, base_channel, A, adaptive=adaptive, **kwargs),
            ST_Block(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive, **kwargs),
            ST_Block(base_channel * 2, base_channel * 2, A, adaptive=adaptive, **kwargs),
            ST_Block(base_channel * 2, base_channel * 2, A, adaptive=adaptive, **kwargs),
            ST_Block(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive, **kwargs),
            ST_Block(base_channel * 4, base_channel * 4, A, adaptive=adaptive, **kwargs),
            ST_Block(base_channel * 4, base_channel * 4, A, adaptive=adaptive, **kwargs),

            ST_Block(base_channel * 4, base_channel * 8, A, stride=2, adaptive=adaptive, **kwargs),
            ST_Block(base_channel * 8, base_channel * 8, A, adaptive=adaptive, **kwargs),
            ST_Block(base_channel * 8, base_channel * 8, A, adaptive=adaptive, **kwargs),
        )
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # data normalization

        N, C, T, V, M = x.shape
        x = x.transpose((0, 4, 1, 2, 3))  # N, M, C, T, V
        x = x.reshape((N * M, C, T, V))
        x = self.ctrgcn(x)
        # x = self.dropout(x)

        # x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
        C = x.shape[1]
        # x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1

        x = paddle.reshape(x, (N, M, C, -1))
        x = x.mean(3).mean(1).unsqueeze(-1).unsqueeze(-1)
        # x = self.dropout(x)


        return x
