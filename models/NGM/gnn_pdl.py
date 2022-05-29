import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from src.lap_solvers_pdl.sinkhorn import Sinkhorn

from collections import Iterable


class GNNLayer(nn.Layer):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features,
                 sk_channel=0, sk_iter=20, sk_tau=0.05, edge_emb=False):
        super(GNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.sk = Sinkhorn(sk_iter, sk_tau)
            self.classifier = nn.Linear(self.out_nfeat, self.sk_channel, bias_attr=True)
        else:
            self.out_nfeat = out_node_features
            self.sk = self.classifier = None

        if edge_emb:
            self.e_func = nn.Sequential(
                nn.Linear(self.in_efeat + self.in_nfeat, self.out_efeat, bias_attr=True),
                nn.ReLU(),
                nn.Linear(self.out_efeat, self.out_efeat, bias_attr=True),
                nn.ReLU()
            )
        else:
            self.e_func = None

        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat, bias_attr=True),
            # nn.Linear(self.in_nfeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat, bias_attr=True),
            # nn.Linear(self.out_nfeat // self.out_efeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
        )

        self.n_self_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat, bias_attr=True),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat, bias_attr=True),
            nn.ReLU()
        )

    def forward(self, A, W, x, n1=None, n2=None, norm=True):
        """
        :param A: adjacent matrix in 0/1 (b x n x n)
        :param W: edge feature tensor (b x n x n x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        if self.e_func is not None:
            W1 = paddle.multiply(A.unsqueeze(-1), x.unsqueeze(1))
            W2 = paddle.concat((W, W1), axis=-1)
            W_new = self.e_func(W2)
        else:
            W_new = W

        if norm is True:
            A = F.normalize(A, p=1, axis=2)

        x1 = self.n_func(x)
        x2 = paddle.matmul((A.unsqueeze(-1) * W_new).transpose((0, 3, 1, 2)), x1.unsqueeze(2).transpose((0, 3, 1, 2))). \
            squeeze(-1).transpose((0, 2, 1))
        x2 += self.n_self_func(x)

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(x2)
            n1_rep = paddle.to_tensor(np.repeat(n1, self.sk_channel, axis=0))
            n2_rep = paddle.to_tensor(np.repeat(n2, self.sk_channel, axis=0))
            x4 = x3.transpose((0, 2, 1)).reshape((x.shape[0] * self.sk_channel, n2.max(), n1.max())).transpose((0, 2, 1))
            x5 = self.sk(x4, n1_rep, n2_rep, dummy_row=True).transpose((0, 2, 1))

            x6 = x5.reshape((x.shape[0], self.sk_channel, n1.max() * n2.max())).transpose((0, 2, 1))
            x_new = paddle.concat((x2, x6), axis=-1)
        else:
            x_new = x2

        return W_new, x_new


class HyperGNNLayer(nn.Layer):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features, orders=3, eps=1e-10,
                 sk_channel=False, sk_iter=20, sk_tau=0.05):
        super(HyperGNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.eps = eps
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.sk = Sinkhorn(sk_iter, sk_tau)
            self.classifier = nn.Linear(self.out_nfeat, self.sk_channel, bias_attr=True)
        else:
            self.out_nfeat = out_node_features
            self.sk = self.classifier = None

        # used by forward_dense
        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat, bias_attr=True),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat, bias_attr=True),
            nn.ReLU(),
        )

        self.n_self_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat, bias_attr=True),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat, bias_attr=True),
            nn.ReLU()
        )

    def forward(self, A, W, x, n1=None, n2=None, weight=None, norm=True):
        """wrapper function of forward (support dense/sparse)"""
        if not isinstance(A, Iterable):
            A = [A]
            W = [W]

        W_new = []
        if weight is None:
            weight = [1.] * len(A)
        assert len(weight) == len(A)
        x2 = None
        for i, (_A, _W, w) in enumerate(zip(A, W, weight)):
            _W_new, _x = self.forward_dense(_A, _W, x, norm)
            if i == 0:
                x2 = _x * w
            else:
                x2 += _x * w
            W_new.append(_W_new)

        x2 += self.n_self_func(x)

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(x2)
            n1_rep = paddle.to_tensor(np.repeat(n1, self.sk_channel, axis=0))
            n2_rep = paddle.to_tensor(np.repeat(n2, self.sk_channel, axis=0))
            x4 = x3.transpose((0, 2, 1)).reshape((x.shape[0] * self.sk_channel, n2.max(), n1.max())).transpose((0, 2, 1))
            x5 = self.sk(x4, n1_rep, n2_rep, dummy_row=True).transpose((0, 2, 1))
            x6 = x5.reshape((x.shape[0], self.sk_channel, n1.max() * n2.max())).transpose((0, 2, 1))
            x_new = paddle.concat((x2, x6), axis=-1)
        else:
            x_new = x2

        return W_new, x_new

    def forward_dense(self, A, W, x, norm=True):
        """
        :param A: adjacent tensor in 0/1 (b x {n x ... x n})
        :param W: edge feature tensor (b x {n x ... x n} x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        order = len(A.shape) - 1
        W_new = W

        if norm is True:
            A_sum = paddle.sum(A, axis=tuple(range(2, order + 1)), keepdim=True)
            A = A / A_sum.expand_as(A)
            A[paddle.isnan(A)] = 0

        x1 = self.n_func(x)

        x_new = paddle.multiply(A.unsqueeze(-1), W_new)
        for i in range(order - 1):
            x1_shape = [x1.shape[0]] + [1] * (order - 1 - i) + list(x1.shape[1:])
            x_new = paddle.sum(paddle.multiply(x_new, x1.view(*x1_shape)), axis=-2)

        return W_new, x_new


class HyperConvLayer(nn.Layer):
    """
    Hypergarph convolutional layer inspired by "Dynamic Hypergraph Neural Networks"
    """

    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features, eps=0.0001,
                 sk_channel=False, sk_iter=20, voting_alpha=20):
        super(HyperConvLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.eps = eps
        if sk_channel:
            assert out_node_features == out_edge_features + 1
            self.out_nfeat = out_node_features - 1
            self.sk = Sinkhorn(sk_iter, 1 / voting_alpha)
            self.classifier = nn.Linear(self.out_efeat, 1, bias_attr=True)
        else:
            assert out_node_features == out_edge_features
            self.out_nfeat = out_node_features
            self.sk = self.classifier = None

        self.ne_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_efeat, bias_attr=True),
            nn.ReLU())

        self.e_func = nn.Sequential(
            nn.Linear(self.out_efeat + self.in_efeat, self.out_efeat, bias_attr=True),
            nn.ReLU()
        )

        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat + self.out_efeat, self.out_nfeat, bias_attr=True),
            nn.ReLU()
        )

    def forward(self, H, E, x, n1=None, n2=None, norm=None):
        """
        :param H: connectivity (b x n x e)
        :param E: (hyper)edge feature (b x e x f)
        :param x: node feature (b x n x f)
        :param n1: number of nodes in graph1
        :param n2: number of nodes in graph2
        :param norm: do normalization (only supports dense tensor)
        :return: new edge feature, new node feature
        """
        H_node_sum = paddle.sum(H, axis=1, keepdim=True)
        H_node_norm = H / H_node_sum
        H_node_norm[paddle.isnan(H_node_norm)] = 0
        H_edge_sum = paddle.sum(H, axis=2, keepdim=True)
        H_edge_norm = H / H_edge_sum
        H_edge_norm[paddle.isnan(H_edge_norm)] = 0

        x_to_E = paddle.matmul(H_node_norm.transpose((0, 2, 1)), self.ne_func(x))
        new_E = self.e_func(paddle.concat((x_to_E, E), axis=-1))
        E_to_x = paddle.matmul(H_edge_norm, new_E)
        new_x = self.n_func(paddle.concat((E_to_x, x), axis=-1))

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(new_x)
            x5 = self.sk(paddle.reshape(x3, (x.shape[0], n2.max(), n1.max())).transpose((0, 2, 1)), n1, n2, dummy_row=
            True).transpose((0, 2, 1)).contiguous()
            new_x = paddle.concat((new_x, paddle.reshape(x5, (x.shape[0], n1.max() * n2.max(), -1))), axis=-1)

        return new_E, new_x
