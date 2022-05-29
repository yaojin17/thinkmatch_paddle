import paddle
import paddle.nn as nn
import math
import numpy as np
from numpy.random import uniform
from paddle import Tensor

class Affinity(nn.Layer):
    """ Affinity Layer to compute the affinity matrix via inner product from feature space.
    Me = X * Lambda * Y^T
    Mp = Ux * Uy^T
    Parameter: scale of weight d
    Input: edgewise (pairwise) feature X, Y
           pointwise (unary) feature Ux, Uy
    Output: edgewise affinity matrix Me
            pointwise affinity matrix Mp
    Weight: weight matrix Lambda = [[Lambda1, Lambda2],
                                    [Lambda2, Lambda1]]
            where Lambda1, Lambda2 > 0
    """

    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        # set parameters
        stdv = 1. / math.sqrt(self.d)
        tmp1 = uniform(low=-1 * stdv, high=stdv, size=(self.d, self.d))
        tmp2 = uniform(low=-1 * stdv, high=stdv, size=(self.d, self.d))
        tmp1 += np.eye(self.d) / 2.0
        tmp2 += np.eye(self.d) / 2.0

        paralambda1 = paddle.ParamAttr(initializer=nn.initializer.Assign(paddle.to_tensor(tmp1, dtype='float64')))
        paralambda2 = paddle.ParamAttr(initializer=nn.initializer.Assign(paddle.to_tensor(tmp2, dtype='float64')))
        lambda1 = self.create_parameter(
            [self.d, self.d],
            attr=paralambda1)
        lambda2 = self.create_parameter(
            [self.d, self.d],
            attr=paralambda2)
        self.add_parameter('lambda1', lambda1)
        self.add_parameter('lambda2', lambda2)
        self.relu = nn.ReLU()  # problem: if weight<0, then always grad=0. So this parameter is never updated!

    def forward(self, X, Y, Ux, Uy, w1=1, w2=1):
        assert X.shape[1] == Y.shape[1] == 2 * self.d
        lambda1 = self.relu(self.lambda1 + self.lambda1.transpose((1, 0))) * w1
        lambda2 = self.relu(self.lambda2 + self.lambda2.transpose((1, 0))) * w2
        weight = paddle.concat((paddle.concat((lambda1, lambda2)),
                                paddle.concat((lambda2, lambda1))), 1)
        Me = paddle.matmul(X.transpose((0, 2, 1)), weight)
        Me = paddle.matmul(Me, Y)
        Mp = paddle.matmul(Ux.transpose((0, 2, 1)), Uy)

        return Me, Mp


class GaussianAffinity(nn.Layer):
    """
    Affinity Layer to compute the affinity matrix via gaussian kernel from feature space.
    Me = exp(- L2(X, Y) / sigma)
    Mp = Ux * Uy^T
    Parameter: scale of weight d, gaussian kernel sigma
    Input: edgewise (pairwise) feature X, Y
           pointwise (unary) feature Ux, Uy
    Output: edgewise affinity matrix Me
            pointwise affinity matrix Mp
    """

    def __init__(self, d, sigma):
        super(GaussianAffinity, self).__init__()
        self.d = d
        self.sigma = sigma

    def forward(self, X, Y, Ux=None, Uy=None, ae=1., ap=1.):
        assert X.shape[1] == Y.shape[1] == self.d

        X = X.unsqueeze(-1).expand((*X.shape, Y.shape[2]))
        Y = Y.unsqueeze(-2).expand((*Y.shape[:2], X.shape[2], Y.shape[2]))
        # dist = torch.sum(torch.pow(torch.mul(X - Y, self.w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)), 2), dim=1)
        dist = paddle.sum(paddle.pow(X - Y, 2), axis=1)
        dist[paddle.isnan(dist)] = float("Inf")
        Me = paddle.exp(- dist / self.sigma) * ae

        if Ux is None or Uy is None:
            return Me
        else:
            Mp = paddle.matmul(Ux.transpose((0, 2, 1)), Uy) * ap
            return Me, Mp

