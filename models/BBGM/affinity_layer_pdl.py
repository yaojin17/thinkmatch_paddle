import paddle
import paddle.nn as nn


class InnerProductWithWeightsAffinity(nn.Layer):
    def __init__(self, input_dim, output_dim):
        super(InnerProductWithWeightsAffinity, self).__init__()
        self.d = output_dim
        self.A = paddle.nn.Linear(input_dim, output_dim, bias_attr=True)

    def _forward(self, X, Y, weights):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        coefficients = paddle.tanh(self.A(weights))
        res = paddle.matmul(X * coefficients, Y.transpose((1, 0)))
        res = paddle.nn.functional.softplus(res) - 0.5
        return res

    def forward(self, Xs, Ys, Ws):
        return [self._forward(X, Y, W) for X, Y, W in zip(Xs, Ys, Ws)]
