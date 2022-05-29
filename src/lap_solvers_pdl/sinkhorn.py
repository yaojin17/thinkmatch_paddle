import numpy as np
import paddle
import paddle.nn as nn
from src.utils_pdl.pdl_device_trans import place2str


class Sinkhorn(nn.Layer):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """

    def __init__(self, max_iter=10, epsilon=1e-4, tau=0.05, log_forward=True):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tau = tau
        self.log_forward = log_forward
        if not log_forward:
            print('Warning: Sinkhorn algorithm not in log scale is deprecated since logrithm is more stable')

    def forward(self, *input, **kwinput):
        if self.log_forward:
            return self.forward_log(*input, **kwinput)
        else:
            return self.forward_ori(*input, **kwinput)  # deprecated

    def forward_log(self, s, nrows=None, ncols=None, dummy_row=False, dtype=paddle.float32):
        # global function that sets all tensors' device to the device of "s"
        device_str = place2str(s.place)
        paddle.set_device(device_str)
        # computing sinkhorn with row/column normalization in the log space.
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if s.shape[2] >= s.shape[1]:
            transposed = False
        else:
            s = s.transpose((0, 2, 1))
            transposed = True

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # operations are performed on log_s
        s = s / self.tau
        dummy_shape = None
        ori_nrows = None
        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            s = paddle.concat((s, paddle.full(dummy_shape, -float('inf')).cuda()), axis=1)
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                s[b, nrows[b]:, :] = -float('inf')
                s[b, :, ncols[b]:] = -float('inf')

        ret_log_s = paddle.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), dtype=s.dtype).cuda()
        ret_log_s.stop_gradient = False

        for b in range(batch_size):
            row_slice = slice(0, int(nrows[b]))
            col_slice = slice(0, int(ncols[b]))
            log_s = s[b, row_slice, col_slice]

            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = paddle.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum
                else:
                    log_sum = paddle.logsumexp(log_s, 0, keepdim=True)
                    log_s = log_s - log_sum

            ret_log_s[b, row_slice, col_slice] = log_s

        if dummy_row:
            if dummy_shape[1] > 0:
                ret_log_s = ret_log_s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

        if transposed:
            ret_log_s = ret_log_s.transpose((0, 2, 1))
        if matrix_input:
            ret_log_s.squeeze_(0)

        return paddle.exp(ret_log_s)

    def forward_ori(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=paddle.float32):
        batch_size = s.shape[0]

        # global function that sets all tensors' device to the device of "s"
        device_str = place2str(s.place)
        paddle.set_device(device_str)
        dummy_shape = None
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            s = paddle.concat((s, paddle.full(dummy_shape, 0.).cuda()), axis=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        row_norm_ones = paddle.zeros((batch_size, s.shape[1], s.shape[1]))  # size: row x row
        col_norm_ones = paddle.zeros((batch_size, s.shape[2], s.shape[2]))  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, int(nrows[b]) if nrows is not None else int(s.shape[2]))
            col_slice = slice(0, int(ncols[b]) if ncols is not None else int(s.shape[1]))
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s += self.epsilon

        for i in range(self.max_iter):
            if exp:
                s = paddle.exp(exp_alpha * s)
            if i % 2 == 1:
                # column norm
                sum = paddle.sum(paddle.multiply(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), axis=2)
            else:
                # row norm
                sum = paddle.sum(paddle.multiply(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), axis=2)

            tmp = paddle.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, int(nrows[b]) if nrows is not None else s.shape[2])
                col_slice = slice(0, int(ncols[b]) if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row and dummy_shape[1] > 0:
            s = s[:, :-dummy_shape[1]]

        return s


class GumbelSinkhorn(nn.Layer):
    """
    GumbelSinkhorn Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """

    def __init__(self, max_iter=10, tau=1., epsilon=1e-4, log_forward=False):
        super(GumbelSinkhorn, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter, tau, epsilon, log_forward=log_forward)

    def forward(self, s, nrows=None, ncols=None, sample_num=5, dummy_row=False, dtype=paddle.float32):
        def sample_gumbel(t_like, eps=1e-20):
            """
            randomly sample standard gumbel variables
            """
            u = paddle.empty_like(t_like).uniform_(min=0)
            return -paddle.log(-paddle.log(u + eps) + eps)

        s_rep = paddle.to_tensor(np.repeat(s,sample_num,axis=0))
        s_rep = s_rep + sample_gumbel(s_rep)
        nrows_rep =paddle.to_tensor(np.repeat(nrows, sample_num, axis=0))
        ncols_rep = paddle.to_tensor(np.repeat(ncols, sample_num, axis=0))
        s_rep = self.sinkhorn(s_rep, nrows_rep, ncols_rep, dummy_row, dtype)
        return s_rep
