import paddle
from typing import List
from paddle import Tensor


def construct_aff_mat_dense_larger(Ke: Tensor, Kp: Tensor, KroG: List[Tensor], KroH: List[Tensor]) -> Tensor:
    B, P, Q = Ke.shape
    _, N, M = Kp.shape
    # NOTE: Every single transpose here matters
    # because we should use a COLUMN VECTORIZATION here.
    res = paddle.zeros((B, M * N, M * N))
    for b in range(B):
        KroG_diag = paddle.matmul(
            KroG[b],
            paddle.diag(Ke[b].transpose((1, 0)).reshape([-1])))  # MN, PQ
        KroG_diag_KroH = paddle.matmul(
            KroG_diag,
            KroH[b].transpose((1, 0)))  # MN, MN
        res[b] = paddle.diag(
            Kp[b].transpose((1, 0)).reshape([-1])) + KroG_diag_KroH

    return res


def construct_aff_mat_dense(Ke: Tensor, Kp: Tensor, KroG: List[Tensor], KroH: List[Tensor]) -> Tensor:
    B, P, Q = Ke.shape
    _, N, M = Kp.shape
    res = paddle.zeros((B, M * N, M * N))
    KroG = paddle.stack(KroG)
    KroH = paddle.stack(KroH)
    diag_KroH = paddle.multiply(
        Ke.transpose((0, 2, 1)).reshape([B, -1, 1]),
        KroH.transpose((0, 2, 1)))
    KroG_diag_KroH = paddle.bmm(
        KroG,
        diag_KroH)  # B, MN, MN
    for b in range(B):
        res[b] = paddle.diag(
            Kp[b].transpose((1, 0)).reshape([-1])) + KroG_diag_KroH[b]

    return res