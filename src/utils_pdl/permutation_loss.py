import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .pdl_device_trans import place2int


class CrossEntropyLoss(nn.Layer):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.astype("float32")

        assert ((pred_perm.numpy() >= 0) * (pred_perm.numpy() <= 1)).all()
        assert ((gt_perm.numpy() >= 0) * (gt_perm.numpy() <= 1)).all()

        loss = paddle.to_tensor(0., place=pred_perm.place)
        n_sum = paddle.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :pred_ns[b], :gt_ns[b]],
                gt_perm[b, :pred_ns[b], :gt_ns[b]],
                reduction='sum')
            n_sum += pred_ns[b].astype(n_sum.dtype).cuda(device_id=place2int(pred_perm.place))

        return loss / n_sum
