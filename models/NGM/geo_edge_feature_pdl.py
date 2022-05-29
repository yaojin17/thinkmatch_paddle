import paddle
from paddle import Tensor
from src.utils_pdl.pdl_device_trans import place2int


def geo_edge_feature(P: Tensor, G: Tensor, H: Tensor, norm_d=256, device=None):
    """
    Compute geometric edge features [d, cos(theta), sin(theta)]
    Adjacency matrix is formed by A = G * H^T
    :param P: point set (b x num_nodes x 2)
    :param G: factorized graph partition G (b x num_nodes x num_edges)
    :param H: factorized graph partition H (b x num_nodes x num_edges)
    :param norm_d: normalize Euclidean distance by norm_d
    :param device: device
    :return: feature tensor (b x 3 x num_edges)
    """
    if device is None:
        device = P.place

    p1 = paddle.sum(paddle.multiply(P.unsqueeze(-2), G.unsqueeze(-1)), axis=1)  # (b x num_edges x dim)
    p2 = paddle.sum(paddle.multiply(P.unsqueeze(-2), H.unsqueeze(-1)), axis=1)

    d = paddle.norm((p1 - p2) / (norm_d * paddle.sum(G, axis=1)).unsqueeze(-1), axis=-1)  # (b x num_edges)
    # non-existing elements are nan

    cos_theta = (p1[:, :, 0] - p2[:, :, 0]) / (d * norm_d)  # non-existing elements are nan
    sin_theta = (p1[:, :, 1] - p2[:, :, 1]) / (d * norm_d)
    # paddle.set_device("gpu")
    res = paddle.stack((d, cos_theta, sin_theta), axis=1)
    return paddle.to_tensor(res, place=device)
