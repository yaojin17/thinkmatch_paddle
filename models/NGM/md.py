import paddle
import numpy as np
import paddle.nn as nn

from src.lap_solvers_pdl.sinkhorn import Sinkhorn, GumbelSinkhorn
from src.utils_pdl.build_graphs import reshape_edge_feature
from src.utils_pdl.feature_align import feature_align
from src.utils_pdl.factorize_graph_matching import construct_aff_mat_dense
from models.NGM.gnn_pdl import GNNLayer
from models.NGM.geo_edge_feature_pdl import geo_edge_feature
from models.GMN.affinity_layer_pdl import GaussianAffinity
from models.GMN.affinity_layer_pdl import Affinity as InnerpAffinity
from src.utils_pdl.evaluation_metric import objective_score
from src.lap_solvers_pdl.hungarian import hungarian
import math

from src.utils.config import cfg

import src.utils_pdl.backbone
CNN = eval(f'src.utils_pdl.backbone.{cfg.BACKBONE}')


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()

        if cfg.NGM.EDGE_FEATURE == 'cat':
            self.affinity_layer = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            self.affinity_layer = GaussianAffinity(1, cfg.NGM.GAUSSIAN_SIGMA)
        else:
            raise ValueError(
                f'Unknown edge feature type {cfg.NGM.EDGE_FEATURE}')

        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE

        self.sinkhorn = Sinkhorn(
            max_iter=cfg.NGM.SK_ITER_NUM,
            tau=self.tau,
            epsilon=cfg.NGM.SK_EPSILON)
        self.gumbel_sinkhorn = GumbelSinkhorn(
            max_iter=cfg.NGM.SK_ITER_NUM,
            tau=self.tau * 10,
            epsilon=cfg.NGM.SK_EPSILON,)
        # batched_operation=True)
        # TODO: Paddle impl currently does not support batched_operation
        self.l2norm = nn.LocalResponseNorm(
            cfg.NGM.FEATURE_CHANNEL * 2,
            alpha=cfg.NGM.FEATURE_CHANNEL * 2,
            beta=0.5,
            k=0)

        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            tau = cfg.NGM.SK_TAU
            if i == 0:
                gnn_layer = GNNLayer(
                    1,
                    1,
                    cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0),
                    cfg.NGM.GNN_FEAT[i],
                    sk_channel=cfg.NGM.SK_EMB,
                    sk_tau=tau,
                    edge_emb=cfg.NGM.EDGE_EMB)
            else:
                gnn_layer = GNNLayer(
                    cfg.NGM.GNN_FEAT[i - 1] + (1 if cfg.NGM.SK_EMB else 0),
                    cfg.NGM.GNN_FEAT[i - 1],
                    cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0),
                    cfg.NGM.GNN_FEAT[i],
                    sk_channel=cfg.NGM.SK_EMB,
                    sk_tau=tau,
                    edge_emb=cfg.NGM.EDGE_EMB)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0), 1)

    def add_module(self, key, value):
        """Workaround for paddle.
        The behaviour of this function is the same as
        `torch.nn.Module.add_module()`
        """
        setattr(self, key, value)

    def forward(self, data_dict, **kwargs):
        batch_size = data_dict['batch_size']
        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']

            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, self.rescale)

        elif 'features' in data_dict:
            # synthetic data
            src, tgt = data_dict['features']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]

        elif 'aff_mat' in data_dict:
            K = data_dict['aff_mat']
            ns_src, ns_tgt = data_dict['ns']

        else:
            raise ValueError('Unknown data type for this model.')

        if 'images' in data_dict or 'features' in data_dict:
            tgt_len = P_tgt.shape[1]

            if cfg.NGM.EDGE_FEATURE == 'cat':
                X = reshape_edge_feature(F_src, G_src, H_src)
                Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)
            elif cfg.NGM.EDGE_FEATURE == 'geo':
                X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
                Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
            else:
                raise ValueError(
                    f'Unknown edge feature type {cfg.NGM.EDGE_FEATURE}')

            # affinity layer
            Ke, Kp = self.affinity_layer(X, Y, U_src, U_tgt)

            K = construct_aff_mat_dense(Ke, paddle.zeros_like(Kp), K_G, K_H)

            A = paddle.cast(K > 0, K.dtype)

            if cfg.NGM.FIRST_ORDER:
                emb = Kp.transpose((0, 2, 1)).reshape((Kp.shape[0], -1, 1))
            else:
                emb = paddle.ones((K.shape[0], K.shape[1], 1))
        else:
            tgt_len = int(math.sqrt(K.shape[2]))
            dmax = (paddle.max(paddle.sum(K, axis=2, keepdim=True), axis=1, keepdim=True) + 1e-5)
            K = K / dmax * 1000
            A = paddle.cast(K > 0, K.dtype)
            emb = paddle.ones((K.shape[0], K.shape[1], 1))

        emb_K = K.unsqueeze(-1)

        # NGM qap solver
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_K, emb = gnn_layer(A, emb_K, emb, ns_src, ns_tgt)

        v = self.classifier(emb)
        s = v\
            .reshape((v.shape[0], tgt_len, -1))\
            .transpose((0, 2, 1))

        if self.training or cfg.NGM.GUMBEL_SK <= 0:
            ss = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
            x = hungarian(ss, ns_src, ns_tgt)
        else:
            gumbel_sample_num = cfg.NGM.GUMBEL_SK
            if self.training:
                gumbel_sample_num //= 10
            ss_gumbel = self.gumbel_sinkhorn(
                s,
                ns_src,
                ns_tgt,
                sample_num=gumbel_sample_num,
                dummy_row=True)

            # FIXME: workaround with np. Does NOT support backward propagation
            def repeat_(x, rep_num):
                return paddle.to_tensor(np.repeat(x, rep_num, axis=0))

            if not self.training:
                ss_gumbel = hungarian(ss_gumbel, repeat_(ns_src), repeat_(ns_tgt))
            ss_gumbel = paddle.reshape(
                ss_gumbel,
                (batch_size, gumbel_sample_num, ss_gumbel.shape[-2], ss_gumbel.shape[-1]))

            # TODO: Assumes GPU memory is enough during inference
            # if ss_gumbel.place == 'cuda':
            #     dev_idx = ss_gumbel.device.index
            #     free_mem = gpu_free_memory(dev_idx) - 100 * 1024 ** 2 # 100MB as buffer for other computations
            #     K_mem_size = K.element_size() * K.nelement()
            #     max_repeats = free_mem // K_mem_size
            #     if max_repeats <= 0:
            #         print('Warning: GPU may not have enough memory')
            #         max_repeats = 1
            # else:
            max_repeats = gumbel_sample_num

            obj_score = []
            for idx in range(0, gumbel_sample_num, max_repeats):
                if idx + max_repeats > gumbel_sample_num:
                    rep_num = gumbel_sample_num - idx
                else:
                    rep_num = max_repeats
                obj_score.append(
                    objective_score(
                        paddle.reshape(
                            ss_gumbel[:, idx:(idx+rep_num), :, :],
                            (-1, ss_gumbel.shape[-2], ss_gumbel.shape[-1])),
                        repeat_(K, rep_num)
                    ).reshape(batch_size, -1)
                )
            obj_score = paddle.concat(obj_score, axis=1)
            min_obj_score = paddle.argmin(obj_score, axis=1)
            ss = ss_gumbel[paddle.arange(batch_size), min_obj_score.indices, :, :]
            x = hungarian(ss, repeat_(ns_src), repeat_(ns_tgt))

        data_dict.update({
            'ds_mat': ss,
            'perm_mat': x,
            'aff_mat': K
        })
        return data_dict