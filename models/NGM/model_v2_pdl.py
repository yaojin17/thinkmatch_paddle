import itertools

from models.BBGM.affinity_layer_pdl import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs_pdl import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from models.NGM.gnn_pdl import GNNLayer
from src.lap_solvers_pdl.hungarian import hungarian
from src.lap_solvers_pdl.sinkhorn import Sinkhorn
from src.utils_pdl.config import cfg
from src.utils_pdl.factorize_graph_matching import construct_aff_mat_dense
from src.utils.pad_tensor import pad_tensor
from src.utils_pdl.backbone import *
from src.utils_pdl.feature_align import feature_align

CNN = eval(cfg.BACKBONE)


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = paddle.norm(x, axis=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = paddle.concat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], axis=-1)
    return res.transpose((1,0))


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.NGM.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.NGM.FEATURE_CHANNEL * 2
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.NGM.SK_TAU
        self.mgm_tau = cfg.NGM.MGM_SK_TAU
        self.univ_size = cfg.NGM.UNIV_SIZE

        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.sinkhorn_mgm = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, epsilon=cfg.NGM.SK_EPSILON, tau=self.mgm_tau)
        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            tau = cfg.NGM.SK_TAU
            if i == 0:
                gnn_layer = GNNLayer(1, 1,
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
            else:
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
            self.add_sublayer('gnn_layer_{}'.format(i), gnn_layer)
        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)

    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['batch_size']
        num_graphs = len(images)

        global_list = []
        orig_graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = paddle.concat((U, F), axis=1)
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            paddle.concat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []

        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
            Kp = paddle.stack(pad_tensor(unary_affs), axis=0)
            Ke = paddle.stack(pad_tensor(quadratic_affs), axis=0)
            K = construct_aff_mat_dense(Ke, Kp, kro_G, kro_H)
            if num_graphs == 2: data_dict['aff_mat'] = K

            if cfg.NGM.FIRST_ORDER:
                emb = Kp.transpose((0,2,1)).contiguous().view(Kp.shape[0], -1, 1)
            else:
                emb = paddle.ones(K.shape[0], K.shape[1], 1)

            if cfg.NGM.POSITIVE_EDGES:
                A = (K > 0).to(K.dtype)
            else:
                A = (K != 0).to(K.dtype)

            emb_K = K.unsqueeze(-1)

            # NGM qap solver
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb_K, emb = gnn_layer(A, emb_K, emb, n_points[idx1], n_points[idx2])

            v = self.classifier(emb)
            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose((0,2,1))

            ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
            x = hungarian(ss, n_points[idx1], n_points[idx2])
            s_list.append(ss)
            x_list.append(x)
            indices.append((idx1, idx2))

        if num_graphs > 2:
            joint_indices = paddle.concat((paddle.cumsum(paddle.stack([paddle.max(np) for np in n_points]), axis=0), paddle.zeros((1,), dtype='int64')))
            joint_S = paddle.zeros(batch_size, paddle.max(joint_indices), paddle.max(joint_indices))
            for idx in range(num_graphs):
                for b in range(batch_size):
                    start = joint_indices[idx-1]
                    joint_S[b, start:start+n_points[idx][b], start:start+n_points[idx][b]] += paddle.eye(n_points[idx][b])

            for (idx1, idx2), s in zip(indices, s_list):
                if idx1 > idx2:
                    joint_S[:, joint_indices[idx2-1]:joint_indices[idx2], joint_indices[idx1-1]:joint_indices[idx1]] += s.transpose((0,2,1))
                else:
                    joint_S[:, joint_indices[idx1-1]:joint_indices[idx1], joint_indices[idx2-1]:joint_indices[idx2]] += s

            matching_s = []
            for b in range(batch_size):
                e, v = paddle.linalg.eigh(joint_S[b])
                diff = e[-self.univ_size:-1] - e[-self.univ_size+1:]
                if self.training and paddle.min(paddle.abs(diff)) <= 1e-4:
                    matching_s.append(joint_S[b])
                else:
                    matching_s.append(num_graphs * paddle.mm(v[:, -self.univ_size:], v[:, -self.univ_size:].transpose((1,0))))

            matching_s = paddle.stack(matching_s, axis=0)

            for idx1, idx2 in indices:
                s = matching_s[:, joint_indices[idx1-1]:joint_indices[idx1], joint_indices[idx2-1]:joint_indices[idx2]]
                s = self.sinkhorn_mgm(paddle.log(paddle.nn.functional.relu(s)), n_points[idx1], n_points[idx2]) # only perform row/col norm, do not perform exp
                x = hungarian(s, n_points[idx1], n_points[idx2])

                mgm_s_list.append(s)
                mgm_x_list.append(x)

        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0]
            })
        elif cfg.PROBLEM.TYPE == 'MGM':
            data_dict.update({
                'ds_mat_list': mgm_s_list,
                'perm_mat_list': mgm_x_list,
                'graph_indices': indices,
            })

        return data_dict
