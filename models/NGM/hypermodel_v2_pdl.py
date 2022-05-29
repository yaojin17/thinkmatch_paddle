import paddle
import itertools
from models.BBGM.affinity_layer_pdl import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import SiameseSConvOnNodes
from models.BBGM.sconv_archs import SiameseNodeFeaturesToEdgeFeatures
from src.utils_pdl.feature_align import feature_align
from src.utils_pdl.factorize_graph_matching import construct_aff_mat_dense
from models.NGM.gnn_pdl import HyperGNNLayer
from src.lap_solvers_pdl.sinkhorn import Sinkhorn
from src.lap_solvers_pdl.hungarian import hungarian
from src.utils_pdl.pad_tensor import pad_tensor
from src.utils_pdl.config import cfg
from src.utils_pdl.backbone import *

CNN = eval(cfg.BACKBONE)


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = paddle.norm(x, axis=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = paddle.concat([embedding[:, :num_v] for embedding, num_v in
                         zip(embeddings, num_vertices)], axis=-1)
    return res.transpose((1, 0))


def construct_hyperE(ori_graphs, batch_size, device):
    nmax = max([g.num_nodes for g in ori_graphs])
    emax = max([g.hyperedge_index.shape[1] for g in ori_graphs])
    hyperE = paddle.to_tensor(paddle.zeros([batch_size, nmax, nmax, nmax, emax]), stop_gradient=True)
    for b, g in enumerate(ori_graphs):
        a = paddle.to_tensor(paddle.arange(end=g.hyperedge_index.shape[1]), stop_gradient=True)
        hyperE[b][g.hyperedge_index[0], g.hyperedge_index[1], g.
            hyperedge_index[2], a] = 1
    return hyperE, nmax, emax


def hyperedge_affinity(attrs1, attrs2):
    ret_list = []
    for attr1, attr2 in zip(attrs1, attrs2):
        X = attr1.unsqueeze(1).expand(attr1.shape[0], attr2.shape[0], attr1
                                      .shape[1])
        Y = attr2.unsqueeze(0).expand(attr1.shape[0], attr2.shape[0], attr2
                                      .shape[1])
        dist = paddle.sum(paddle.pow(X - Y, 2), axis=-1)
        dist[paddle.isnan(dist)] = float('inf')
        ret_list.append(paddle.exp(-dist / cfg.NGM.SIGMA3))
    return ret_list


class Net(CNN):

    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim
                                                              =cfg.NGM.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = (
            SiameseNodeFeaturesToEdgeFeatures(total_num_nodes=self.
                                              message_pass_node_features.num_node_features))
        self.global_state_dim = cfg.NGM.FEATURE_CHANNEL * 2
        self.vertex_affinity = InnerProductWithWeightsAffinity(self.
                                                               global_state_dim,
                                                               self.message_pass_node_features.num_node_features
                                                               )
        self.edge_affinity = InnerProductWithWeightsAffinity(self.
                                                             global_state_dim,
                                                             self.build_edge_features_from_node_features.
                                                             num_edge_features)
        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.NGM.SK_TAU
        self.univ_size = cfg.NGM.UNIV_SIZE
        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau,
                                 epsilon=cfg.NGM.SK_EPSILON)
        self.sinkhorn_mgm = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, epsilon= \
            cfg.NGM.SK_EPSILON, tau=1.0)
        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = HyperGNNLayer(1, 1, cfg.NGM.GNN_FEAT[i] + cfg.
                                          NGM.SK_EMB, cfg.NGM.GNN_FEAT[i], sk_channel=cfg.NGM.
                                          SK_EMB, sk_tau=self.tau)
            else:
                gnn_layer = HyperGNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM
                                          .SK_EMB, cfg.NGM.GNN_FEAT[i - 1], cfg.NGM.GNN_FEAT[i] +
                                          cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i], sk_channel=cfg.NGM
                                          .SK_EMB, sk_tau=self.tau)
            self.add_sublayer('gnn_layer_{}'.format(i), gnn_layer)
        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1, bias_attr=True)

    def forward(self, data_dict):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['batch_size']
        num_graphs = len(images)
        if cfg.PROBLEM.TYPE == '2GM' and 'gt_perm_mat' in data_dict:
            gt_perm_mats = [data_dict['gt_perm_mat']]
        elif cfg.PROBLEM.TYPE == 'MGM' and 'gt_perm_mat' in data_dict:
            perm_mat_list = data_dict['gt_perm_mat']
            gt_perm_mats = [paddle.matmul(pm_src, pm_tgt.transpose((0, 2, 1))) for
                            pm_src, pm_tgt in lexico_iter(perm_mat_list)]
        else:
            raise ValueError(
                'Ground truth information is required during training.')
        global_list = []
        orig_graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)
            global_list.append(self.final_layers(edges).reshape((nodes.
                                                                 shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p
                                )
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p
                                )
            node_features = paddle.concat((U, F), axis=1)
            graph.x = node_features
            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph,
                                                                     hyperedge=True)
            orig_graph_list.append(orig_graph)
        global_weights_list = [paddle.concat([global_src, global_tgt],
                                             axis=-1) for global_src, global_tgt in lexico_iter(global_list)]
        global_weights_list = [normalize_over_channels(g) for g in
                               global_weights_list]
        unary_affs_list = [self.vertex_affinity([item.x for item in g_1], [
            item.x for item in g_2], global_weights) for (g_1, g_2),
                                                         global_weights in zip(lexico_iter(orig_graph_list),
                                                                               global_weights_list)]
        quadratic_affs_list = [self.edge_affinity([item.edge_attr for item in
                                                   g_1], [item.edge_attr for item in g_2], global_weights) for (
                                                                                                                   g_1,
                                                                                                                   g_2), global_weights
                               in zip(lexico_iter(orig_graph_list),
                                      global_weights_list)]
        quadratic_affs_list = [[(0.5 * x) for x in quadratic_affs] for
                               quadratic_affs in quadratic_affs_list]
        order3_affs_list = [hyperedge_affinity([item.hyperedge_attr for
                                                item in g_1], [item.hyperedge_attr for item in g_2]) for (g_1,
                                                                                                          g_2), global_weights
                            in zip(lexico_iter(orig_graph_list),
                                   global_weights_list)]
        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []
        for unary_affs, quadratic_affs, order3_affs, (g1, g2), (idx1, idx2
                                                                ) in zip(unary_affs_list, quadratic_affs_list,
                                                                         order3_affs_list,
                                                                         lexico_iter(orig_graph_list),
                                                                         lexico_iter(range(num_graphs))):
            kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict[
                'KGHs']['{},{}'.format(idx1, idx2)]
            Kp = paddle.stack(pad_tensor(unary_affs), axis=0)
            Ke = paddle.stack(pad_tensor(quadratic_affs), axis=0)
            He = paddle.stack(pad_tensor(order3_affs))
            K = construct_aff_mat_dense(Ke, Kp, kro_G, kro_H)
            hyperE1, nmax1, emax1 = construct_hyperE(g1, batch_size, He.device)
            hyperE2, nmax2, emax2 = construct_hyperE(g2, batch_size, He.device)
            H = paddle.matmul(paddle.matmul(hyperE1.reshape(batch_size, -1,
                                                            emax1), He), hyperE2.reshape(batch_size, -1, emax2).
                              transpose((0, 2, 1))).reshape(batch_size, nmax1, nmax1, nmax1,
                                                            nmax2, nmax2, nmax2).transpose((0, 4, 1, 5, 2, 6, 3)).reshape(
                batch_size, nmax1 * nmax2, nmax1 * nmax2, nmax1 * nmax2)
            if num_graphs == 2:
                data_dict['aff_mat'] = K
            if cfg.NGM.FIRST_ORDER:
                emb = paddle.reshape(Kp.transpose((0, 2, 1)), (Kp.shape[0], -1, 1))
            else:
                emb = paddle.to_tensor(paddle.ones([K.shape[0], K.shape[1], 1]),stop_gradient=True)

            if cfg.NGM.POSITIVE_EDGES:
                adjs = [(K > 0).to(K.dtype), (H > 0).to(H.dtype)]
            else:
                adjs = [(K != 0).to(K.dtype), (H != 0).to(H.dtype)]
            emb_edges = [K.unsqueeze(-1), H.unsqueeze(-1)]
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb_edges, emb = gnn_layer(adjs, emb_edges, emb, n_points[
                    idx1], n_points[idx2])
            v = self.classifier(emb)
            s = paddle.reshape(v, (v.shape[0], points[idx2].shape[1], -1)).transpose((0, 2, 1))
            ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row = True)
            x = hungarian(ss, n_points[idx1], n_points[idx2])
            s_list.append(ss)
            x_list.append(x)
            indices.append((idx1, idx2))
        if num_graphs > 2:
            a = paddle.to_tensor(paddle.zeros(1, dtype=paddle.int64), stop_gradient=True)
            joint_indices = paddle.concat((paddle.cumsum(paddle.stack
                                                         ([paddle.max(np) for np in n_points]), axis=0), a))
            joint_S = paddle.zeros([batch_size, paddle.max(
                joint_indices), paddle.max(joint_indices)])
            joint_S = paddle.to_tensor(joint_S, stop_gradient=True)
            for idx in range(num_graphs):
                for b in range(batch_size):
                    start = joint_indices[idx - 1]
                    joint_S[b, start:start + n_points[idx][b], start:start +
                                                                     n_points[idx][b]] += paddle.eye(n_points[idx][b],
                                                                                                     device=K.device)
            for (idx1, idx2), s in zip(indices, s_list):
                if idx1 > idx2:
                    joint_S[:, joint_indices[idx2 - 1]:joint_indices[idx2],
                    joint_indices[idx1 - 1]:joint_indices[idx1]
                    ] += s.transpose((0, 2, 1))
                else:
                    joint_S[:, joint_indices[idx1 - 1]:joint_indices[idx1],
                    joint_indices[idx2 - 1]:joint_indices[idx2]] += s
            matching_s = []
            for b in range(batch_size):
                e, v = paddle.linalg.eigh(joint_S[b])
                diff = e[-self.univ_size:-1] - e[-self.univ_size + 1:]
                if self.training and paddle.min(paddle.abs(diff)
                                                ) <= 0.0001:
                    matching_s.append(joint_S[b])
                else:
                    matching_s.append(num_graphs * paddle.mm(v[:, -self.
                                                             univ_size:], v[:, -self.univ_size:].transpose((1, 0))))
            matching_s = paddle.stack(matching_s, axis=0)
            for idx1, idx2 in indices:
                s = matching_s[:, joint_indices[idx1 - 1]:joint_indices[
                    idx1], joint_indices[idx2 - 1]:joint_indices[idx2]]
                s = self.sinkhorn_mgm(paddle.log(paddle.nn.functional.relu(
                    s)), n_points[idx1], n_points[idx2])
                x = hungarian(s, n_points[idx1], n_points[idx2])
                mgm_s_list.append(s)
                mgm_x_list.append(x)
        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({'ds_mat': s_list[0], 'perm_mat': x_list[0]})
        elif cfg.PROBLEM.TYPE == 'MGM':
            data_dict.update({'ds_mat_list': mgm_s_list, 'perm_mat_list':
                mgm_x_list, 'graph_indices': indices, 'gt_perm_mat_list':
                                  gt_perm_mats})
        return data_dict
