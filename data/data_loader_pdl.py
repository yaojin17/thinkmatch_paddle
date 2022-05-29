import random
import paddle
import numpy as np
import paddle.nn.functional as F
from paddle.vision import transforms
# from data.pascal_voc import PascalVOC
# from data.willow_obj import WillowObject
from paddle.io import Dataset, DataLoader
from src.utils_pdl.build_graphs import build_graphs
# NOTE: The sparse matrices will be used when computing Kronecker Product
#       currently we use a dense implementation as a workaround
from src.utils.fgm import kronecker_sparse
from src.sparse_torch import CSRMatrix3d

from src.utils.config import cfg


class GMDataset(Dataset):
    def __init__(self, name, bm, length, cls=None, **args):
        self.name = name
        self.bm = bm
        # NOTE images pairs are sampled randomly, so there is no exact definition of dataset size
        self.length = length
        # length here represents the iterations between two checkpoints
        self.obj_size = self.bm.obj_resize
        self.cls = None if cls == 'none' else cls

        if self.cls is None:
            self.classes = self.bm.classes
        else:
            self.classes = [self.cls]

        self.problem_type = '2GM'
        self.img_num_list = self.bm.compute_img_num(self.classes)

        self.id_combination, self.length = self.bm.get_id_combination(self.cls)
        self.length_list = []
        for cls in self.classes:
            cls_length = self.bm.compute_length(cls)
            self.length_list.append(cls_length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cls_num = random.randrange(0, len(self.classes))
        ids = list(self.id_combination[cls_num]
                   [idx % self.length_list[cls_num]])
        anno_pair, perm_mat_, id_list = self.bm.get_data(ids)
        perm_mat = perm_mat_[(0, 1)].toarray()
        while min(perm_mat.shape[0], perm_mat.shape[1]) <= 2 or perm_mat.size >= cfg.PROBLEM.MAX_PROB_SIZE > 0:
            anno_pair, perm_mat_, id_list = self.bm.rand_get_data(self.cls)
            perm_mat = perm_mat_[(0, 1)].toarray()

        cls = [anno['cls'] for anno in anno_pair]
        P1 = [(kp['x'], kp['y']) for kp in anno_pair[0]['kpts']]
        P2 = [(kp['x'], kp['y']) for kp in anno_pair[1]['kpts']]

        n1, n2 = len(P1), len(P2)
        univ_size = [anno['univ_size'] for anno in anno_pair]

        P1 = np.array(P1)
        P2 = np.array(P2)

        A1, G1, H1, e1 = build_graphs(
            P1, n1, stg=cfg.GRAPH.SRC_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)
        if cfg.GRAPH.TGT_GRAPH_CONSTRUCT == 'same':
            G2 = perm_mat.transpose().dot(G1)
            H2 = perm_mat.transpose().dot(H1)
            A2 = G2.dot(H2.transpose())
            e2 = e1
        else:
            A2, G2, H2, e2 = build_graphs(
                P2, n2, stg=cfg.GRAPH.TGT_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)

        # pyg_graph1 = self.to_pyg_graph(A1, P1)
        # pyg_graph2 = self.to_pyg_graph(A2, P2)

        ret_dict = {'Ps': [paddle.to_tensor(data=x, dtype='float32') for x in [P1, P2]],
                    'ns': [paddle.to_tensor(data=x, dtype='int64') for x in [n1, n2]],
                    'es': [paddle.to_tensor(data=x, dtype='float32') for x in [e1, e2]],
                    'gt_perm_mat': perm_mat,
                    'Gs': [paddle.to_tensor(data=x, dtype='float32') for x in [G1, G2]],
                    'Hs': [paddle.to_tensor(data=x, dtype='float32') for x in [H1, H2]],
                    'As': [paddle.to_tensor(data=x, dtype='float32') for x in [A1, A2]],
                    # 'pyg_graphs': [pyg_graph1, pyg_graph2],
                    'cls': [str(x) for x in cls],
                    'id_list': id_list,
                    'univ_size': [paddle.to_tensor(int(x)) for x in univ_size],
                    }

        imgs = [anno['img'] for anno in anno_pair]
        if imgs[0] is not None:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
            ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
        elif 'feat' in anno_pair[0]['kpts'][0]:
            feat1 = np.stack([kp['feat']
                              for kp in anno_pair[0]['kpts']], axis=-1)
            feat2 = np.stack([kp['feat']
                              for kp in anno_pair[1]['kpts']], axis=-1)
            ret_dict['features'] = [paddle.to_tensor(x) for x in [feat1, feat2]]

        return ret_dict


def collate_fn(data: list):
    """Create mini-batch data for training.

    Args:
        `data` (list): data dict

    Retruns:
        mini-batched data for training
    """

    def pad_tensor(inp):
        assert type(inp[0]) == paddle.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = pad_pattern.tolist()
            # print('pad_pattern is', pad_pattern)
            if (len(pad_pattern) == 2):
                tt = t.reshape((1, 1, t.shape[0]))
                padded_ts.append(
                    F.pad(tt, pad_pattern, 'constant', 0, 'NCL').squeeze())
            elif len(pad_pattern) == 4:
                tt = t.reshape((1, 1, t.shape[0], -1))
                tt = F.pad(tt, pad_pattern, 'constant', 0, 'NCHW')
                padded_ts.append(tt.reshape((1, tt.shape[2], -1)).squeeze())
            elif len(pad_pattern) == 6:
                tt = t.reshape((1, 1, t.shape[0], t.shape[1], -1))
                padded_ts.append(
                    F.pad(tt, pad_pattern, 'constant', 0, data_format='NCDHW').squeeze())

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == paddle.Tensor:
            new_t = pad_tensor(inp)
            if len(new_t) == 0:
                ret = paddle.to_tensor(new_t)
            else:
                ret = paddle.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([paddle.to_tensor(x) for x in inp])
            ret = paddle.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive matrix K1, K2 here to leverage multi-processing nature of dataloader
    if 'Gs' in ret and 'Hs' in ret:
        try:
            G1_gt, G2_gt = ret['Gs']
            # print("G", G1_gt.shape, G2_gt.shape)
            H1_gt, H2_gt = ret['Hs']
            # print("H", H1_gt.shape, H2_gt.shape)
            sparse_dtype = np.float32
            # K1G = [kronecker_sparse(x.squeeze(), y.squeeze()).astype(
            #     sparse_dtype) for x, y in zip(G2_gt, G1_gt)]  # 1 as source graph, 2 as target graph
            # K1H = [kronecker_sparse(x.squeeze(), y.squeeze()).astype(
            #     sparse_dtype) for x, y in zip(H2_gt, H1_gt)]
            # K1G = CSRMatrix3d(K1G)
            # K1H = CSRMatrix3d(K1H).transpose()
            # NOTE: use dense implementation as workaround
            K1G = [np.kron(x.squeeze(), y.squeeze()).astype(sparse_dtype) for x, y in zip(G2_gt, G1_gt)]
            K1H = [np.kron(x.squeeze(), y.squeeze()).astype(sparse_dtype) for x, y in zip(H2_gt, H1_gt)]

            # , K1G.transpose(keep_type=True), K1H.transpose(keep_type=True)
            # ret['KGHs'] = K1G, paddle.transpose(K1H, (1, 0))
            ret['KGHs'] = [K1G, K1H]
        except ValueError:
            print('BUGBUGBUGBUG')
            pass

    # NOTE: NGMv2 require ret['aff_mat'], constructed via Fi and Fj
    #       But since NGMv2 has not yet implemented in paddle

    ret['batch_size'] = len(data)

    for v in ret.values():
        if type(v) is list:
            ret['num_graphs'] = len(v)
            break

    return ret


def worker_init_fix(worker_id):
    """Init dataloader workers with fixed seed."""
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(paddle.initial_seed())
    np.random.seed(paddle.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False) -> DataLoader:
    fix_seed = True  # "Paddle version now do NOT support unfixed seed"
    return paddle.io.DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER_NUM,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fix if fix_seed else worker_init_rand)
