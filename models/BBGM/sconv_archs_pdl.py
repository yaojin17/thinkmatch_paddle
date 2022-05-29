import paddle.nn
import paddle
import paddle.nn.functional as F
import paddle_spline_conv


class SiameseSConvOnNodes(paddle.nn.Layer):
    def __init__(self, input_node_dim):
        super(SiameseSConvOnNodes, self).__init__()
        self.num_node_features = input_node_dim
        self.mp_network = paddle_spline_conv.nn.SConv(input_features=self.num_node_features, output_features=self.num_node_features)

    def forward(self, graph):
        old_features = graph.x
        result = self.mp_network(graph)
        graph.x = old_features + 0.1 * result
        return graph


class SiameseNodeFeaturesToEdgeFeatures(paddle.nn.Layer):
    def __init__(self, total_num_nodes):
        super(SiameseNodeFeaturesToEdgeFeatures, self).__init__()
        self.num_edge_features = total_num_nodes

    def forward(self, graph, hyperedge=False):
        orig_graphs = graph.to_data_list()
        orig_graphs = [self.vertex_attr_to_edge_attr(graph) for graph in orig_graphs]
        if hyperedge:
            orig_graphs = [self.vertex_attr_to_hyperedge_attr(graph) for graph in orig_graphs]
        return orig_graphs

    def vertex_attr_to_edge_attr(self, graph):
        """Assigns the difference of node features to each edge"""
        flat_edges = graph.edge_index.transpose((1, 0)).reshape(-1)
        vertex_attrs = paddle.index_select(graph.x, axis=0, index=flat_edges)

        new_shape = (graph.edge_index.shape[1], 2, vertex_attrs.shape[1])
        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose((1, 0))
        new_edge_attrs = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        graph.edge_attr = new_edge_attrs
        return graph

    def vertex_attr_to_hyperedge_attr(self, graph):
        """Assigns the angle of node features to each hyperedge.
           graph.hyperedge_index is the incidence matrix."""
        flat_edges = graph.hyperedge_index.transpose((1, 0)).reshape(-1)
        vertex_attrs = paddle.index_select(graph.x, axis=0, index=flat_edges)

        new_shape = (graph.hyperedge_index.shape[1], 3, vertex_attrs.shape[1])

        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose((1, 0))
        v01 = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        v02 = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[2]
        v12 = vertex_attrs_reshaped[1] - vertex_attrs_reshaped[2]
        nv01 = paddle.norm(v01, p=2, axis=-1)
        nv02 = paddle.norm(v02, p=2, axis=-1)
        nv12 = paddle.norm(v12, p=2, axis=-1)

        cos1 = paddle.sum(v01 * v02, axis=-1) / (nv01 * nv02)
        cos2 = paddle.sum(-v01 * v12, axis=-1) / (nv01 * nv12)
        cos3 = paddle.sum(-v12 * -v02, axis=-1) / (nv12 * nv02)

        graph.hyperedge_attr = paddle.stack((cos1, cos2, cos3), axis=-1)
        return graph
