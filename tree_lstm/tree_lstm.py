"""
referenced codes in https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py
https://arxiv.org/abs/1503.00075
"""
import torch
import dgl
from copy import deepcopy


class Tree:
    def __init__(self, h_size):
        self.dgl_graph = dgl.DGLGraph()
        self.h_size = h_size

    def add_node(self, parent_id=None, tensor:torch.Tensor = torch.Tensor()):
        self.dgl_graph.add_nodes(1, data={'x': tensor.unsqueeze(0),
                                          'h': tensor.new_zeros(size=(1, self.h_size)),
                                          'c': tensor.new_zeros(size=(1, self.h_size))})
        added_node_id = self.dgl_graph.number_of_nodes() - 1
        if parent_id:
            self.dgl_graph.add_edge(added_node_id, parent_id)
        return added_node_id

    def add_node_bottom_up(self, child_ids, tensor: torch.Tensor):
        self.dgl_graph.add_nodes(1, data={'x': tensor.unsqueeze(0),
                                          'h': tensor.new_zeros(size=(1, self.h_size)),
                                          'c': tensor.new_zeros(size=(1, self.h_size))})
        added_node_id = self.dgl_graph.number_of_nodes() - 1
        for child_id in child_ids:
            self.dgl_graph.add_edge(child_id, added_node_id)
        return added_node_id

    def add_link(self, child_id, parent_id):
        self.dgl_graph.add_edge(child_id, parent_id)


class BatchedTree:
    def __init__(self, tree_list):
        graph_list = []
        for tree in tree_list:
            graph_list.append(tree.dgl_graph)
        self.batch_dgl_graph = dgl.batch(graph_list)

    def get_hidden_state(self):
        graph_list = dgl.unbatch(self.batch_dgl_graph)
        hidden_states = []
        max_nodes_num = max([len(graph.nodes) for graph in graph_list])
        for graph in graph_list:
            hiddens = graph.ndata['h']
            node_num, hidden_num = hiddens.size()
            if len(hiddens) < max_nodes_num:
                padding = hiddens.new_zeros(size=(max_nodes_num - node_num, hidden_num))
                hiddens = torch.cat((hiddens, padding), dim=0)
            hidden_states.append(hiddens)
        return torch.stack(hidden_states)


class TreeLSTM(torch.nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout,
                 cell_type='n_ary',
                 n_ary=None,
                 num_stacks=2):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = torch.nn.Dropout(dropout)
        if cell_type == 'n_ary':
            self.cell = NaryTreeLSTMCell(n_ary, x_size, h_size)
        else:
            self.cell = ChildSumTreeLSTMCell(x_size, h_size)
        self.num_stacks = num_stacks

    def forward(self, batch: BatchedTree):
        batches = [deepcopy(batch) for _ in range(self.num_stacks)]
        for stack in range(self.num_stacks):
            cur_batch = batches[stack]
            if stack > 0:
                prev_batch = batches[stack - 1]
                cur_batch.batch_dgl_graph.ndata['x'] = prev_batch.batch_dgl_graph.ndata['h']
            cur_batch.batch_dgl_graph.register_message_func(self.cell.message_func)
            cur_batch.batch_dgl_graph.register_reduce_func(self.cell.reduce_func)
            cur_batch.batch_dgl_graph.register_apply_node_func(self.cell.apply_node_func)
            cur_batch.batch_dgl_graph.ndata['iou'] = self.cell.W_iou(self.dropout(batch.batch_dgl_graph.ndata['x']))
            dgl.prop_nodes_topo(cur_batch.batch_dgl_graph)
        return batches


class NaryTreeLSTMCell(torch.nn.Module):
    def __init__(self, n_ary, x_size, h_size):
        super(NaryTreeLSTMCell, self).__init__()
        self.n_ary = n_ary
        self.h_size = h_size
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(n_ary * h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(n_ary * h_size, n_ary * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        padding_hs = self.n_ary - nodes.mailbox['h'].size(1)
        padding = h_cat.new_zeros(size=(nodes.mailbox['h'].size(0), padding_hs * self.h_size))
        h_cat = torch.cat((h_cat, padding), dim=1)
        f = torch.sigmoid(self.U_f(h_cat)).view(nodes.mailbox['h'].size(0), self.n_ary, self.h_size)
        padding_cs = self.n_ary - nodes.mailbox['c'].size(1)
        padding = h_cat.new_zeros(size=(nodes.mailbox['c'].size(0), padding_cs, self.h_size))
        c = torch.cat((nodes.mailbox['c'], padding), dim=1)
        c = torch.sum(f * c, 1)
        return {'iou': nodes.data['iou'] + self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class ChildSumTreeLSTMCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': nodes.data['iou'] + self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}