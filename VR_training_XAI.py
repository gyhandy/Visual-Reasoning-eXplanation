'''
This script train the graph reasoning algorithm use detected concept for XAI
'''

import torch
from torch_geometric.data import DataLoader
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import sys
import random
import numpy as np
from src.image_folder import make_dataset
from torch_geometric.data import Data
import os
import torch.nn.functional as f
from torch.nn import Parameter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from torch_geometric.nn import GraphConv
from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
import argparse
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
'''dataset class'''
# only see first
class VR_graph(InMemoryDataset): # create training dataset
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        super(VR_graph, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.XAI_act = torch.load(self.processed_file_names[0])


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [os.path.join(self.root, 'vec2graph', 'Xception_XAI_graph_training.dataset')]

    def download(self):
        pass
    '''useful for utils'''
    def read_txt(self, path):
        f = open(path, 'r')
        b = f.read()
        pos_graph = eval(b)
        f.close()
        return pos_graph

    def choose_graph(self, list_mix):
        while 1:
            pos_graph_path = random.choice(list_mix)
            if '_graph.' in pos_graph_path:
                return pos_graph_path
    '''process for contrastive loss'''

    def _get_3class_eij(self):
        class_list = ['fire_engine', 'ambulance', 'school_bus']

        graph_data_list = []
        for label, cate in enumerate(class_list): # 3 category, 6 graph for each class [+1,-1,-1, +1,-1,-1]
            pos_class_path = os.path.join('./result/img2vec/', 'train', cate, cate) # real + eij
            pos_class_g = make_dataset(pos_class_path)
            for i in range(2): # each batch we choose 2 instance for each category

                #  positive
                graph_path = self.choose_graph(pos_class_g)
                graph = self.read_txt(graph_path)
                '''x, y'''
                x = torch.ones((len(graph), 512))  # (4, 512)
                for n, vec in enumerate(graph.values()):
                    x[n] = torch.LongTensor(vec)
                y = torch.FloatTensor([label + 1]) # label of each category
                '''edge_index'''
                source_nodes = []
                target_nodes = []
                start_choice = [n for n in range(len(graph))]
                for startpt in start_choice:
                    # do not set loop
                    end_choice = start_choice.copy()
                    end_choice.remove(startpt)
                    for endpt in end_choice:
                        source_nodes.append(startpt)
                        target_nodes.append(endpt)
                edge = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

                '''edge_feature'''
                '''1 edge feature extraction'''
                edge_path = graph_path.replace('_graph.', '_edge.')
                edge_feature = self.read_txt(edge_path)
                e_all = torch.zeros((len(edge_feature), 4, 4))  # (4, 4, 4)
                for startpt, startpt_values in edge_feature.items():
                    for endpt, endpt_values in startpt_values.items():
                        e_all[startpt][endpt] = torch.tensor(endpt_values)
                '''2 get edge_feature form edge_index'''
                e = []
                for index, source_p in enumerate(source_nodes):
                    target_p = target_nodes[index]
                    e.append(np.array(e_all[source_p][target_p]))
                e = torch.Tensor(e)
                data = Data(x=x, edge_attr=e, edge_index=edge, y=y)
                graph_data_list.append(data)

                #  negative
                remain_cate = class_list.copy()
                remain_cate.remove(cate)
                for eij_label, neg_cate in enumerate(remain_cate):
                    neg_class_path = os.path.join('./result/img2vec/', 'train', neg_cate, cate)
                    neg_class_g = make_dataset(neg_class_path)
                    neg_graph_path = self.choose_graph(neg_class_g)
                    graph = self.read_txt(neg_graph_path)
                    x = torch.ones((len(graph), 512))  # (3, 512)
                    for n, vec in enumerate(graph.values()):
                        x[n] = torch.LongTensor(vec)
                    y = torch.FloatTensor([3 + 2 * (label) + (eij_label+1)])  # label of each category (45,67,89)
                    '''edge_index'''
                    source_nodes = []
                    target_nodes = []
                    start_choice = [n for n in range(len(graph))]
                    for startpt in start_choice:
                        # do not set loop
                        end_choice = start_choice.copy()
                        end_choice.remove(startpt)
                        for endpt in end_choice:
                            source_nodes.append(startpt)
                            target_nodes.append(endpt)
                    edge = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

                    '''edge_feature'''
                    '''1 edge feature extraction'''
                    edge_path = neg_graph_path.replace('_graph.', '_edge.')
                    edge_feature = self.read_txt(edge_path)
                    e_all = torch.zeros((len(edge_feature), 4, 4))  # (4, 4, 4)
                    for startpt, startpt_values in edge_feature.items():
                        for endpt, endpt_values in startpt_values.items():
                            e_all[startpt][endpt] = torch.tensor(endpt_values)
                    '''2 get edge_feature form edge_index'''
                    e = []
                    for index, source_p in enumerate(source_nodes):
                        target_p = target_nodes[index]
                        e.append(np.array(e_all[source_p][target_p]))
                    e = torch.Tensor(e)
                    data = Data(x=x, edge_attr=e, edge_index=edge, y=y)
                    graph_data_list.append(data)

        return graph_data_list
    def _get_1class_XAI(self, cate, label):
        """form as graph and store activation value for given class"""
        class_list = ['fire_engine', 'ambulance', 'school_bus']

        graph_data_list = []
        XAI_activation_list = []

        pos_class_path = os.path.join(self.root, 'img2vec', cate + '_detect_graph')
        for roots, dirs, files in os.walk(pos_class_path):
            for i, file in enumerate(files):  # we only need limited number for each class
                if '_graph' in file: # first process graph
                    try:
                        graph_path = os.path.join(roots,file)
                        graph = self.read_txt(graph_path)
                        '''x, y'''
                        x = torch.ones((len(graph), 2048))  # (4, 2048)
                        for n, vec in enumerate(graph.values()):
                            x[n] = torch.tensor(vec)
                        y = torch.FloatTensor([label + 1]) # label of each category
                        '''edge_index'''
                        source_nodes = []
                        target_nodes = []
                        start_choice = [n for n in range(len(graph))]
                        for startpt in start_choice:
                            # do not set loop
                            end_choice = start_choice.copy()
                            end_choice.remove(startpt)
                            for endpt in end_choice:
                                source_nodes.append(startpt)
                                target_nodes.append(endpt)
                        edge = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

                        '''edge_feature'''
                        '''1 edge feature extraction'''
                        edge_path = graph_path.replace('_graph.', '_edge.')
                        edge_feature = self.read_txt(edge_path)
                        e_all = torch.zeros((len(edge_feature), 4, 4))  # (4, 4, 4)
                        for startpt, startpt_values in edge_feature.items():
                            for endpt, endpt_values in startpt_values.items():
                                e_all[startpt][endpt] = torch.tensor(endpt_values)
                        '''2 get edge_feature form edge_index'''
                        e = []
                        for index, source_p in enumerate(source_nodes):
                            target_p = target_nodes[index]
                            e.append(np.array(e_all[source_p][target_p]))
                        e = torch.Tensor(e)
                        '''XAI_activation'''
                        act_path = graph_path.replace('_graph.txt', '_XAI.npy')
                        act_vector = np.load(act_path)
                        # merge the act into y label
                        data_pos = Data(x=x, edge_attr=e, edge_index=edge, y=[(y, cate, torch.FloatTensor(act_vector))]) # y is list
                        # graph_data_list.append(data)

                        #  negative
                        remain_cate = class_list.copy()
                        remain_cate.remove(cate)
                        for eij_label, neg_cate in enumerate(remain_cate): # only load graph, no activation
                            neg_cate_file = cate + '2' + neg_cate # e.g. fire_engine2ambulance
                            neg_graph_path = graph_path.replace(cate, neg_cate_file)
                            graph = self.read_txt(neg_graph_path)
                            x = torch.ones((len(graph), 2048))  # (4, 512)
                            for n, vec in enumerate(graph.values()):
                                # x[n] = torch.LongTensor(vec) # will make all of them 0
                                x[n] = torch.tensor(vec)
                            y = torch.FloatTensor([label + 1])  # label of each category (45,67,89)
                            '''edge_index'''
                            source_nodes = []
                            target_nodes = []
                            start_choice = [n for n in range(len(graph))]
                            for startpt in start_choice:
                                # do not set loop
                                end_choice = start_choice.copy()
                                end_choice.remove(startpt)
                                for endpt in end_choice:
                                    source_nodes.append(startpt)
                                    target_nodes.append(endpt)
                            edge = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

                            '''edge_feature'''
                            '''1 edge feature extraction'''
                            edge_path = neg_graph_path.replace('_graph.', '_edge.')
                            edge_feature = self.read_txt(edge_path)
                            e_all = torch.zeros((len(edge_feature), 4, 4))  # (4, 4, 4)
                            for startpt, startpt_values in edge_feature.items():
                                for endpt, endpt_values in startpt_values.items():
                                    e_all[startpt][endpt] = torch.tensor(endpt_values)
                            '''2 get edge_feature form edge_index'''
                            e = []
                            for index, source_p in enumerate(source_nodes):
                                target_p = target_nodes[index]
                                e.append(np.array(e_all[source_p][target_p]))
                            e = torch.Tensor(e)
                            # '''XAI_activation'''
                            data_neg = Data(x=x, edge_attr=e, edge_index=edge, y=(y, neg_cate))
                            data_pos.y.append(data_neg) # merge into y
                        '''finally add all into data'''
                        graph_data_list.append(data_pos)
                    except:
                        print(graph_path)



        return graph_data_list, XAI_activation_list

    def process(self):
        class_list = ['fire_engine', 'ambulance', 'school_bus']

        data_list = [] # store all graphs
        XAI_list = []
        """Yields pair data"""
        pair_labels = [] # 1 or 0
        for class_label, target_class in enumerate(class_list):
            graph_data_list, XAI_activation_list = self._get_1class_XAI(target_class, class_label)  # get the graphs for this label

            '''add pair of data each time'''
            data_list += graph_data_list
            XAI_list += XAI_activation_list

        data, slices = self.collate(data_list)
        # ava
        # torch.save((data, slices, XAI_list), self.processed_paths[0])
        torch.save((data, slices, XAI_list), self.processed_file_names[0])

'''Graph model class'''
# only see these two
class GraphConv(MessagePassing):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}_2 \mathbf{x}_j.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, aggr: str = 'add', bias: bool = True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        '''fi = W1 * fi + sum(W3* concat(eij * W2 * fj, edge_ij);   edge = W4 * edge'''
        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class MyGCNConv_eij_adap_batch(MessagePassing):
    def __init__(self, in_channels, out_channels, in_edge_features, out_edge_features):
        super(MyGCNConv_eij_adap_batch, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin_W1 = torch.nn.Linear(in_channels, out_channels)# W1 for self
        self.lin_W2 = torch.nn.Linear(in_channels, out_channels)# W2 for neighbors
        self.lin_edge = torch.nn.Linear(in_edge_features, out_edge_features)
        self.bn_edge = torch.nn.BatchNorm1d(out_edge_features)
        self.lin_message = torch.nn.Linear(out_channels + out_edge_features, out_channels)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_attr, edge_index, eij, batch_size):
        # x has shape [N, in_channels]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # edge_index has shape [2, E]
        '''fi = W1 * fi + sum(W3* concat(eij * W2 * fj, edge_ij);   edge = W4 * edge'''
        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)

        x_j = self.lin_W2(x[0])  # W2 * fj
        edge_attr = self.lin_edge(edge_attr) #  W4 * edge
        edge_attr =  self.bn_edge(edge_attr)
        out = self.propagate(edge_index, x=x_j, edge_weight=eij.repeat(batch_size), edge_attr=edge_attr)

        x_i = x[1]
        if x_i is not None:
            out += self.lin_W1(x_i) # W1 * fi

        #  Linearly transform node feature matrix.

        return edge_attr, out

    def message(self, x_j, edge_weight, edge_attr):
        # x_j has shape [E, out_channels]
        # each node prepare some information ready to pass to neighbor based on edge connection
        # Step 4: Normalize node features.
        # return norm.view(-1, 1) * x_j
        after_eij = edge_weight.view(-1, 1) * x_j # use eij first and only on the node feature
        all_feature = torch.cat((after_eij, edge_attr), dim=1) # then concate the edgefeature as input
        return self.lin_message(all_feature) # W3 * (concat(node, edge))
class MyGCNNet_shareW_adap_batch(torch.nn.Module):
    def __init__(self, dataset, interest_class_num):
        super(MyGCNNet_shareW_adap_batch, self).__init__()

        num_features = dataset.num_features
        dim = 32
        edge_features = 4
        dim_edge_features = 5
        self.output_dim = interest_class_num # one graph only predict corresponding value

        '''GCN_eij'''
        self.eij_1 = Parameter(torch.Tensor(12))
        torch.nn.init.normal(self.eij_1, mean=0, std=1)
        self.eij_2 = Parameter(torch.Tensor(12))
        torch.nn.init.normal(self.eij_2, mean=0, std=1)
        self.eij_3 = Parameter(torch.Tensor(12))
        torch.nn.init.normal(self.eij_3, mean=0, std=1)

        self.conv1 = MyGCNConv_eij_adap_batch(num_features, dim * 2, edge_features, dim_edge_features)
        self.bn1 = torch.nn.BatchNorm1d(dim * 2) # only do on edge feature

        self.conv2 = MyGCNConv_eij_adap_batch(dim * 2, dim, dim_edge_features, dim_edge_features)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = MyGCNConv_eij_adap_batch(dim, dim, dim_edge_features, dim_edge_features)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        all_graph_vector_len = 3*(dim * 4 + dim_edge_features * 12)
        self.bn6 = torch.nn.BatchNorm1d(all_graph_vector_len) # 564 = 188 * 3
        self.fc2 = Linear(all_graph_vector_len, self.output_dim)

    def forward_1(self, x, edge_attr, edge_index, batch):
        '''both x and edge_attr been updated'''
        edge_attr, x = self.conv1(x, edge_attr, edge_index, self.eij_1, self.batch_size)
        x = F.relu(x)
        x = self.bn1(x)
        edge_attr, x = self.conv2(x, edge_attr, edge_index, self.eij_1, self.batch_size)
        x = F.relu(x)
        x = self.bn2(x)
        edge_attr, x = self.conv3(x, edge_attr, edge_index, self.eij_1, self.batch_size)
        x = F.relu(x)
        x = self.bn3(x)

        '''add a normalization'''
        # x = global_add_pool(x, batch)
        '''use concatenete for both x and edge feature'''
        # x = x.view(int(batch.max().item() + 1), -1)  # (Num of graph, feature_dim * Num of node) (6, 128)
        x = x.view(batch, -1)
        edge_attr = edge_attr[0: edge_index.shape[-1]].view(batch, -1)
        # edge_attr = f.normalize(edge_attr, p=2, dim=1)
        graph_vector = torch.cat((x, edge_attr), dim=1)  # (Num of graph, 128+60)
        return graph_vector


    def forward_2(self, x, edge_attr, edge_index, batch):
        '''both x and edge_attr been updated'''
        edge_attr, x = self.conv1(x, edge_attr, edge_index, self.eij_2, self.batch_size)
        x = F.relu(x)
        x = self.bn1(x)
        edge_attr, x = self.conv2(x, edge_attr, edge_index, self.eij_2, self.batch_size)
        x = F.relu(x)
        x = self.bn2(x)
        edge_attr, x = self.conv3(x, edge_attr, edge_index, self.eij_2, self.batch_size)
        x = F.relu(x)
        x = self.bn3(x)
        '''add a normalization'''
        # x = global_add_pool(x, batch)
        '''use concatenete for both x and edge feature'''
        x = x.view(batch, -1)  # (Num of graph, feature_dim * Num of node) (1, 128)

        edge_attr = edge_attr[0: edge_index.shape[-1]].view(batch, -1)
        graph_vector = torch.cat((x, edge_attr), dim=1)  # (Num of graph, 128+60)
        return graph_vector

    def forward_3(self, x, edge_attr, edge_index, batch):
        '''both x and edge_attr been updated'''
        edge_attr, x = self.conv1(x, edge_attr, edge_index, self.eij_3, self.batch_size)
        x = F.relu(x)
        x = self.bn1(x)
        edge_attr, x = self.conv2(x, edge_attr, edge_index, self.eij_3, self.batch_size)
        x = F.relu(x)
        x = self.bn2(x)
        edge_attr, x = self.conv3(x, edge_attr, edge_index, self.eij_3, self.batch_size)
        x = F.relu(x)
        x = self.bn3(x)
        '''add a normalization'''
        # x = global_add_pool(x, batch)
        '''use concatenete for both x and edge feature'''
        x = x.view(batch, -1) # (Num of graph, feature_dim * Num of node) (6, 128)

        edge_attr = edge_attr[0: edge_index.shape[-1]].view(batch, -1)
        graph_vector = torch.cat((x, edge_attr),dim=1) # (Num of graph, 128+60)
        return graph_vector

    def forward(self, graph_1, graph_2, graph_3):
        all_graph_vector = torch.cat((graph_1, graph_2, graph_3), dim=1)
        x = self.fc2(all_graph_vector)
        return x

'''Loss class'''
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

'''util functions'''

def plot_acc_loss(loss_all, inter_loss, intra_loss, epoch, args):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("batchs")
    host.set_ylabel("training-loss")

    # plot curves
    p1, = host.plot(range(len(loss_all)), loss_all, label="loss_all")
    p2, = host.plot(range(len(inter_loss)), inter_loss, label="inter_loss")
    p3, = host.plot(range(len(intra_loss)), intra_loss, label="intra_loss")
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["left"].label.set_color(p2.get_color())
    host.axis["left"].label.set_color(p3.get_color())

    plt.draw()
    plt.savefig(os.path.join(args.root, 'vec2graph', 'train_log', 'GraphConv_Xception', 'train_{}.jpg'.format(epoch)))
    plt.show()

def show_eij(eij, self_eij, num_node, normalize=False):
    eij = eij.reshape((num_node, num_node - 1)).cpu().detach().numpy()
    self_eij = self_eij.cpu().detach().numpy()
    new_eij = np.zeros((num_node, num_node))
    for index, row in enumerate(eij):
        if normalize:
            new_eij[index] = np.insert(row, index, 0, 0) # diagnal = 0
        else:
            new_eij[index] = np.insert(row, index, self_eij[index], 0)
    if normalize:
        new_eij = f.normalize(torch.tensor(new_eij), p=1, dim=0).detach().numpy()
    return new_eij

def minmaxscaler(data):
    min1 = min(data[0])
    max1 = max(data[0])
    return (data - min1)/(max1-min1)
'''train and test'''
def train_share(epoch, model_share, optimizer_share, train_dataset, train_loader, batch_size, MSE_Loss,
                loss_log_package, device):
    model_share.train()

    '''print eij'''
    if epoch % 200 == 0: # lower the learning rate every 3 epoch
        for param_group in optimizer_share.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
    loss_all = 0
    label_2_category = { '1' :'fire_engine',  '2': 'ambulance',  '3': 'school_bus'}
    batch_loss = 0
    for batch_index, data in tqdm(enumerate(train_loader)):
        label = int(data.y[0][0][0][0]) # 1, 2, 3
        all_data = {'fire_engine': None, 'ambulance': None, 'school_bus': None}
        # positive data
        all_data[label_2_category[str(label)]] = data
        # negative data
        for data_index, neg_data in enumerate(data.y[0]):
            if data_index != 0: # 0 is the class
                neg_cate = neg_data.y[1]
                all_data[neg_cate] = neg_data
        graph_1 = all_data['fire_engine'].to(device)
        output_1 = model_share.forward_1(graph_1.x, graph_1.edge_attr, graph_1.edge_index, batch=1)
        graph_2 = all_data['ambulance'].to(device)
        output_2 = model_share.forward_2(graph_2.x, graph_2.edge_attr, graph_2.edge_index, batch=1)
        graph_3 = all_data['school_bus'].to(device)
        output_3 = model_share.forward_3(graph_3.x, graph_3.edge_attr, graph_3.edge_index, batch=1)
        graph_predict = model_share.forward(output_1, output_2, output_3) # MLP merge
        norm_graph_predict = f.normalize(graph_predict, p=1, dim=1)  # L1 normalize
      

        '''GT(Xception)'''
        model_category_index = [555, 407, 779]# fire_engine 555; ambulance 407; school_bus 779
        XAI_act = data.y[0][0][2] # all [5, 1008]
        first_order_act = XAI_act[0].cpu().numpy()
        interest_vec = []
        for interest_dim in model_category_index:
            interest_vec.append(first_order_act[interest_dim])
        interest_vec_1 = torch.tensor(np.array(interest_vec)[np.newaxis, :])
        interest_vec = f.normalize(interest_vec_1, p=1, dim=1) # L1 normalize
        loss = MSE_Loss(interest_vec.to(device), norm_graph_predict) # for each graph
        loss.backward()
        batch_loss += loss # show every batch
        if batch_index % batch_size == 0 and batch_index !=0: # every batch_size(32) samples, we update the param
            '''update param'''
            optimizer_share.step()
            optimizer_share.zero_grad()
            '''store the loss'''
            loss_log_package['all_loss_log'].append(batch_loss)
            loss_all += batch_loss  # show after each epoch
            batch_loss = 0
        if batch_index % 100 == 0:
            print(batch_index, 'batch_MSE_loss Loss:{}, GT{}, Graph_predict{}'.format(batch_loss, interest_vec, norm_graph_predict))

    return loss_all / len(train_dataset), loss_log_package


def main(args):
    dataset = VR_graph(root=args.result_root) # this root will not be used
    train_dataset = dataset  # total 180000
    train_epoch = 1000
    batch_size = 128

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # due to eij switch, batch implement inside
    save_root = os.path.join(args.result_root, 'model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_share = MyGCNNet_shareW_adap_batch(dataset, interest_class_num=3)  # all class share base structure

    '''load pretrain model'''
    if args.load_epoch:
        save_name = os.path.join(save_root, 'GraphConv_Xception_epoch_{}.pt'.format(args.load_epoch))
        model_share.load_state_dict(torch.load(save_name))
        ''' load pretrained model'''
        print('[Pre] 1 Loaded pretrianed model{}'.format(save_name))

    model_share = model_share.to(device)
    model_share.batch_size = 1 # fix the batch size

    ''' load pretrained model'''
    print('[Pre] 1 Loading class-specific model')

    optimizer_share = torch.optim.Adam(model_share.parameters(), lr=0.01)
    MSE_Loss = torch.nn.MSELoss()

    '''store the loss to print'''
    loss_log_package = {'all_loss_log': [0], 'inter_loss_log': [0], 'intra_loss_log': [0]}
    for epoch in range(1, train_epoch):
        '''train_share'''
        train_loss, loss_log_package = train_share(epoch, model_share, optimizer_share,
                                                   train_dataset, train_loader, batch_size,
                                                   MSE_Loss, loss_log_package, device)
        print(train_loss)
        print('Epoch: {:03d}, Train Loss: {:.7f} '.format(epoch, train_loss))

        '''save model'''
        if epoch % 20 == 0:
            save_name = os.path.join(save_root, 'GraphConv_Xception_epoch_{}.pt'.format(epoch))
            torch.save(model_share.state_dict(), os.path.join(save_root, save_name))
            print('saved!', epoch)
            print('learning rate{}'.format(optimizer_share.param_groups))

        '''draw the loss'''
        all_loss_log = loss_log_package['all_loss_log']
        inter_loss_log = loss_log_package['inter_loss_log']
        intra_loss_log = loss_log_package['intra_loss_log']

        plot_acc_loss(all_loss_log, inter_loss_log, intra_loss_log, epoch, args)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--result_root', type=str, help='directory to results of discover concept.py.', default='result')
  parser.add_argument('--load_epoch', type=int, help='checkpoints want to load', default=None)
  return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))