'''
This script use trained model explain Why and Why Not for
'''

import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
import random
import numpy as np
from src.image_folder import make_dataset
from torch_geometric.data import Data
import os
from sklearn.datasets import load_digits
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch.nn.functional as f

import matplotlib.image as mpimg

from torch.nn import Parameter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
'''dataset class'''
# only see first
class VR_graph(InMemoryDataset): # create training dataset
    def __init__(self, root, transform=None, pre_transform=None):
        super(VR_graph, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.XAI_act = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # return ['/lab/tmpig23b/u/andy/VR_SC/ACEdata/results/vec2graph/VR_graph_ImageNet_2contrast_edge.dataset']
        # return ['/lab/tmpig23b/u/andy/VR_SC/ACEdata/results/vec2graph/XAI_graph_training.dataset']
        return ['/lab/tmpig23b/u/andy/VR_SC/ACEdata/results/vec2graph/Xception_XAI_graph_training.dataset']

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

    def _get_pair(self, positive):
        """Generate one pair of graphs."""
        # g = self._get_graph()
        '''fill in our wanted node'''
        class_dict = {'0': 'Zebra', '1': 'Ambulance', '2': 'School_bus', '3': 'Fire_engine', '4': 'Basketball'}
        left_class = random.choice(list(class_dict.keys()))
        left_path = '/data/data/ACEdata/results/' + class_dict[left_class] + '_detect_graph'
        left_g = make_dataset(left_path)
        '''fill node representation'''
        left_graph_path = self.choose_graph(left_g)
        left_graph = self.read_txt(left_graph_path)
        left_x = torch.ones((len(left_graph), 512))  # (3, 512)
        for n, vec in enumerate(left_graph.values()):
            left_x[n] = torch.tensor(vec)
        left_y = torch.FloatTensor([int(left_class)])

        '''fully connection for edges'''
        source_nodes = []
        target_nodes = []
        start_choice = [n for n in range(len(left_graph))]
        for startpt in start_choice:
            # do not set loop
            end_choice = start_choice.copy()
            end_choice.remove(startpt)
            for endpt in end_choice:
                source_nodes.append(startpt)
                target_nodes.append(endpt)
        left_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        left_data = Data(x=left_x, edge_index=left_edge_index, y=left_y)


        '''fill edge representation'''
        # pos_graph_edge_path = pos_graph_path.replace('_graph.', '_edge.')
        # pos_graph_edge = self.read_txt(pos_graph_edge_path)
        # permuted_g.edge = pos_graph_edge

        '''
        for changed_g, two choice
        '''
        if positive: # positive same as left
            '''fill node representation'''
            right_graph_path = self.choose_graph(left_g)
            right_y = torch.FloatTensor([int(left_class)])  # positive same as left
        else: # negative different from left
            rest_class = list(class_dict.keys())
            rest_class.remove(left_class)  # negative different from left
            neg_class = random.choice(rest_class)
            negative_path = '/data/data/ACEdata/results/' + class_dict[neg_class] + '_detect_graph'
            negative_g = make_dataset(negative_path)
            '''fill node representation'''
            right_graph_path = self.choose_graph(negative_g)
            right_y = torch.FloatTensor([int(neg_class)])  # negative different from left

        right_graph = self.read_txt(right_graph_path)
        right_x = torch.ones((len(right_graph), 512))  # (3, 512)
        for n, vec in enumerate(right_graph.values()):
            right_x[n] = torch.tensor(vec)


        '''fully connection for edges'''
        source_nodes = []
        target_nodes = []
        start_choice = [n for n in range(len(right_graph))]
        for startpt in start_choice:
            # do not set loop
            end_choice = start_choice.copy()
            end_choice.remove(startpt)
            for endpt in end_choice:
                source_nodes.append(startpt)
                target_nodes.append(endpt)
        right_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        right_data = Data(x=right_x, edge_index=right_edge_index, y=right_y)


        # n_changes = self._k_pos if positive else self._k_neg
        # changed_g = substitute_random_edges(g, n_changes)
        return left_data, right_data
    '''(batch ==6)'''
    def _get_3class_eij(self):
        """Generate 18 graphs(6 for each class [+1,-1,-1, +1,-1,-1])."""
        # g = self._get_graph()
        '''fill in our wanted node'''
        # class_dict = {'0': 'Zebra', '1': 'Ambulance', '2': 'School_bus', '3': 'Fire_engine', '4': 'Basketball'}
        # class_dict = {'0': 'Zebra', '1': 'Ambulance', '2': 'School_bus', '3': 'Fire_engine', '4': 'Basketball'}
        # class_dict = {'0': 'Fracture', '1': 'Enlarge', '2': 'Supported'}
        # class_dict = {'0': 'Fire_engine', '1': 'Ambulance', '2': 'School_bus'}
        class_list = ['fire_engine', 'ambulance', 'school_bus']

        graph_data_list = []
        for label, cate in enumerate(class_list): # 3 category, 6 graph for each class [+1,-1,-1, +1,-1,-1]
            pos_class_path = os.path.join('/lab/tmpig23b/u/andy/VR_SC/ACEdata/results/img2vec/', 'train', cate, cate) # real + eij
            pos_class_g = make_dataset(pos_class_path)
            for i in range(2): # each batch we choose 2 instance for each category

                #  positive
                graph_path = self.choose_graph(pos_class_g)
                graph = self.read_txt(graph_path)
                '''x, y'''
                x = torch.ones((len(graph), 512))  # (4, 512)
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
                data = Data(x=x, edge_attr=e, edge_index=edge, y=y)
                graph_data_list.append(data)

                #  negative
                remain_cate = class_list.copy()
                remain_cate.remove(cate)
                for eij_label, neg_cate in enumerate(remain_cate):
                    neg_class_path = os.path.join('/lab/tmpig23b/u/andy/VR_SC/ACEdata/results/img2vec/', 'train', neg_cate, cate)
                    neg_class_g = make_dataset(neg_class_path)
                    neg_graph_path = self.choose_graph(neg_class_g)
                    graph = self.read_txt(neg_graph_path)
                    x = torch.ones((len(graph), 512))  # (3, 512)
                    for n, vec in enumerate(graph.values()):
                        x[n] = torch.tensor(vec)
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
        # class_list = ['fire_engine', 'ambulance', 'school_bus']

        graph_data_list = []
        XAI_activation_list = []

        pos_class_path = os.path.join('/lab/tmpig23b/u/andy/VR_SC/ACEdata/results/img2vec/', cate + '_detect_graph') # real + eij
        # pos_class_g = make_dataset(pos_class_path)
        for roots, dirs, files in os.walk(pos_class_path):
            for i, file in enumerate(files):  # we only need limited number for each class
                if '_graph' in file: # first process graph
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
                    data = Data(x=x, edge_attr=e, edge_index=edge, y=(y,torch.FloatTensor(act_vector)))
                    graph_data_list.append(data)



        return graph_data_list, XAI_activation_list

    def process(self):
        class_list = ['fire_engine', 'ambulance', 'school_bus']

        data_list = [] # store all graphs
        XAI_list = []
        """Yields pair data"""
        pair_labels = [] # 1 or 0
        # pair_class_labels = [] # class_dict = {'0': 'Zebra', '1': 'Ambulance',
        # '2': 'School_bus', '3': 'Fire_engine', '4': 'Basketball'}

        for class_label, target_class in enumerate(class_list):
            graph_data_list, XAI_activation_list = self._get_1class_XAI(target_class, class_label)  # get the graphs for this label

            '''add pair of data each time'''
            data_list += graph_data_list
            XAI_list += XAI_activation_list

        data, slices = self.collate(data_list)
        torch.save((data, slices, XAI_list), self.processed_paths[0])

'''Graph model class'''
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
def reshape_and_split_tensor(tensor, n_splits):
  """Reshape and split a 2D tensor along the last dimension.

  Args:
    tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
      multiple of `n_splits`.
    n_splits: int, number of splits to split the tensor into.

  Returns:
    return left and right
  """
  feature_dim = list(tensor.size())[-1]
  # feature dim must be known, otherwise you can provide that as an input
  assert isinstance(feature_dim, int)
  tensor_after = torch.reshape(tensor, (-1, feature_dim * n_splits))
  return torch.split(tensor_after, feature_dim, dim=1)
def vis_tsne_check(graph_vectors_save, class_label_save):
    # class_dict = {'1': 'Fire_engine', '2': 'Ambulance', '3': 'School_bus'}
    class_dict = {'1': 'Fire_engine', '2': 'Ambulance', '3': 'School_bus', '4': 'Ambulance_Fire', '5': 'School_Fire',
                  '6': 'Fire_Ambulance', '7':'School_Ambulance', '8': 'Fire_School_bus', '9': 'Ambulance_School_bus'}

    label_list = list(class_label_save)

    # label_list = []
    # for label in class_label_save:
    #     for n in label:
    #         label_list.append(n)

    digits = load_digits()
    embeddings = TSNE(n_jobs=1).fit_transform(graph_vectors_save)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    classes = list(class_dict.keys())
    unique = np.unique(classes)
    colors = [plt.cm.jet(i / float(len(unique) - 1)) for i in range(len(unique))]
    for i, u in enumerate(unique):
        xi = [vis_x[j] for j in range(len(vis_x)) if str(label_list[j]) == u]
        yi = [vis_y[j] for j in range(len(vis_y)) if str(label_list[j]) == u]
        plt.scatter(xi, yi, c=colors[i], label=class_dict[u])
    plt.legend()
    plt.show()
    '''before '''
    # plt.savefig('./graph_before.png')
    '''after '''
    plt.savefig('./graph_after.png')


def plot_acc_loss(loss_all, inter_loss, intra_loss):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    # par1 = host.twinx()  # 共享x轴

    # set labels
    host.set_xlabel("batchs")
    host.set_ylabel("training-loss")
    # par1.set_ylabel("test-accuracy") # second y axis, we do not need it now

    # plot curves
    p1, = host.plot(range(len(loss_all)), loss_all, label="loss_all")
    p2, = host.plot(range(len(inter_loss)), inter_loss, label="inter_loss")
    p3, = host.plot(range(len(intra_loss)), intra_loss, label="intra_loss")
    # p2, = par1.plot(range(len(acc)), acc, label="accuracy")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["left"].label.set_color(p2.get_color())
    host.axis["left"].label.set_color(p3.get_color())
    # par1.axis["right"].label.set_color(p2.get_color())

    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
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

'''reasoning'''
def hook_fn_backward(module, grad_input, grad_output): # get gradient
    # print(module) # 为了区分模块
    # 为了符合反向传播的顺序，我们先打印 grad_output
    # print('grad_output', grad_output)
    # 再打印 grad_input
    # print('grad_input', grad_input)
    # # 保存到全局变量
    # total_grad_in.append(grad_input)
    total_grad_out.append(grad_input) # W * X + b
def hook_fn_forward(module, input, output): # get activation of X
    # print(module) # 用于区分模块
    # print('input', input) # 首先打印出来
    # print('output', output)
    # total_activation_out.append(output) # 然后分别存入全局 list 中
    total_activation_out.append(input)

def reasoning(model_share, data_package, device):
    model_share.train()

    '''print eij'''
    # print('eij_fire\n',  show_eij(model_share.eij_1, model_share.self_eij_1, 4))
    # # print('self_eij_1', model_share.self_eij_1)
    # print('eij_ambu\n', show_eij(model_share.eij_2, model_share.self_eij_2, 4))
    # # print('self_eij_2', model_share.self_eij_2)
    # print('eij_scho\n', show_eij(model_share.eij_3, model_share.self_eij_3, 4))
    # # print('self_eij_3', model_share.self_eij_3)
    '''
    graph information
    '''
    node_dim = 32
    node_num = 4
    edge_dim = 5
    edge_num = 12
    '''
    class index
    '''
    class_list = ['fire_engine', 'ambulance', 'school_bus']
    all_data = {}
    # graph_output = [0]*len(class_list)
    # data = data_package[1][2].to(device) # get the positive class
    label = class_list.index((data_package[1][0]))+1 # 1, 2, 3
    for pack in data_package[1:]:
        cate, path, data = pack
        all_data[cate.split('2')[-1]] = data

    graph_1 = all_data['fire_engine'].to(device)
    output_1 = model_share.forward_1(graph_1.x, graph_1.edge_attr, graph_1.edge_index, batch=1)
    graph_2 = all_data['ambulance'].to(device)
    output_2 = model_share.forward_2(graph_2.x, graph_2.edge_attr, graph_2.edge_index, batch=1)
    graph_3 = all_data['school_bus'].to(device)
    output_3 = model_share.forward_3(graph_3.x, graph_3.edge_attr, graph_3.edge_index, batch=1)
    graph_predict = model_share.forward(output_1, output_2, output_3)  # MLP merge
    # norm_graph_predict = f.normalize(graph_predict, p=1, dim=1)  # normalize
    norm_graph_predict = F.softmax(graph_predict, dim=1)  # softmax
    '''
    Gradient analysis for reasoning
    '''
    # output.backward(torch.ones(output.shape).to(device)) # if we do as a whole, will mix all grad
    for index in range(graph_predict.shape[-1]):
        graph_predict[0][index].backward(retain_graph=True) # for each output, calculate the gradient
    print('reasoning finished')

    # Gradient analysis: all class's gradient are stroed in total_grad_out(3 in this case)
    GT_gradient = total_grad_out[label - 1][1]  # 0, bias; 1, Grad_X (1, 188), 2, Grad_W
    GT_activation = total_activation_out[0][0]  # 0, bias; 1, Grad_X (1, 188), 2, Grad_W
    Attention = GT_activation * GT_gradient
    # dim(32) * 4 + dim_edge_features(5) * 12
    Attention = Attention.resize(3,1,188)

    # answer why
    Pos_Attention = Attention[label - 1]
    visual_related = Pos_Attention[:, : node_dim * node_num].resize(node_num, node_dim)
    # visual_reason = np.mean(visual_related.cpu().detach().numpy(), axis=1) #(4 each one for a node)
    visual_reason = np.sum(visual_related.cpu().detach().numpy(), axis=1)  # (4 each one for a node)
    structure_related = Pos_Attention[:, node_dim * node_num:].resize(edge_num, edge_dim)
    structure_reason = np.sum(structure_related.cpu().detach().numpy(), axis=1)  # (12 each one for a edge)
    '''
    Answer correct or not (based on original model output)
    '''
    GT = data_package[0].cpu().detach().numpy().squeeze()
    output_list = list(GT)
    predict = output_list.index(max(output_list)) + 1
    if predict == label: # Correct, we answer why correct
        print('Correct prediction: {}'.format(class_list[predict-1]), '\n')
        print('Why correct? positive node and edge are correct.\n',
              'Visual reason:{}'.format(visual_reason),'\n',
              'structure reason:{}'.format(structure_reason))
        print('Start answering why not')
        all_label = [1, 2, 3]
        remain_label = all_label.copy()
        remain_label.remove(label)
        for neg_label in remain_label: # exclude positive class 0, keep only 1 and 2
            negative_activation = Attention[neg_label - 1]  # 0, bias; 1, Grad_X (1, 188), 2, Grad_W
            # dim(32) * 4 + dim_edge_features(5) * 12
            visual_related = negative_activation[:, : node_dim * node_num].resize(node_num, node_dim)
            # visual_reason = np.mean(visual_related.cpu().detach().numpy(), axis=1) #(4 each one for a node)
            visual_reason = np.sum(visual_related.cpu().detach().numpy(), axis=1)  # (4 each one for a node)
            structure_related = negative_activation[:, node_dim * node_num:].resize(edge_num, edge_dim)
            structure_reason = np.sum(structure_related.cpu().detach().numpy(), axis=1)  # (12 each one for a edge)
            print('Why not {}? negative node and edge explain.\n'.format(class_list[neg_label-1]),
                  'Visual reason:{}'.format(visual_reason), '\n',
                  'structure reason:{}'.format(structure_reason))
        # normalize the output
        # norm_graph_output = f.normalize(torch.tensor(graph_output), p=2, dim=0)
        print('Graph Prediction: {}\n GT: {}'.format(norm_graph_predict, GT))
    else: # Wrong, we answer why wrong
        print('Wrong prediction: should be {}, wrong predict as {}'.format(class_list[label-1],
                                                                          class_list[predict-1]))
        print('Why wrong? (1)for the GT class, negative node and edge are bad for GT class.\n',
              'Visual reason:{}'.format(visual_reason),'\n',
              'structure reason:{}'.format(structure_reason))
        negative_activation = Attention[predict - 1]  # 0, bias; 1, Grad_X (1, 188), 2, Grad_W
        # dim(32) * 4 + dim_edge_features(5) * 12
        visual_related = negative_activation[:, : node_dim * node_num].resize(node_num, node_dim)
        # visual_reason = np.mean(visual_related.cpu().detach().numpy(), axis=1) #(4 each one for a node)
        visual_reason = np.sum(visual_related.cpu().detach().numpy(), axis=1)  # (4 each one for a node)
        structure_related = negative_activation[:, node_dim * node_num:].resize(edge_num, edge_dim)
        structure_reason = np.sum(structure_related.cpu().detach().numpy(), axis=1)  # (12 each one for a edge)
        print('Why wrong? (2)for the Predicted class, positive node and edge are bad for Predicted class .\n',
              'Visual reason:{}'.format(visual_reason), '\n',
              'structure reason:{}'.format(structure_reason))
        pass
    '''
    show all detected concept
    '''
    plt.figure()
    plt.subplot(3, 1, 1)
    img1 = mpimg.imread(data_package[1][1])
    plt.imshow(img1)
    plt.subplot(3, 1, 2)
    img2 = mpimg.imread(data_package[2][1])
    plt.imshow(img2)
    plt.subplot(3, 1, 3)
    img3 = mpimg.imread(data_package[3][1])
    plt.imshow(img3)
    plt.show()

    total_grad_out.clear()
    total_activation_out.clear()





        #
        # print('Why wrong? for the GT class, negative node and edge are bad for GT class.\n',
        #       'Visual reason:{}'.format(visual_reason),'\n',
        #       'structure reason:{}'.format(structure_reason))
        # # analize the predicted class
        # Predict_gradient = total_grad_out[predict - 1][1]  # 0, bias; 1, Grad_X (1, 188), 2, Grad_W
        # # dim(32) * 4 + dim_edge_features(5) * 12
        # Pred_visual_related = Predict_gradient[:, : node_dim * node_num].resize(node_num, node_dim)
        # Pred_visual_reason = np.sum(Pred_visual_related.cpu().detach().numpy(), axis=1)  # (4 each one for a node)
        # Pred_structure_related = Predict_gradient[:, node_dim * node_num:].resize(edge_num, edge_dim)
        # Pred_structure_reason = np.sum(Pred_structure_related.cpu().detach().numpy(), axis=1)  # (12 each one for a edge)
        #
        # print('Why wrong? for the Predicted class, positive node and edge are bad for Predicted class .\n',
        #       'Visual reason:{}'.format(Pred_visual_reason),'\n',
        #       'structure reason:{}'.format(Pred_structure_reason))
def read_txt(path):
    f = open(path, 'r')
    b = f.read()
    pos_graph = eval(b)
    f.close()
    return pos_graph
def select_data(ori_graph_path, positive_class):
    data_package = []
    class_dict = {'fire_engine': 1, 'ambulance': 2, 'school_bus': 3}
    class_list = ['fire_engine', 'ambulance', 'school_bus']
    package_dict = {} # store info to load
    package_dict[positive_class] = class_dict[positive_class] # add positive class and target label(eij label)
    remain_cate = class_list.copy()
    remain_cate.remove(positive_class)
    for negative_class in remain_cate: # answer why not
        package_dict[positive_class+'2'+negative_class] = class_dict[negative_class] # add negative label (eij label)


    for interest_class in package_dict.keys():
        if interest_class == positive_class:
            graph_path = ori_graph_path
            '''XAI_activation'''
            act_path = graph_path.replace('_graph.txt', '_XAI.npy')
            act_vector = np.load(act_path)
            # model_category_index = [279, 265, 962]  # fire_engine 279; ambulance 265; school_bus 962
            model_category_index = [555, 407, 779]  # fire_engine 555; ambulance 407; school_bus 779 # Xception
            XAI_act = act_vector  # all [5, 1008]
            first_order_act = XAI_act[0]
            interest_vec = []
            for interest_dim in model_category_index:
                interest_vec.append(first_order_act[interest_dim])
            interest_vec_1 = torch.tensor(np.array(interest_vec)[np.newaxis, :])
            # interest_vec = f.normalize(interest_vec_1, p=1, dim=1)  # normalize
            interest_vec = F.softmax(interest_vec_1, dim=1)
            data_package.append(interest_vec)
        else:
            graph_path = ori_graph_path.replace(positive_class, interest_class) # e.g. fire : fine2school, fire2ambu
        '''
        load data
        '''
        graph = read_txt(graph_path)
        '''x, y'''
        x = torch.ones((len(graph), 2048))  # (4, 2048)
        for n, vec in enumerate(graph.values()):
            x[n] = torch.tensor(vec)
        y = torch.FloatTensor([package_dict[interest_class]])  # label of each category
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
        edge_feature = read_txt(edge_path)
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

        # # merge the act into y label
        # data = Data(x=x, edge_attr=e, edge_index=edge, y=(y, torch.FloatTensor(act_vector)))
        data = Data(x=x, edge_attr=e, edge_index=edge, y=y)
        img_path = graph_path.replace('_detect_graph', '_detect_img').replace('_graph.txt', '_img.png')
        data_package.append((interest_class, img_path, data))

    return data_package


class MyGCNNet_shareW_adap_batch_withhook(torch.nn.Module):
    def __init__(self, dataset, interest_class_num):
        super(MyGCNNet_shareW_adap_batch_withhook, self).__init__()
        num_features = dataset
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
        '''use concatenete for both x and edge feature'''
        x = x.view(batch, -1)
        edge_attr = edge_attr[0: edge_index.shape[-1]].view(batch, -1)
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
        '''use concatenete for both x and edge feature'''
        x = x.view(batch, -1) # (Num of graph, feature_dim * Num of node) (6, 128)

        edge_attr = edge_attr[0: edge_index.shape[-1]].view(batch, -1)
        graph_vector = torch.cat((x, edge_attr),dim=1) # (Num of graph, 128+60)
        return graph_vector

    def forward(self, graph_1, graph_2, graph_3):
        all_graph_vector = torch.cat((graph_1, graph_2, graph_3), dim=1)
        x = self.fc2(all_graph_vector)
        return x



def main():
    '''
    testing dataset may be same as training
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_share = MyGCNNet_shareW_adap_batch_withhook(2048, interest_class_num=3)  # all class share base structure
    '''
    we will use the trained model
    '''
    # model_share.load_state_dict(torch.load(
    #     "/lab/tmpig23b/u/andy/VR_SC/ACEdata/results/model/model_shareW_ImageNet_345_epoch.pt"))
    model_share.load_state_dict(torch.load(
        "/lab/tmpig23b/u/andy/VR_SC/ACEdata/results/model/Xception_GraphConv_model_shareW_ImageNet_400_epoch_shallow_softmax.pt"))


    model_share = model_share.to(device)
    model_share.batch_size = 1 # fix the batch size
    ''' load pretrained model'''
    print('[Pre] 1 Loading class-specific model')

    '''register hook for reasoning'''
    model_share.fc2.register_forward_hook(hook_fn_forward)
    model_share.fc2.register_backward_hook(hook_fn_backward)


    '''
    Start reasoning
    '''
    # 1 select data
    class_list = ['fire_engine', 'ambulance', 'school_bus']
    select_class = 'fire_engine'
    # select_root = os.path.join('/lab/tmpig23b/u/andy/VR_SC/ACEdata/results/img2vec/',
    #                               'google_inveption_v1', select_class + '_detect_graph')  # all image
    select_root = os.path.join('/lab/tmpig23b/u/yao_data/VR_SC/ACEdata/results/Xception/block14_sepconv2_act/img2vec',
                               'duplicate',   select_class + '_detect_graph')  # all image
    for roots, dirs, files in os.walk(select_root):
        for i, file in enumerate(files):  # we only need limited number for each class
            if '_graph' in file:
                graph_path = os.path.join(roots, file)
                data_package = select_data(graph_path, select_class) # contain both positive and negative graph
                reasoning(model_share, data_package, device)
    print('total_grad_out', total_grad_out)




if __name__ == '__main__':
    total_grad_out = []
    total_activation_out = []
    main()
    # print('grad output: ', total_grad_out)