'''
This script use trained model explain Why and Why Not for
'''
import tensorflow as tf
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import os
import torch.nn.functional as f
import heapq
import matplotlib.image as mpimg
import argparse
import matplotlib.pyplot as plt
from src.VR_reasoning_Xception import MyGCNNet_shareW_adap_batch_withhook
from scipy.special import softmax
import sys
import collections
from pylab import arrow
from tcav import utils
import src.ace_helpers as ace_helpers
from src.ace_match_dummy import ConceptDiscovery
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib import ticker, cm
import matplotlib
from torch.nn import Parameter


def hook_fn_backward(module, grad_input, grad_output): # get gradient
    total_grad_out.append(grad_input) # W * X + b
def hook_fn_forward(module, input, output): # get activation of X
    total_activation_out.append(input)

def reasoning(model_share, data_package, device, args):
    model_share.train()
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
    norm_graph_predict = softmax(graph_predict.cpu().detach().numpy())  # normalize
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
    visual_reason = np.sum(visual_related.cpu().detach().numpy(), axis=1)  # (4 each one for a node)
    structure_related = Pos_Attention[:, node_dim * node_num:].resize(edge_num, edge_dim)
    structure_reason = np.sum(structure_related.cpu().detach().numpy(), axis=1)  # (12 each one for a edge)
    '''
    Answer correct or not (based on original model output)
    '''
    reason_dict = collections.defaultdict(dict)
    GT = data_package[0]
    output_list = list(GT)
    predict = output_list[0].argmax() + 1
    if predict == label: # Correct, we answer why correct
        print('\nCorrect prediction: {}'.format(class_list[predict-1]), '\n')
        print('Why correct? positive node and edge are correct.\n',
              'Visual reason:{}'.format(visual_reason),'\n',
              'structure reason:{}\n'.format(structure_reason))
        reason_dict[class_list[predict-1]]["visual_reason"] = visual_reason
        reason_dict[class_list[predict-1]]["structure_reason"] = structure_reason

        with open(os.path.join(args.result_root, "output", "%s_%s.txt" % (args.img_class, args.img_idx)), 'w') as f:
            f.write('Correct prediction: {}\n'.format(class_list[predict-1]))
            f.write('Why correct? positive node and edge are correct.\n\n')
            f.write('Visual reason:{}\n'.format(visual_reason))
            f.write('Structure reason:{}\n\n'.format(structure_reason))

        print('\nStart answering why not')
        all_label = [1, 2, 3]
        remain_label = all_label.copy()
        remain_label.remove(label)
        for neg_label in remain_label: # exclude positive class 0, keep only 1 and 2
            negative_activation = Attention[neg_label - 1]  # 0, bias; 1, Grad_X (1, 188), 2, Grad_W
            visual_related = negative_activation[:, : node_dim * node_num].resize(node_num, node_dim)
            visual_reason = np.sum(visual_related.cpu().detach().numpy(), axis=1)  # (4 each one for a node)
            structure_related = negative_activation[:, node_dim * node_num:].resize(edge_num, edge_dim)
            structure_reason = np.sum(structure_related.cpu().detach().numpy(), axis=1)  # (12 each one for a edge)

            reason_dict[class_list[neg_label-1]]["visual_reason"] = visual_reason
            reason_dict[class_list[neg_label-1]]["structure_reason"] = structure_reason

            print('Why not {}? negative node and edge explain.\n'.format(class_list[neg_label-1]),
                  'Visual reason:{}\n'.format(visual_reason),
                  'structure reason:{}\n'.format(structure_reason))

            with open(os.path.join(args.result_root, "output", "%s_%s.txt" % (args.img_class, args.img_idx)), 'a') as f:
                f.write('Why not {}? negative node and edge explain.\n'.format(class_list[neg_label-1]))
                f.write('Visual reason:{}\n'.format(visual_reason))
                f.write('structure reason:{}\n\n'.format(structure_reason))

    else: # Wrong, we answer why wrong
        print('Wrong prediction: should be {}, wrong predict as {}'.format(class_list[label-1],
                                                                          class_list[predict-1]))
        print('Why wrong? (1)for the GT class, negative node and edge are bad for GT class.\n',
              'Visual reason:{}\n'.format(visual_reason),
              'structure reason:{}\n\n'.format(structure_reason))

        reason_dict[class_list[label-1]]["visual_reason"] = visual_reason
        reason_dict[class_list[label-1]]["structure_reason"] = structure_reason

        with open(os.path.join(args.result_root, "output", "%s_%s.txt" % (args.img_class, args.img_idx)), 'w') as f:
            f.write('Wrong prediction: should be {}, wrong predict as {}\n'.format(class_list[label-1],
                                                                          class_list[predict-1]))
            f.write('Why wrong? (1)for the GT class, negative node and edge are bad for GT class.\n\n')
            f.write('Visual reason:{}\n'.format(visual_reason))
            f.write('Structure reason:{}\n\n'.format(structure_reason))

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

        reason_dict[class_list[predict-1]]["visual_reason"] = visual_reason
        reason_dict[class_list[predict-1]]["structure_reason"] = structure_reason

        with open(os.path.join(args.result_root, "output", "%s_%s.txt" % (args.img_class, args.img_idx)), 'a') as f:
            f.write('Why wrong? (2)for the Predicted class, positive node and edge are bad for Predicted class .\n')
            f.write('Visual reason:{}\n'.format(visual_reason))
            f.write('structure reason:{}\n'.format(structure_reason))
    '''
    show all detected concept
    '''
    print('\nGraph Prediction: {}\n GT: {}'.format(norm_graph_predict, GT))

    total_grad_out.clear()
    total_activation_out.clear()
    return reason_dict

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
            interest_vec = f.normalize(interest_vec_1, p=1, dim=1)  # normalize
            # interest_vec = softmax(interest_vec_1.cpu().detach().numpy())
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
        data = Data(x=x, edge_attr=e, edge_index=edge, y=y)
        img_path = graph_path.replace('_detect_graph', '_detect_img').replace('_graph.txt', '_img.png')
        data_package.append((interest_class, img_path, data))

    return data_package


def compute_locaton(data, img):
    locations = []
    y, x = img.shape[:2]
    locations.append(data.edge_attr[0].cpu().numpy()[:2])
    locations.append(data.edge_attr[0].cpu().numpy()[2:])
    locations.append(data.edge_attr[1].cpu().numpy()[2:])
    locations.append(data.edge_attr[2].cpu().numpy()[2:])
    locations = [(int(loc[0]*x), int(loc[1]*y)) for loc in locations]
    return locations


def select_color(split_list, value):
    if value <= split_list[0]:
        color = 'red'
    elif value <= split_list[1]:
        color = 'orange'
    elif value <= split_list[2]:
        color = 'green'
    else:
        color = 'blue'
    return color


def load_image_from_file(filename, shape):
    img = np.array(Image.open(filename).resize(
        shape, Image.BILINEAR))
    img = np.float32(img) / 255.0
    if not (len(img.shape) == 3 and img.shape[2] == 3):
      return None
    else:
      return img


class show_concept():
    def __init__(self, args):
        # self.bottlenecks = 'block14_sepconv2_act'
        self.data_path = args.result_root  # data path
        self.source_dir = args.source_dir  # data path
        self.model_path = args.model_path
        self.img_size = (299, 299)
        self.class2label = {'fire_engine': 1, 'ambulance': 2, 'school_bus': 3}
        self.label2class = {'1': 'fire_engine', '2': 'ambulance', '3': 'school_bus'}
        self.concept_dict = {
            'ambulance': [18, 20, 11, 13],
            'fire_engine': [8, 17, 16, 15],
            'school_bus': [9, 3, 5, 19]}
        self.cd_init()  # a dic contains all of cd we have, {name:cd}


    def hook_fn_backward(self, module, grad_input, grad_output):  # get gradient
        total_grad_out.append(grad_input)  # W * X + b

    def hook_fn_forward(self, module, input, output):  # get activation of X
        total_activation_out.append(input)

    def cd_init(self):  # TBD have all CD knowledge
        '''image init'''
        model_to_run = 'Xception'
        labels_path = 'src/Xception_labels.json'
        sess = utils.create_session()
        mymodel = ace_helpers.make_model(
            None, model_to_run, None, labels_path)

        self.cd = ConceptDiscovery(
            mymodel,
            target_class="",
            random_concept='random_concept',
            bottlenecks='block14_sepconv2_act',
            sess=sess,
            source_dir=self.source_dir,
            activation_dir="",
            cav_dir="",
            num_random_exp=20,
            channel_mean=True,
            max_imgs=40,
            min_imgs=40,
            num_discovery_imgs=40,
            num_workers=0)

    def cd_assign(self, class_name):
        self.cd.target_class = class_name

    def get_prior_knowledge(self, class_name):
        prior_knowledge_path = os.path.join(self.data_path, 'output', class_name, 'concepts',
                                            'all_concept_dict_X.txt')
        with open(prior_knowledge_path, "rb") as fp:  # Pickling
            prior_knowledge = pickle.load(fp)
        return prior_knowledge


    def show_concept(self, img, class_name, gradcam):
        self.cd_assign(class_name)
        prior_knowledge = self.get_prior_knowledge(class_name)
        self.cd.create_patches(param_dict={'n_segments': [15, 50, 80]}, discovery_images=img[np.newaxis, :], gradcam=gradcam)
        self.cd.match_dummy(prior=prior_knowledge, target = self.concept_dict[class_name] ,method='KM',
                               param_dicts={'n_clusters': 25})

        concept_images = []
        target_concept = list(self.cd.center_match.keys())
        for n, concept in enumerate(target_concept):
            index = self.cd.center_match[concept]
            concept_image = self.cd.dataset[index]
            concept_images.append(concept_image)
        return img, concept_images



def main(args):
    '''
    testing dataset may be same as training
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_share = MyGCNNet_shareW_adap_batch_withhook(2048, interest_class_num=3)  # all class share base structure
    '''
    we will use the trained model
    '''
    model_share.load_state_dict(torch.load(args.model_path))

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
    class_dict = {
        'ambulance': 'n02701002',
        'fire_engine': 'n03345487',
        'school_bus': 'n04146614'
    }

    class_list = ['fire_engine', 'ambulance', 'school_bus']

    image_path = os.path.join(args.source_dir, args.img_class, "%s_%s.JPEG"%(class_dict[args.img_class], args.img_idx))
    graph_path = os.path.join(args.result_root, 'img2vec', args.img_class+'_detect_graph',
                              '{}_{}_graph.txt'.format(class_dict[args.img_class], args.img_idx))
    data_package = select_data(graph_path, args.img_class)  # contain both positive and negative graph
    reason_dict = reasoning(model_share, data_package, device, args)

    concept_score = []
    edge_score = []
    img = load_image_from_file(image_path, (299,299))
    concept_class = show_concept(args)
    for k, v in reason_dict.items():
        loc_data = [loc[2] for loc in data_package[1:] if loc[0].split('2')[-1] == k][0]
        reason_dict[k]['location'] = compute_locaton(loc_data, img)
        edge_list = [abs(x) for x in v["structure_reason"]]
        reason_dict[k]['important_edge'] = heapq.nlargest(6, range(len(edge_list)), edge_list.__getitem__)
        concept_score.extend(v["visual_reason"])
        edge_score.extend(v["structure_reason"][[reason_dict[k]['important_edge']]])

    split_concept = np.percentile(concept_score, (25, 50, 75), interpolation='midpoint')
    split_edge = np.percentile(edge_score, (25, 50, 75), interpolation='midpoint')
    edge_index = data_package[1][2]['edge_index'].cpu().numpy()

    ball_size = 10
    num_concepts, num = 4, 1
    ground_truth = class_list[data_package[0].argmax()]
    for idx, (k, v) in enumerate(reason_dict.items()):
        print('Start plot the visualization for label %s...'%k)
        concept_loc = v['location']
        img, concept_images = concept_class.show_concept(img, k, ground_truth==k)

        fig = plt.figure(figsize=(num * num_concepts, 5.5))
        outer = gridspec.GridSpec(5, num_concepts)
        for i in range(num_concepts):
            color = select_color(split_concept, v["visual_reason"][i])
            ax = plt.Subplot(fig, outer[4, i])
            ax.imshow(concept_images[i])
            ax.set_xticks([])
            ax.set_yticks([])
            title = 'concept {}'.format(i + 1)
            ax.set_title(title, color=color,fontsize= ball_size)
            ax.grid(False)
            fig.add_subplot(ax)

        ax = plt.Subplot(fig, outer[:4, :])
        ax.imshow(img)
        for i in range(len(concept_loc)):
            color = select_color(split_concept, v["visual_reason"][i])
            ax.plot(concept_loc[i][0], concept_loc[i][1], 'o', color=color, markersize=ball_size,alpha=0.8)
            ax.text(concept_loc[i][0], concept_loc[i][1], s=str(i+1), fontsize=ball_size, color='white', verticalalignment='center', horizontalalignment='center')
        edges = v['important_edge']
        for i in range(len(edges)):
            color = select_color(split_edge, v["structure_reason"][edges[i]])
            index_list = edge_index[:, edges[i]]
            # start_x = concept_loc[index_list[0]][0], concept_loc[index_list[1]][0]
            # start_y = concept_loc[index_list[0]][1], concept_loc[index_list[1]][1]
            # ax.plot(start_x, start_y, color=color, linewidth=4,alpha=0.7)
            start = concept_loc[index_list[0]]
            end = concept_loc[index_list[1]]
            dx = abs(end[0]-start[0])
            dy = abs(end[1]-start[1])
            ddx = ball_size/(dx+dy)*dx * np.sign(end[0]-start[0])
            ddy = ball_size/(dx+dy)*dy * np.sign(end[1]-start[1])
            ax.arrow(start[0]+ddx, start[1]+ddy, end[0]-start[0]-2*ddx, end[1]-start[1]-2*ddy, head_width=ball_size, linewidth=int(ball_size/3),fc=color,ec=color,length_includes_head=True,alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        if args.img_class == k:
            title = 'why ' + k.replace('_', ' ')
        else:
            title = 'why not ' + k.replace('_', ' ')
        ax.set_title(title)
        ax.grid(False)
        fig.add_subplot(ax)
        plt.show()

        with tf.gfile.Open(os.path.join(args.result_root, "output", "%s2%s_%s.png"%(args.img_class, k, args.img_idx)), 'w') as f:
            fig.savefig(f,dpi=600)

    fig = plt.figure(figsize=(13, 6.5))
    outer = gridspec.GridSpec(1, 3, wspace=0.00001, hspace=0.00001)
    for idx, (k, v) in enumerate(reason_dict.items()):
        ax = plt.Subplot(fig, outer[idx])
        img = mpimg.imread(os.path.join(args.result_root, "output", "%s2%s_%s.png"%(args.img_class, k, args.img_idx)))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        fig.add_subplot(ax)
        plt.axis('off')
    fig.add_axes([0.87, 0.15, 0.05, 0.7],frameon=False)
    plt.axis('off')
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap="jet"))
    cb.set_ticks([])
    cb.update_ticks()
    plt.text(0.85, 0.20, s='Positive', fontsize=ball_size)
    plt.text(0.84, 0.78, s='Negative', fontsize=ball_size)
    plt.show()
    with tf.gfile.Open(os.path.join(args.result_root, "output", "%s_%s.png" % (args.img_class, args.img_idx)), 'w') as f:
        fig.savefig(f)

    print('Finished analyze! You can find the output files at %s' % os.path.join(args.result_root, "output"))


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str, default='/lab/tmpig23b/u/andy/VR_SC/ACEdata/source',
      help='''Directory where the network's classes image folders and random concept folders are saved.''')
  parser.add_argument('--result_root', type=str, help='directory to results of discover concept.py.',
                      default='result')
  parser.add_argument('--img_class', type=str, help='the class of image you want to test', default='fire_engine')
  parser.add_argument('--img_idx', type=str, help='the idx of image you want to test', default="19835")
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='src/GraphConv_Xception_model.pt')
  return parser.parse_args(argv)



if __name__ == '__main__':
    total_grad_out = []
    total_activation_out = []
    main(parse_arguments(sys.argv[1:]))