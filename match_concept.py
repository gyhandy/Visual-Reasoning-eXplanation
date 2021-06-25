"""This script discover concepts given images for each class
by using ACE method. And store the detected concept and position as node and edge features
which can be used to form graphs
Note: For intra-class contrastive learning, we form positive samples and negative semples for each class (eij)
"""

from tcav import utils
import sys
import os
import src.ace_helpers as ace_helpers
from src.ace_match_dummy import ConceptDiscovery
import argparse
import numpy as np
import pickle


def main(args):
    class_dict = {
        'ambulance': 'n02701002',
        'fire_engine': 'n03345487',
        'school_bus': 'n04146614',
    }

    class_list = list(class_dict.keys())  # for collect negative samples
    concept_dict = {
        'ambulance': [10, 6, 18, 13],
        'fire_engine': [8, 17, 16, 15],
        'school_bus': [9, 3, 5, 19]}
    concept_score_dict = {
        'ambulance': [0.84, 0.82, 0.75, 0.7],
        'fire_engine': [0.638, 0.635, 0.5, 0.4],
        'school_bus': [0.92, 0.87, 0.5, 0.42]}


    '''
    each image only go through it's own class eij, extract concepts based on corresponding label
    '''
    for target_class in class_dict.keys():

        '''Load the learnt concept to detect concepts form images after multi-resolution segmentation'''
        path = os.path.join(args.working_dir, 'output', target_class) # detected with Xception
        prior_knowledge_path = os.path.join(path, 'concepts', 'all_concept_dict_X.txt') # with Xception
        test_image_path = os.path.join(args.source_dir, target_class)  # selected image

        '''step 1 load prior knowledge for each class'''
        with open(prior_knowledge_path, "rb") as fp:  # Pickling
            prior_knowledge = pickle.load(fp)

        ''' these file used for Concept Discovery 'cd' class to do multi-resolution segmentation '''
        cavs_dir = os.path.join(args.working_dir, 'concept_result', target_class + '_results', 'cavs/')
        activations_dir = os.path.join(args.working_dir, 'concept_result', target_class + '_results', 'acts/')

        random_concept = 'random_discovery'  # Random concept for statistical testing??


        sess = utils.create_session()
        if args.model_to_run != 'Xception':
            mymodel = ace_helpers.make_model(
              sess, args.model_to_run, args.model_path, args.labels_path)
        else:
            mymodel = ace_helpers.make_model(
                None, args.model_to_run, None, args.labels_path)
        # Creating the ConceptDiscovery class instance
        if args.bottlenecks:
            args.bottlenecks.split(',')
        else:
            bottlenecks = mymodel.find_target_layer()
        cd = ConceptDiscovery(
          mymodel,
          target_class, # we run a for loop
          random_concept,
          bottlenecks,
          sess,
          args.source_dir,
          activations_dir,
          cavs_dir,
          num_random_exp=args.num_random_exp,
          channel_mean=True,
          max_imgs=args.max_imgs,
          min_imgs=args.min_imgs,
          num_discovery_imgs=args.max_imgs,
          num_workers=args.num_parallel_workers)
        # Creating the dataset of image patches

        ''' step 2 Load images for each class and do multi-resolution segmentation 
              process more than one test image, propose patches'''
        # these are just stored image to save time of loading image
        saved_imgpath_file = os.path.join(args.working_dir, 'output', target_class,
                                          target_class + '_img_paths.npy')
        saved_img_file = os.path.join(args.working_dir, 'output', target_class,
                                      target_class + '_img.npy')
        if os.path.exists(saved_imgpath_file) and os.path.exists(saved_img_file):
            test_imgs = np.load(saved_img_file)
            test_imgs_paths = np.load(saved_imgpath_file)
            test_imgs_paths = test_imgs_paths.tolist()
        else:
            test_imgs, test_imgs_paths = cd.load_concept_imgs(
                test_image_path, 400)  # we use only 400 images (no split train and test)
            np.save(saved_img_file, test_imgs)
            np.save(saved_imgpath_file, test_imgs_paths)
        '''# for each image, segment and detect concepts, positions and activation at ori model'''
        for n, test_img in enumerate(test_imgs): # select how much you need
            cd.create_patches(param_dict={'n_segments': [15, 50, 80]}, discovery_images=test_img[np.newaxis, :], gradcam=True)
            # Saving the concept discovery target class images
            print(target_class, n, 'image finished')

            ''' step 3 match, find nearest patch for each concept center'''
            cd.match_dummy(prior=prior_knowledge, target=concept_dict[target_class],
                           method='KM', param_dicts={'n_clusters': 25})  # 25: number of concept

            ''' step 4 store the result of discovered concepts, position and activateion'''
            for bn in cd.bottlenecks:  # we use only mixed4c layer to cluster concept
                # store address
                img_address = os.path.join(args.working_dir, 'img2vec', 'new',  target_class + '_detect_img')
                if not os.path.exists(img_address):
                    os.makedirs(img_address)
                graph_address = os.path.join(args.working_dir, 'img2vec', 'new', target_class + '_detect_graph')  # with edges
                if not os.path.exists(graph_address):
                    os.makedirs(graph_address)
                XAI_address = os.path.join(args.working_dir, 'img2vec', 'new', target_class + '_detect_graph')
                if not os.path.exists(graph_address):
                    os.makedirs(graph_address)
                filename = test_imgs_paths[n].split('/')[-1].split('.')[0]


                # save XAI
                '''
                XAI part
                process input image to [ori, ori-concept1, ori-concept2,ori-concept3...]
                '''
                processed_img = ace_helpers.get_concept_img(cd, concept_dict[target_class])  # [N , 224, 224, 3]
                '''get the activation and save for the following training'''
                activation_vec = ace_helpers.get_linears_from_images(processed_img, cd.model)  # N * 1008
                activation_save_path = os.path.join(XAI_address, filename + '_XAI' + '.npy', )
                np.save(activation_save_path, activation_vec)

                # save both image and graph
                ace_helpers.plot_target_match_concepts_test_dummy_edge(cd, tcav_score=concept_score_dict[target_class],
                                                                       num=1,
                                                                       img_address=img_address,
                                                                       graph_address=graph_address,
                                                                       filename=filename)

        '''
        Negative creation!
        For each class CD, we pass all images and detect wrong class image for potential XAI and answer why not
        '''
        remain_class = class_list.copy()
        remain_class.remove(target_class)
        for negative_class in remain_class:  # if target is fire, now only ambu and school left
            test_image_path = os.path.join(args.source_dir, negative_class)  # selected image
            saved_imgpath_file = os.path.join(args.working_dir, 'output', negative_class,
                                              negative_class + '_img_paths.npy')
            saved_img_file = os.path.join(args.working_dir, 'output', negative_class,
                                          negative_class + '_img.npy')
            if os.path.exists(saved_imgpath_file) and os.path.exists(saved_img_file):
                test_imgs = np.load(saved_img_file)
                test_imgs_paths = np.load(saved_imgpath_file)
                test_imgs_paths = test_imgs_paths.tolist()
            else:
                test_imgs, test_imgs_paths = cd.load_concept_imgs(
                    test_image_path, 400)  # we use only 400 images (no split train and test)
                np.save(saved_img_file, test_imgs)
                np.save(saved_imgpath_file, test_imgs_paths)
            '''# for each image, segment and detect concepts, positions and activation at ori model'''
            save_class_name = negative_class + '2' + target_class # put a real negative class detected as target class
            for n, test_img in enumerate(test_imgs[0:200]):
                cd.create_patches(param_dict={'n_segments': [15, 50, 80]}, discovery_images=test_img[np.newaxis, :],
                                  gradcam=False)
                # Saving the concept discovery target class images
                print(save_class_name, n, 'image finished')

                ''' step 3 match, find nearest patch for each concept center'''
                cd.match_dummy(prior=prior_knowledge, target=concept_dict[target_class],
                               method='KM', param_dicts={'n_clusters': 25})  # 25: number of concept

                ''' step 4 store the result of discovered concepts, position and activateion'''
                for bn in cd.bottlenecks:  # we use only mixed4c layer to cluster concept
                    # store address
                    img_address = os.path.join(args.working_dir, 'img2vec', save_class_name + '_detect_img')
                    if not os.path.exists(img_address):
                        os.makedirs(img_address)
                    graph_address = os.path.join(args.working_dir, 'img2vec',
                                                 save_class_name + '_detect_graph')  # with edges
                    if not os.path.exists(graph_address):
                        os.makedirs(graph_address)
                    XAI_address = os.path.join(args.working_dir, 'img2vec', save_class_name + '_detect_graph')
                    if not os.path.exists(graph_address):
                        os.makedirs(graph_address)
                    filename = test_imgs_paths[0:200][n].split('/')[-1].split('.')[0]

                    # save both image and graph
                    ace_helpers.plot_target_match_concepts_test_dummy_edge(cd,
                                                                           tcav_score=concept_score_dict[target_class],
                                                                           num=1,
                                                                           img_address=img_address,
                                                                           graph_address=graph_address,
                                                                           filename=filename)




def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='source',
                        help='''Directory where the network's classes image folders and random concept folders are saved.''')
    parser.add_argument('--working_dir', type=str,
                        help='Directory where you save the results of discover_concept.py', default='result')

    parser.add_argument('--model_to_run', type=str,
                        help='The name of the model.', default='Xception')
    parser.add_argument('--model_path', type=str,
                        help='Path to model checkpoints.', default='')
    parser.add_argument('--labels_path', type=str,
                        help='Path to model checkpoints.', default='src/Xception_labels.json')
    parser.add_argument('--target_class', type=str,
                        help='The name of the target class to be interpreted', default='zebra')
    parser.add_argument('--bottlenecks', type=str,
                        help='Names of the target layers of the network (comma separated)',
                        default='')
    parser.add_argument('--num_random_exp', type=int,
                        help="Number of random experiments used for statistical testing, etc",
                        default=20)
    parser.add_argument('--max_imgs', type=int,
                        help="Maximum number of images in a discovered concept",
                        default=50)
    parser.add_argument('--min_imgs', type=int,
                        help="Minimum number of images in a discovered concept",
                        default=40)
    parser.add_argument('--num_parallel_workers', type=int,
                        help="Number of parallel jobs.",
                        default=0)

    ''' test mode for single image'''
    parser.add_argument('--test_img_path', type=str,
                        help='Path to test_img_path.',
                        default='test')
    parser.add_argument('--test_img_class', type=str,
                        help='the class for test image', default='School_bus')
    return parser.parse_args(argv)



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

