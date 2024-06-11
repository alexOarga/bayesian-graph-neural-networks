'''
Copyright (C) 2019. Huawei Technologies Co., Ltd and McGill University. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the MIT License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
MIT License for more details.
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from BGCN.src.GNN_models import GnnModel
from BGCN.src.data_partition import data_partition_fix, data_partition_random
import random
from BGCN.src.utils import save_log_func
import argparse
import os
from BGCN.src.flags import flags

import csv

code_path = os.path.abspath('')
FLAGS = flags()


with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='cora', help='which dataset to use')
        parser.add_argument('--label_n_per_class', type=int, default=10, help='trial index')
        parser.add_argument('--data_partition_seed', type=int, default=101,
                            help='The seed to use split the data for trial.')
        parser.add_argument('--trial_index', type=int, default=0, help='trial index')
        parser.add_argument('--model_name', type=str, default='BGCN', help='which model we use for training (GCN or BGCN)')
        parser.add_argument('--save_log', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False,
                            help='Save log or not')
        parser.add_argument('--random_partition', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True,
                            help='Save log or not')
        parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')

        label_values = [5,10]
        random_partition_values = ['1','0']
        num_runs = 10

        # Escribir encabezado opcional
        writer.writerow(['label_value', 'random_partition_value', 'run', 'value'])

        results = np.zeros((len(label_values)+1, len(random_partition_values)+1, num_runs+1))

        for label_value in label_values:
            for random_partition_value in random_partition_values:
                for run in range(num_runs):

                    args = parser.parse_args()
                    data_partition_seed = args.data_partition_seed
                    trial_index = args.trial_index
                    dataset = args.dataset
                    model_name = args.model_name
                    save_log = args.save_log
                    random_partition = random_partition_value
                    label_n_per_class = label_value
                    gpu = args.gpu
                    acc_sample_graph = 0

                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

                    print("Dataset: {}".format(dataset))
                    print("Model: {}".format(model_name))
                    print("Trial index: {}".format(trial_index))
                    print("Data partition seed: {}".format(data_partition_seed))
                    print("Label Value: {}".format(label_value))
                    print("Random partition value: {}".format(random_partition_value))
                    print("Run: {}".format(run))
                    if save_log:
                        file_name = dataset + '_' + model_name + '_softmax_trail_' + str(trial_index) + '_random_seed_' + str(
                        data_partition_seed) + '.txt'
                        print("Save log mode activated, training log will be saved to /log/" + file_name)

                    
                    # ==================================Set random seed for result reproduce===============================

                    tf.set_random_seed(data_partition_seed)
                    np.random.seed(data_partition_seed)
                    random.seed(data_partition_seed)

                    # =============================================Save log=================================================

                    if save_log:
                        save_log_func(code_path, dataset, model_name, trial_index, data_partition_seed)

                    # =============================Load data=================================================

                    dataset_dir = code_path + '' + '/data'
                    if not random_partition:
                        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = data_partition_fix(
                            dataset_dir=dataset_dir, dataset_name=dataset, label_n_per_class=label_n_per_class)
                    elif random_partition:
                        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = data_partition_random(
                            dataset_dir=dataset_dir, dataset_name=dataset, label_n_per_class=label_n_per_class)
                    else:
                        "Wrong input data format"

                    # ==================================Train Model===========================================
                    print("Inicializo_modelo")
                    GNN_Model = GnnModel(FLAGS, features, labels, adj, y_train, y_val, y_test, train_mask, val_mask,
                                        test_mask, acc_sample_graph, checkpt_name='model_1', model_name=model_name)
                    GNN_Model.model_initialization()
                    print("Empiezo_training")
                    

                    int_rpv = int(random_partition_value)
                    int_lv = int(label_value/5)
                    results[int_lv, int_rpv , run] =GNN_Model.train()
                    writer.writerow([int_lv,  int_rpv, run, results[int_lv, int_rpv, run]])

