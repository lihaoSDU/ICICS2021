#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2020/8/20 18:47
# @Author  : Hao Li
# @Lab     :
# @File    : fuzzer_main.py
# **************************************

"""usage:
python3 fuzzer_main.py --input_data ../../datasets/office_home/RealWorld/test/ --output_dir ./data/fuzzed_data/ \
--models_dir ./models/office_home/
"""

import torch
import os
import argparse
import shutil
import time

from utils.data_loader import load_data, load_testing_data
from coverage import CoverageTable
from lib import Fuzzer


parser = argparse.ArgumentParser(description='Generate adversarial samples by improving '
                                             'DeepXplore fuzzing method')

parser.add_argument('--model_type', type=int, default=1, choices=[0, 1],
                    help='model type for fuzzing (normal DNN: 0, transfer learning: 1)')

parser.add_argument('--noise_param', type=int, default=1, choices=[1, 2],
                    help='noise params to choice noise addition mode')

parser.add_argument('--seed', type=int, default=233, metavar='S',
                    help='random seed (default: 233)')

parser.add_argument('--iters_num', type=int, default=50,
                    help='number of iteration to fuzz')

# user config
parser.add_argument('--input_data', type=str, default=None,
                    help='path of image seeds data')

parser.add_argument('--output_dir', type=str, default=None,
                    help='output fuzzed adversarial samples path')

parser.add_argument('--input_model', type=str, default=None,
                    help='model for fuzzing')

args = parser.parse_args()

torch.manual_seed(args.seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    """ load local trained models to fuzz """

    if args.model_type == 0:
        model = torch.load(args.input_model)
        model.to(DEVICE).eval()

        return model

    elif args.model_type == 1:
        # transfer learning model
        # choice: alexnet, resnet{34, 50, 101}, vgg{16, 19}
        from net import ResNet

        model = ResNet.DSAN(num_classes=65)
        model.load_state_dict(torch.load(args.input_model))
        model.to(DEVICE).eval()

        return model


def main():

    # load data for fuzzing with one batch size
    data_loader, class_to_idx, dataset_sizes = load_data(data_folder=args.input_data, batch_size=1,
                                                         train_flag=False, kwargs={'num_workers': 0})

    # load models for fuzzing
    model = load_model()

    # initial neuron coverage table
    coverage_table_init = CoverageTable.CoverageTableInit(model)
    model_layer_dict, model_layer_names = coverage_table_init.init_deepxplore_coverage_tables()

    # params of Fuzzer
    data = [data_loader, class_to_idx]

    # remove if exist
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)

    fuzzer = Fuzzer.Fuzzer(data, args, model, model_layer_dict, model_layer_names, DEVICE)
    start_time = time.time()
    fuzzer.loop()
    end_time = time.time()
    print('time cost: {}'.format(end_time - start_time))


if __name__ == '__main__':

    main()
