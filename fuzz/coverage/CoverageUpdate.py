#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2020/8/20 18:47
# @Author  : Hao Li
# @Lab     :
# @File    : CoverageUpdate.py
# **************************************
import torch
import torch.nn as nn
import random

from utils.mutators import scale


class CoverageUpdate():
    def __init__(self, model):
        self.model = model

    def update_nc_coverage(self, input_data, model_layer_dict, layer_names, threshold=0):
        """
        Neuron coverage update proposed from DeepXplore.
        :param model: DNN model
        :param input_data: image
        :param model_layer_dict:
        :param layer_names:
        :param threshold:
        :return:
        """
        model_name = self.model.__class__.__name__
        input = input_data

        for idx, (key, value) in enumerate(layer_names.items()):
            global intermediate_layer_output

            if model_name in ['AlexNet', 'VGG', 'ResNet', 'DSAN'] and isinstance(layer_names.get(key), nn.Linear):
                input = torch.flatten(input, 1)

            if (model_name is 'DSAN' and 'ResNet' in self.model.feature_layers.__class__.__name__) or (model_name is 'ResNet'):
                # resnet model layers
                # print(key, value)

                if 'downsample' not in key:
                    if ('feature_layers.layer' in key) and ('0' in key) and ('conv1' in key):
                        ''' domain adaptation model '''
                        identity = input
                    elif ('layer' in key) and ('0' in key) and ('conv1' in key):
                        ''' normal dnn model '''
                        identity = input
                    # print(key, layer_names.get(key), input.shape)
                    intermediate_layer_output = layer_names.get(key).forward(input)
                
                else:
                    # downsample layer
                    identity = layer_names.get(key).forward(identity)
                    if identity.shape == intermediate_layer_output.shape:
                        intermediate_layer_output += identity
                    else:
                        continue
            else:
                # alexnet, vgg models layer

                intermediate_layer_output = layer_names.get(key).forward(input)

            scaled = scale(intermediate_layer_output[0])

            for num_neuron in range(scaled.shape[-1]):
                if torch.mean(scaled[..., num_neuron]) > threshold and \
                        not model_layer_dict[(list(layer_names.keys())[idx], num_neuron)]:
                    # model_layer_dict[(list(layer_names.keys())[idx], num_neuron)] = True
                    model_layer_dict[(list(layer_names.keys())[idx], num_neuron)] = True

            input = intermediate_layer_output

        return model_layer_dict

    def neuron_covered(self, model_layer_dict):

        covered_neurons = len([v for v in model_layer_dict.values() if v])

        total_neurons = len(model_layer_dict)

        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def neuron_to_cover(self, model_layer_dict):

        not_covered = [(layer_name, index, v) for (layer_name, index), v in model_layer_dict.items() if not v]

        if not_covered:
            layer_name, index, v = random.choice(not_covered)
        else:
            layer_name, index, v = random.choice([(layer_name, index, v) for (layer_name, index), v in model_layer_dict.items()])
            # layer_name, index, v = random.choice(model_layer_dict.keys())

        print("model nueron number: {}, uncovered number: {}, layer_name: {}, index: {}, flag: {}".format(
            len(model_layer_dict.items()), len(not_covered), layer_name, index, v))

        return layer_name, index
