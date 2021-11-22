#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2020/8/20 18:50
# @Author  : Hao Li
# @Lab     :
# @File    : Fuzzer.py
# **************************************
import torch
import torch.nn.functional as F
import torchvision
import os
import random
from copy import deepcopy
import pytorch_msssim as msssim

from coverage import CoverageUpdate
from utils.config import bcolors
from utils import mutators


class Fuzzer(object):

    def __init__(self, data, args, model, model_layer_dict, model_layer_names, DEVICE):

        self.data_loader = data[0]
        self.class_to_idx = data[1]
        self.model_type = args.model_type  # normal DNN model or DA model
        self.output_dir = args.output_dir
        self.args = args
        self.device = DEVICE
        self.model = model
        self.model_layer_dict = model_layer_dict
        self.model_layer_names = model_layer_names

        self.clip_min = -1.
        self.clip_max = 1.
        self.purb = .5
        self.s = 1.
        self.levels = 256
        self.quantize = True

    def loop(self):

        # average structural similarity
        ASS = 0

        for idx, (image, labels, img_path) in enumerate(self.data_loader):
            img_name = os.path.split(img_path[0])[-1]

            image.requires_grad = True
            labels = labels.to(self.device)
            init_image = deepcopy(image)

            output = self.predict(image)  # probability [batch_size, classes_number]
            # loss function
            loss = F.cross_entropy(output, labels)

            # adversarial images saved dir
            global image_dir
            for key, value in self.class_to_idx.items():
                if labels.item() == value:
                    image_dir = os.path.join(self.args.output_dir, key)
                    if not os.path.isdir(image_dir):
                        os.makedirs(image_dir)

            update_coverage = CoverageUpdate.CoverageUpdate(self.model)

            print()
            print('********************* start fuzzing: {} *********************'.format(idx))
            print()

            # coordinate of constraint occlusion
            k = 2
            x = random.sample(range(image.shape[-1]), k)
            y = random.sample(range(image.shape[-1]), k)

            worst_norm = image.flatten(1).norm(p=2, dim=1)  # condition norm
            adv_noise = torch.zeros_like(image)
            tf_noise = torch.zeros_like(image)

            for iter in range(1, self.args.iters_num + 1):

                # choice uncovered neuron to cover
                self.layer_name, self.index = update_coverage.neuron_to_cover(self.model_layer_dict)
                weight = self.model_layer_names.get(self.layer_name).state_dict().get('weight')
                loss_neuron = torch.mean(weight[self.index, ...])

                # final loss
                alpha = -1.0
                final_loss = alpha * loss + loss_neuron

                self.model.zero_grad()
                final_loss.backward()

                # grads init
                grads = F.normalize(image.grad.data)

                # """ mutation strategy of deepxplore (dx) """
                for i in range(k):
                    # grads_value = mutators.constraint_black(gradients=grads)
                    dx_noise = mutators.constraint_occl(gradients=grads[0], start_point=(x[i], y[i]),
                                                                rect_shape=(10, 10))
                    '''
                    adv_noise.add_(dx_noise)
                    '''
                image = torch.add(image, adv_noise)

                # """ mutation strategy of tranfuzz (tf) """
                tf_noise = mutators.image_noise(image, self.args.noise_param, grads[0])

                # referenced from advertorch.ddn
                tf_noise.add_(image)
                if self.quantize:
                    tf_noise.sub_(self.clip_min).div_(self.s)
                    tf_noise.mul_(self.levels - 1).round_().div_(self.levels - 1)
                    tf_noise.mul_(self.s).add_(self.clip_min)
                adv_noise.clamp_(self.clip_min, self.clip_max).sub_(image)

                # adv_noise addition
                adv_noise.add_(tf_noise)

                # horizontal flip transformation of adv_noise
                # rotate noise over 180 degrees and add to image

                image = image + torch.rot90(adv_noise, 2, [1, 3]) * self.purb

                image = torch.tensor(image, dtype=torch.float, requires_grad=True)

                # re-predict to adv_image
                output = self.predict(image)
                loss = F.cross_entropy(output, labels)
                pred_labels = torch.max(output, 1)[1]

                # distance calculator
                ssim = msssim.SSIM(data_range=255, size_average=False, channel=3)
                dist = ssim(init_image, image)
                print('iter: {}, images similarity: {:.4f} \n'.format(iter, torch.mean(dist).item()))

                # objective function
                condition_pred = (pred_labels != labels)  # supervised mode
                condition_max_sim = torch.mean(dist).item() < 0.995 * torch.mean(ssim(init_image, init_image)).item()
                condition_min_sim = torch.mean(dist).item() < 0.9 * torch.mean(ssim(init_image, init_image)).item()
                condition_max_iter = iter >= self.args.iters_num
                l2_diff = torch.sub(image.flatten(1).norm(2, dim=1), worst_norm).item()
                print(l2_diff)

                if condition_pred :

                    # update neuron coverage
                    self.coverage_update(update_coverage, image)
                    # save adversarial image
                    torchvision.utils.save_image(image, os.path.join(image_dir, str(img_name)),
                                                 normalize=True, scale_each=True)
                    ASS += dist

                    break

                # over maximum iterations
                if condition_max_iter or condition_min_sim:
                    torchvision.utils.save_image(image, os.path.join(image_dir, str(img_name)),
                                                 normalize=True, scale_each=True)
                    ASS += dist
                    break
        print('ASS: {}'.format(ASS / len(self.data_loader)))
        return 0

    def coverage_update(self, update_coverage, image):
        # update neuron coverage
        self.model_layer_dict[(self.layer_name, self.index)] = True

        self.model_layer_dict = update_coverage.update_nc_coverage(image.to(self.device),
                                                                   self.model_layer_dict,
                                                                   self.model_layer_names,
                                                                   threshold=0)

        covered_neurons, total_neurons, covered_neurons_rate = update_coverage.neuron_covered(self.model_layer_dict)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f' % (total_neurons, covered_neurons)
              + bcolors.ENDC)

    def predict(self, image):
        output = None
        if self.args.model_type == 0:
            # normal DNN model
            output = self.model(image.to(self.device))

        elif self.args.model_type == 1:
            # domain adaptation model
            output = self.model.predict(image.to(self.device))

        return output
