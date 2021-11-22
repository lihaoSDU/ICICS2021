#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2020/8/20 18:47
# @Author  : Hao Li
# @Lab     :
# @File    : mutators.py
# **************************************
from __future__ import print_function
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_gradient_magnitude
from skimage.util import random_noise


def image_noise(img, params, grads, pixel=800):

    if params == 1:
        # first interpolate cdf noise, then multiple upsampling image

        # 直方图均衡化
        def histeq(im):

            imhist, bins = np.histogram(im.detach().numpy().flatten(), bins=256)
            cdf = imhist.cumsum()
            cdf = cdf / cdf[-1]
            new_im = np.interp(im.detach().numpy().flatten(), bins[:-1], cdf)

            return new_im

        # channel = 3
        grads = np.array([histeq(grads[0]).reshape(grads[0].shape),
                          histeq(grads[1]).reshape(grads[1].shape),
                          histeq(grads[2]).reshape(grads[2].shape)])

        gauss_noise = torch.tensor([gaussian_gradient_magnitude(grads, sigma=2)])

        img = torch.add(img, torch.tensor(grads))

        batch, ch, row, col = img.shape
        up = nn.Upsample(size=(pixel, pixel), mode='bilinear', align_corners=True)
        img_up = up(img) + up(gauss_noise)
        down = nn.Upsample(size=(row, col), mode='bilinear', align_corners=True)
        new_img = down(img_up)

        return torch.sub(new_img, img)

    elif params == 2:
        # Gaussian-distributed (laplace) additive noise.

        up = nn.Upsample(size=(pixel, pixel), mode='bilinear', align_corners=True)
        img_up = up(img)

        batch, ch, row, col = img_up.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = torch.tensor(np.random.normal(mean, sigma, (batch, ch, row, col))).to(dtype=torch.float)
        # laplace = torch.tensor(np.random.laplace(mean, sigma, (batch, ch, row, col))).to(dtype=torch.float)
        new_img = torch.add(img_up, gauss * 0.5)

        batch, ch, row, col = img.shape
        down = nn.Upsample(size=(row, col), mode='bilinear', align_corners=True)
        new_img = down(new_img)

        return new_img


def image_blur(image, params):

    # print("blur")
    img = image[0].detach().numpy()
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    # if params == 9:
    #     blur = cv2.blur(img, (6, 6))
    if params == 9:
        blur = cv2.bilateralFilter(img, 6, 50, 50)
        # blur = cv2.bilateralFilter(img, 9, 75, 75)
    image_blur = torch.tensor(blur).view(image.shape)
    return image_blur.detach().numpy()


def constraint_black(gradients, rect_shape=(10, 10)):

    start_point = (torch.randint(0, gradients.shape[1] - rect_shape[0], (1,)),
                   torch.randint(0, gradients.shape[2] - rect_shape[1], (1,)))

    new_grads = torch.zeros_like(gradients)

    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
            start_point[1]:start_point[1] + rect_shape[1]]

    if torch.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -torch.ones_like(patch)

    return new_grads


def constraint_occl(gradients, start_point, rect_shape):

    new_grads = torch.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]] = \
        gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]

    return new_grads


def salt_pepper_noise(img):

    s_vs_p = 0.5
    amount = 0.004
    new_img = np.copy(img.detach().numpy())

    # Salt mode
    num_salt = np.ceil(amount * img.detach().numpy().size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
    new_img[tuple(coords)] = new_img.min()

    # Pepper mode
    num_pepper = np.ceil(amount * img.detach().numpy().size * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
    new_img[tuple(coords)] = new_img.max()

    return torch.tensor(new_img, dtype=torch.float, requires_grad=True)


def scale(x, rmax=1, rmin=0):
    """
    :param x: Array
    :param rmax:
    :param rmin:
    :return:
    """

    X_std = (x - x.min()) / (x.max() - x.min())

    X_scaled = X_std * (rmax - rmin) + rmin

    return X_scaled
