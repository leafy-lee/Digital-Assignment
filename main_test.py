from abc import ABC
from typing import Tuple
import numpy as np
from functools import singledispatchmethod
import cv2
import skimage

from main import *


def main_test(test_dir: str = "./test_image"):
    # Reading images
    i = 1
    j = 2
    path = f"test{i}.jpg"
    input_image = cv2.imread(os.path.join(test_dir, path), 1) / 255
    noise = np.random.normal(0, 0.1 ** 0.5, input_image.shape)
    inner_gaussian_noise_generator = GaussianNoiseGenerator()
    noisy = input_image + noise
    noise = inner_gaussian_noise_generator.generate_gaussian_noise(input_image.shape, 0, 0.1)
    noisy12 = input_image + noise
    noise_im_ch = np.clip(noisy, 0.0, 1.0) * 255
    noise_im_12 = np.clip(noisy12, 0.0, 1.0) * 255
    cv2.imshow('noise_im_ch', np.uint8(noise_im_ch))
    cv2.imshow('noise_im_12', np.uint8(noise_im_12))
    # print(noise_im_ch)
    out_im3, _ = mean_filter(noise_im_ch, 3)
    out_im312, _ = mean_filter(noise_im_12, 3)
    cv2.imshow('_im_ch', np.uint8(out_im3))
    cv2.imshow('_im_12', np.uint8(out_im312))
    """
    out_im3, _ = mean_filter(noise_im_ch, 3)
    cv2.imwrite(f'./result/new/mean_gaussian_channel_{i}.png', out_im3)
    out_im31, _ = median_filter(noise_im_ch, 3)
    cv2.imwrite(f'./result/new/median_gaussian_channel_{i}.png', out_im31)
    out_im32, _ = median_adaptive_filter(noise_im_ch, 3, 7)
    cv2.imwrite(f'./result/new/median_adaptive_gaussian_channel_{i}.png', out_im32)

    path = f"test{j}.jpg"
    input_image = cv2.imread(os.path.join(test_dir, path), 1) / 255
    noise = np.random.normal(0, 0.1 ** 0.5, input_image.shape)
    noisy = input_image + noise
    noise_im_ch = np.clip(noisy, 0.0, 1.0) * 255
    # cv2.imshow('noise_im_ch', np.uint8(noise_im_ch))
    # print(noise_im_ch)
    out_im3, _ = mean_filter(noise_im_ch, 3)
    cv2.imwrite(f'./result/new/mean_gaussian_channel_{j}.png', out_im3)
    # cv2.imshow('out_im3', np.uint8(out_im3))
    out_im31, _ = median_filter(noise_im_ch, 3)
    cv2.imwrite(f'./result/new/median_gaussian_channel_{j}.png', out_im31)
    # cv2.imshow('out_im31', np.uint8(out_im31))
    out_im32, _ = median_adaptive_filter(noise_im_ch, 3, 7)
    cv2.imwrite(f'./result/new/median_adaptive_gaussian_channel_{j}.png', out_im32)
    """
    cv2.waitKey(0)


if __name__ == '__main__':
    # input_image3 = cv2.imread('test3.jpg', 1)
    # a, b = generate_gaussian_noise(input_image3, 0, 0.05)
    # cv2.imshow('gaussian_noise', a)
    # cv2.imshow('gaussian_noise_channel', b)
    # cv2.waitKey(0)
    main_test()
