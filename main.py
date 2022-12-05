# The implementation of digital image processing assignment 2.
# Random Noise and Spatial Filter
# author: leafy
# 2022-12-4
# last_modified: 2022-12-5
import os
from typing import Tuple
import cv2
import numpy as np
import sys
from itertools import product

from noise import GaussianNoiseGenerator, SaltPepperNoiseGenerator
from spatial_filter import SpatialFilter
from metric import compare

from IPython import embed


def generate_gaussian_noise(input_img: np.ndarray, mean: float, var: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param:     input_img:      Input image.
    :param:     mean:           Mean of gaussian noise.
    :param:     var:            Variance of gaussian noise.
    :return:    output_image:   Output image with gaussian noise added.
    """
    inner_gaussian_noise_generator = GaussianNoiseGenerator()
    output_img = inner_gaussian_noise_generator.add_all_channel_noise(input_img, mean, var)
    output_img_channel_wise = inner_gaussian_noise_generator.add_channel_wise_noise(input_img, mean, var)
    return output_img, output_img_channel_wise


def generate_saltpepper_noise(prob_1: float, prob_2: float, input_img: np.ndarray) -> np.ndarray:
    """
    :param:     prob_1:         prob_1 of saltpepper noise.
    :param:     prob_2:         prob_1 of saltpepper noise.
    :param:     input_img:      Input image.
    :return:    output_image:   Output image with saltpepper noise added.
    """
    saltpepper_noise_generator = SaltPepperNoiseGenerator()
    output_img = saltpepper_noise_generator.add_saltpepper_noise(prob_1, prob_2, input_img)
    return output_img


def mean_filter(noised_img: np.ndarray, kernel_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param:     input_img:      Input image.
    :param:     kernel_size:    kernel size of filter.
    :return:    output_image:   Output filtered image.
    """
    sp_filter = SpatialFilter(noised_img)
    output_img_zero = sp_filter.mean_filter(noised_img, kernel_size=kernel_size, padding_type="zero")
    output_img_reflect = sp_filter.mean_filter(noised_img, kernel_size=kernel_size, padding_type="reflect")
    return output_img_zero, output_img_reflect


def median_filter(noised_img: np.ndarray, kernel_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param:     input_img:      Input image.
    :param:     kernel_size:    kernel size of filter.
    :return:    output_image:   Output filtered image.
    """
    sp_filter = SpatialFilter(noised_img)
    output_img_zero = sp_filter.median_filter(noised_img, kernel_size=kernel_size, padding_type="zero")
    output_img_reflect = sp_filter.median_filter(noised_img, kernel_size=kernel_size, padding_type="reflect")
    return output_img_zero, output_img_reflect


def median_adaptive_filter(noised_img: np.ndarray, kernel_size: int = None,
                           max_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param:     input_img:      Input image.
    :param:     kernel_size:    kernel size of filter.
    :param:     max_size:       Max size of adaptive filter.
    :return:    output_image:   Output filtered image.
    """
    sp_filter = SpatialFilter(noised_img)
    output_img_zero = sp_filter.adaptive_median_filter(noised_img, kernel_size=kernel_size,
                                                       max_size=max_size, padding_type="zero")
    output_img_reflect = sp_filter.adaptive_median_filter(noised_img, kernel_size=kernel_size,
                                                          max_size=max_size, padding_type="reflect")
    return output_img_zero, output_img_reflect


def main(test_dir: str = "./test_image"):
    # Reading images
    input_images = os.listdir(test_dir)
    for idx, path in input_images:
        locals()[f"input_image{idx}"] = cv2.imread(os.path.join(test_dir, path), 1)

    # Recording input images
    for i in range(1, 5):
        cv2.imwrite(f'./result/input_image_{i}.png',
                    locals()[f"input_image{i}"])

    # Recording gaussian noise images
    for i in range(1, 5):
        locals()[f"gaussian_img_full_{i}"], locals()[f"gaussian_img_channel_{i}"] = \
            generate_gaussian_noise(locals()[f"input_image{i}"], 0, 0.05)
        for types in ["channel", "full"]:
            cv2.imwrite(f'./result/gaussian_img_{types}_{i}.png',
                        locals()[f"gaussian_img_{types}_{i}"])

    # recording saltpepper noise images
    for i in range(1, 5):
        locals()[f"sp_img_full_{i}"] = generate_saltpepper_noise(0.1, 0.1,
                                                                 locals()[f"input_image{i}"])
        cv2.imwrite(f'./result/sp_img_full_{i}.png',
                    locals()[f"sp_img_full_{i}"])
    for i in range(1, 5):
        locals()[f"low_sp_img_full_{i}"] = generate_saltpepper_noise(0.01, 0.01,
                                                                     locals()[f"input_image{i}"])
        cv2.imwrite(f'./result/low_sp_img_full_{i}.png',
                    locals()[f"low_sp_img_full_{i}"])

    # Filtering and recording the output images
    print(locals().keys())
    for i in range(1, 5):
        for f_type, noise in product(["mean", "median", "median_adaptive"], ["gaussian", "sp", "low_sp"]):
            # for noise in ["gaussian", "sp", "low_sp"]:
            for types in ["channel", "full"]:
                if "sp" in noise:
                    types = "full"
                print(f"Generating images {f_type}_{noise}_{types}_{i}")
                if f_type == "median_adaptive":
                    locals()[f"{f_type}_{noise}_{types}_{i}"], locals()[f"{f_type}_{noise}_{types}_reflect_{i}"] = \
                        getattr(sys.modules[__name__], f"{f_type}_filter")(locals()[f"{noise}_img_{types}_{i}"],
                                                                           kernel_size=3,
                                                                           max_size=7)
                else:
                    locals()[f"{f_type}_{noise}_{types}_{i}"], locals()[f"{f_type}_{noise}_{types}_reflect_{i}"] = \
                        getattr(sys.modules[__name__], f"{f_type}_filter")(locals()[f"{noise}_img_{types}_{i}"],
                                                                           kernel_size=3)
                for pad in ["_reflect", ""]:
                    print(f"Saving images in ./result/{f_type}_{noise}_{types}{pad}_{i}.png")
                    cv2.imwrite(f'./result/{f_type}_{noise}_{types}{pad}_{i}.png',
                                locals()[f"{f_type}_{noise}_{types}{pad}_{i}"])

    for f_type, noise in product(["mean", "median", "median_adaptive", "no_filter"], ["gaussian", "sp", "low_sp"]):
        for types in ["channel", "full"]:
            if "sp" in noise and types == "channel":
                continue
            for pad in ["_reflect", ""]:
                cnt_psnr, cnt_ssim, cnt_mse = 0, 0, 0
                if f_type == "no_filter":
                    filt = ""
                    img = "img_"
                else:
                    filt = f_type + "_"
                    img = ""
                if f_type == "no_filter" and pad != "":
                    continue
                for i in range(1, 5):
                    cur_psnr, cur_ssim, cur_mse = compare(globals()[f"input_image{i}"],
                                                          globals()[f"{filt}{noise}_{img}{types}{pad}_{i}"])
                    for metric in ["psnr", "ssim", "mse"]:
                        globals()[f"cnt_{metric}"] += globals()[f"cur_{metric}"]
                cnt_psnr, cnt_ssim, cnt_mse = cnt_psnr / 4, cnt_ssim / 4, cnt_mse / 4
                print(f"Difference between {filt}{noise}_{img}{types}{pad}.png and input is "
                      f"\n{cnt_psnr} \n{cnt_ssim} \n{cnt_mse}")


if __name__ == '__main__':
    # input_image3 = cv2.imread('test3.jpg', 1)
    # a, b = generate_gaussian_noise(input_image3, 0, 0.05)
    # cv2.imshow('gaussian_noise', a)
    # cv2.imshow('gaussian_noise_channel', b)
    # cv2.waitKey(0)
    main()
