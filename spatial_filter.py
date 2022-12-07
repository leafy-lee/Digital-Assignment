# The implementation of digital image processing assignment 2.
# Random Noise and Spatial Filter

# Spatial Filter
# author: leafy
# 2022-12-4
# last_modified: 2022-12-4

from typing import Tuple, List, Optional
import numpy as np
import logging
import sys
import functools
import cv2
from IPython import embed

level = logging.DEBUG
fmt = "[%(levelname)s] %(asctime)s - %(message)s"
logging.basicConfig(level=level, format=fmt)


def get_median(noised_image: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int, c: int) -> float:
    median_area = noised_image[x_min:x_max + 1, y_min:y_max + 1, c].flatten()
    sorted_area = sorted(median_area)
    n = len(median_area)
    median = sorted_area[n // 2 + 1] if n % 2 else (sorted_area[n // 2] + sorted_area[n // 2 + 1]) / 2
    return median


def get_adaptive_median(noised_image: np.ndarray, x: int, y: int, h: int, w: int, c: int,
                        max_size: int = None) -> float:
    median_area = noised_image[max(0, x - h):x + h + 1, max(0, y - w):y + w + 1, c].flatten()
    max_val, min_val = np.max(median_area), np.min(median_area)
    sorted_area = sorted(median_area)
    n = len(median_area)
    median = sorted_area[n // 2 + 1] if n % 2 else (int(sorted_area[n // 2]) + int(sorted_area[n // 2 + 1])) // 2
    if min_val < median < max_val:
        return noised_image[x][y][c] if min_val < noised_image[x][y][c] < max_val else median
    elif h < max_size and w < max_size:
        return get_adaptive_median(noised_image, x, y, h + 1, w + 1, c, max_size)
    else:
        return noised_image[x][y][c] if min_val < noised_image[x][y][c] < max_val else median


def get_mean(noised_image: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
    return np.mean(noised_image[x_min:x_max + 1, y_min:y_max + 1], axis=(0, 1))


class SpatialFilter:
    """
    This is description which will be showed
    The Class of Padding Adder
    """
    instance = None                                 # There should be ONLY ONE Spatial Filter once
    __slots__ = ['noised_image', 'kernel_size']

    def __new__(cls, *args):
        if cls.instance is None:  # Once no adder exists
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, noised_image: np.ndarray, kernel_size: int = 3):
        self.noised_image = noised_image
        self.kernel_size = kernel_size

    def reverse_image(self, image: np.ndarray, reflect_size: int = None, axis: int = 0) -> np.ndarray:
        """
        reverse image in one axis for reflect padding
        :param:     image:          Input image for reflection
        :param:     reflect_size:   The size of reflect area edge + 1
        :param:     axis:           The axis that will be reflected
        :return:    reversed_img:   Reversed image at axis
        """
        if reflect_size is None:
            reflect_size = self.kernel_size
            logging.info(f"No kernel size input, automatically use size = {self.kernel_size}")
        if axis == 0:
            reflect_area_left = image[reflect_size - 1::-1]
            reflect_area_right = image[-1 * reflect_size:][::-1]
        else:
            reflect_area_left = image[:, reflect_size - 1::-1]
            reflect_area_right = image[:, -1 * reflect_size:][:, ::-1]
        reversed_img = np.concatenate([reflect_area_left, image, reflect_area_right], axis=axis)
        return reversed_img

    def add_reflect_padding(self, image: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """
        Adding reflect padding
        :param:     image:          Input image for reflection
        :param:     kernel_size:    The size of reflect kernel
        :return:    reversed_img:   Padded image.
        """
        if kernel_size is None:
            kernel_size = self.kernel_size
            logging.info(f"No kernel size input, automatically use size = {self.kernel_size}")
        horizontal_flip = self.reverse_image(image, kernel_size, 0)
        reversed_img = self.reverse_image(horizontal_flip, kernel_size, 1)
        return reversed_img

    def add_zero_padding(self, image: np.ndarray, kernel_size: int = None):
        """
        Adding zero padding
        :param:     image:          Input image for padding
        :param:     kernel_size:    The size of padding kernel
        :return:    output_img:     Padded image.
        """
        if kernel_size is None:
            kernel_size = self.kernel_size
            logging.info(f"No kernel size input, automatically use size = {self.kernel_size}")
        if len(image.shape) > 2:
            h, w, c = image.shape
            output_img = np.zeros((h + 2 * kernel_size, w + 2 * kernel_size, c))
        else:
            h, w = image.shape
            output_img = np.zeros((h + 2 * kernel_size, w + 2 * kernel_size))
        output_img[kernel_size:h + kernel_size, kernel_size:w + kernel_size] = image
        return output_img

    def _preprocess_args(self,
                         noised_image: np.ndarray = None,
                         kernel_size: int = None,
                         padding_func: Optional = None,
                         padding_type: str = None,
                         max_size: int = None):
        assert (padding_type is None or padding_func is None)
        assert (kernel_size % 2)
        if noised_image is None:
            noised_image = self.noised_image
        if kernel_size is None:
            kernel_size = self.kernel_size
        if max_size is None:
            max_size = self.kernel_size + 4
        if padding_func is not None:
            padded_image = padding_func(noised_image, (kernel_size - 1) // 2)
        elif padding_type is not None:
            padding_func = getattr(sys.modules[__name__].SpatialFilter, "add_%s_padding" % padding_type)
            padded_image = padding_func(self, noised_image, (kernel_size - 1) // 2)
        else:
            padded_image = self.add_zero_padding(noised_image, (kernel_size - 1) // 2)
        depth = (kernel_size - 1) // 2
        c_img = 1
        if len(padded_image.shape) > 2:
            h_img, w_img, c_img = noised_image.shape
        else:
            h_img, w_img = noised_image.shape
        output_img = padded_image.copy()
        return noised_image, padded_image, kernel_size, padding_func, max_size, depth, h_img, w_img, c_img, output_img

    def mean_filter(self,
                    noised_image: np.ndarray = None,
                    kernel_size: int = None,
                    padding_func: Optional = None,
                    padding_type: str = None):
        """
        Mean filter
        Support two type of padding:
            1. define it yourself and put it with padding_func
            2. use the predefined type
        :param:     noised_image:   Input image for padding
        :param:     kernel_size:    The size of padding kernel
        :param:     padding_func:   Padding function defined by yourself
        :param:     padding_type:   Type of padding, use in ["reflect", "zero"]
        :return:    output_img:     Output image after filter
        """
        noised_image, padded_image, kernel_size, padding_func,  _, depth, h_img, w_img, c_img, output_img = \
            self._preprocess_args(noised_image, kernel_size, padding_func, padding_type)
        for h in range(depth, h_img + depth):
            for w in range(depth, w_img + depth):
                output_img[h, w, :] = get_mean(padded_image, h - depth, h + depth, w - depth, w + depth)
        output_img = output_img[depth:h_img + depth, depth:w_img + depth]
        return np.uint8(output_img)

    def median_filter(self,
                      noised_image: np.ndarray = None,
                      kernel_size: int = None,
                      padding_func: Optional = None,
                      padding_type: str = None):
        noised_image, padded_image, kernel_size, padding_func, _, depth, h_img, w_img, c_img, output_img = \
            self._preprocess_args(noised_image, kernel_size, padding_func, padding_type)
        for h in range(depth, h_img + depth):
            for w in range(depth, w_img + depth):
                for c in range(c_img):
                    output_img[h, w, c] = get_median(padded_image, h - depth, h + depth, w - depth, w + depth, c)
        output_img = output_img[depth:h_img + depth, depth:w_img + depth]
        return np.uint8(output_img)

    def adaptive_median_filter(self,
                               noised_image: np.ndarray = None,
                               kernel_size: int = None,
                               max_size: int = None,
                               padding_func: Optional = None,
                               padding_type: str = None):
        noised_image, padded_image, kernel_size, padding_func, _, depth, h_img, w_img, c_img, output_img = \
            self._preprocess_args(noised_image, kernel_size, padding_func, padding_type)
        for h in range(depth, h_img + depth):
            for w in range(depth, w_img + depth):
                for c in range(c_img):
                    output_img[h, w, c] = get_adaptive_median(padded_image, h, w, depth, depth, c, max_size)
        output_img = output_img[depth:h_img + depth, depth:w_img + depth]
        return np.uint8(output_img)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arr = np.array([[i + j * 4 for i in range(4)] for j in range(4)])
    adder = SpatialFilter(arr)
    # print(arr)
    # print()
    # print(adder.reverse_image(arr))
    # print()
    # print(adder.add_reflect_padding(arr))
    # print()
    # module = sys.modules['__main__']
    # reflect_padding_func = functools.partial(getattr(module.SpatialFilter, "add_reflect_padding"), adder)
    # zero_padding_func = functools.partial(getattr(module.SpatialFilter, "add_zero_padding"), adder)
    # print(reflect_padding_func, zero_padding_func)
    # print()
    # print(reflect_padding_func(arr))
    # print()
    # print(zero_padding_func(arr))
    # print()
    from noise import GaussianNoiseGenerator, SaltPepperNoiseGenerator
    input_image = cv2.imread('test3.jpg', 1)
    print(type(input_image))
    # cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('input_image', input_image)
    saltpepper_noise_generator = SaltPepperNoiseGenerator()
    noise_img = saltpepper_noise_generator.add_saltpepper_noise(0.1, 0.1, input_image)
    cv2.namedWindow('noise_img', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('noise_img', noise_img)
    spf = SpatialFilter(noise_img)
    # tar = spf.mean_filter(kernel_size=7)
    # cv2.imshow('mean_filter', tar)
    # tar2_channel_wise = spf.median_filter(kernel_size=7)
    tar2 = spf.adaptive_median_filter(kernel_size=3, max_size=7)
    # cv2.imshow('median_filter', tar2_channel_wise)
    cv2.imshow('adaptive_median_filter', tar2)
    # tar12 = spf.mean_filter(kernel_size=7, padding_type="reflect")
    # cv2.imshow('reflect_mean_filter', tar12)
    # tar12_channel_wise = spf.median_filter(kernel_size=7, padding_type="reflect")
    # cv2.imshow('reflect_median_filter', tar12_channel_wise)
    tar122 = spf.adaptive_median_filter(kernel_size=3, max_size=7, padding_type="reflect")
    cv2.imshow('reflect_adaptive_median_filter', tar122)
    cv2.waitKey(0)
