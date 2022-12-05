# The implementation of digital image processing assignment 2.
# Random Noise and Spatial Filter

# Noise Generate
# author: leafy
# 2022-12-3
# last_modified: 2022-12-4

from abc import ABC
from typing import Tuple
import numpy as np
from functools import singledispatchmethod
import cv2
# from IPython import embed

# from numpy.typing import ArrayLike

__all__ = [
    "NoiseGenerator",
    "UniformNoiseGenerator",
    "NormalNoiseGenerator",
    "GaussianNoiseGenerator",
    "SaltPepperNoiseGenerator",
]


class NoiseGenerator:
    """The base distribution (without uniform noise)."""
    __slots__ = []

    def __init__(self):
        pass

    @singledispatchmethod
    def generate_noise(self):
        """Generating noise which should be implemented"""
        raise NotImplementedError

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Adding noise to image"""
        image_shape = image.shape
        return image + self.generate_noise(image_shape=image_shape)


class UniformNoiseGenerator(NoiseGenerator, ABC):
    """The base distribution with uniform noise."""
    __slots__ = ['min_val', 'max_val']

    def __init__(self, min_val: float = 0, max_val: float = 1):
        super(UniformNoiseGenerator, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def generate_uniform_noise(self,
                               min_val: float = 0,
                               max_val: float = 1,
                               image_shape: Tuple[int, int] = None) -> np.ndarray:
        """Generating noise
        :param:     image_shape:        Image shape of generated noise
        :return:    output_noise:       ArrayLike noise or single noise
        """
        if min_val is None and max_val is None:
            min_val = self.min_val
            max_val = self.max_val
        assert (min_val < max_val)  # assertion
        distribution_range = max_val - min_val  # getting range of Uniform(0, b - a)
        moving_factor = distribution_range - max_val  # getting move of Uniform(a, b) from Uniform(0, b - a)

        if image_shape is None:  # just generate one noise
            return np.random.rand() * distribution_range - moving_factor
        return np.random.rand(*image_shape) * distribution_range - moving_factor  # ArrayLike output

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Adding noise to image"""
        image_shape = image.shape
        return image + self.generate_uniform_noise(image_shape=image_shape)


class NormalNoiseGenerator(UniformNoiseGenerator, ABC):
    """The base distribution with additive i.i.d. uniform noise."""

    def __init__(self):
        super(NormalNoiseGenerator, self).__init__()

    def generate_normal_noise(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generating Normal noise"""
        uniform_noise_1 = self.generate_uniform_noise(0, 1, image_shape)
        uniform_noise_2 = self.generate_uniform_noise(0, 1, image_shape)
        normal_noise = np.cos(2.0 * np.pi * uniform_noise_1) * np.sqrt(-2.0 * np.log(uniform_noise_2))
        return normal_noise

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Adding noise to image"""
        image_shape = image.shape
        return image + self.generate_normal_noise(image_shape=image_shape)


class GaussianNoiseGenerator(NormalNoiseGenerator, ABC):
    """Gaussian distribution with reparameterization trick."""

    def __init__(self):
        super(GaussianNoiseGenerator, self).__init__()

    def generate_gaussian_noise(self, image_shape: Tuple[int, int], mean: float = 0, var: float = 1) -> np.ndarray:
        """
        Generating Normal noise
        :param      image_shape:        Shape of input image
        :param      mean:               Mean of gaussian
        :param      var:                Variance of gaussian
        :return:    gaussian_noise:     Output noise with same size
        """
        normal_noise = self.generate_normal_noise(image_shape)
        gaussian_noise = normal_noise * np.sqrt(var) + mean
        return gaussian_noise

    def add_all_channel_noise(self, image: np.ndarray, mean: float = 0, var: float = 1) -> np.ndarray:
        """Adding gaussian noise to image with SAME noise in each channel
        :param      image:              input image
        :param      mean:               Mean of gaussian
        :param      var:                Variance of gaussian
        :return:    output_img:         Output image with same size adding gaussian noise
        """
        image_shape = image.shape
        input_img = image / 255
        gaussian_noise = np.expand_dims(self.generate_gaussian_noise(image_shape[:2], mean, var), -1)
        gaussian_noise = gaussian_noise.repeat(3, axis=-1)
        output_img = input_img + gaussian_noise
        output_img[input_img > 1] = 1
        output_img[input_img < 0] = 0
        return output_img * 255

    def add_channel_wise_noise(self, image: np.ndarray, mean: float = 0, var: float = 1) -> np.ndarray:
        """Adding gaussian noise to image with DIFFERENT noise in each channel
        :param      image:              input image
        :param      mean:               Mean of gaussian
        :param      var:                Variance of gaussian
        :return:    output_img:         Output image with same size adding channel-wise gaussian noise
        """
        image_shape = image.shape
        input_img = image / 255
        gaussian_noise = self.generate_gaussian_noise(image_shape, mean, var)
        output_img = input_img + gaussian_noise
        output_img[input_img > 1] = 1
        output_img[input_img < 0] = 0
        return output_img * 255


class SaltPepperNoiseGenerator(UniformNoiseGenerator, ABC):
    """Gaussian distribution with additive i.i.d. uniform noise."""

    def __init__(self):
        super(SaltPepperNoiseGenerator, self).__init__()

    def generate_saltpepper_noise(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        :param:     prob_1:             Probability of "Salt" noise.
        :param:     prob_2:             Probability of "Pepper" noise.
        :param:     image_shape:        Shape of input image.
        :return:    saltpepper_noise:   ArrayLike output salt pepper noise.
        """
        uniform_noise = self.generate_uniform_noise(0, 1, image_shape)
        saltpepper_noise = uniform_noise.copy()
        return saltpepper_noise

    def add_saltpepper_noise(self, prob_1: float, prob_2: float, image: np.ndarray) -> np.ndarray:
        """Adding noise to image"""
        output_img = image.copy()
        image_shape = image.shape
        saltpepper_noise = self.generate_saltpepper_noise(image_shape[:2])
        output_img[saltpepper_noise > 1 - prob_1] = 255
        output_img[saltpepper_noise < prob_2] = 0
        return output_img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uniform_noise_generator = UniformNoiseGenerator()
    print(uniform_noise_generator.generate_uniform_noise(2, 4, (2, 3)),
          uniform_noise_generator.generate_uniform_noise())
    normal_noise_generator = NormalNoiseGenerator()
    print(normal_noise_generator.generate_normal_noise((2, 3)))
    gaussian_noise_generator = GaussianNoiseGenerator()
    print(gaussian_noise_generator.generate_gaussian_noise((2, 3)))
    input_image = cv2.imread('./test_image/test3.jpg', 1)
    print(type(input_image))
    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('input_image', input_image)
    # saltpepper_noise_generator = SaltPepperNoiseGenerator()
    # tar = saltpepper_noise_generator.add_saltpepper_noise(0.12, 0.1, input_image)
    # cv2.imshow('saltpepper_noise', tar)
    gaussian_noise_generator = GaussianNoiseGenerator()
    tar2 = gaussian_noise_generator.add_all_channel_noise(input_image, 0, 0.05)
    tar2_channel_wise = gaussian_noise_generator.add_channel_wise_noise(input_image, 0, 0.05)
    cv2.imshow('gaussian_noise', tar2)
    cv2.imshow('gaussian_noise_channel', tar2_channel_wise)
    cv2.waitKey(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
