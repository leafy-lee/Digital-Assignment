# The implementation of digital image processing assignment 2.
# Random Noise and Spatial Filter

# Metrics
# author: leafy
# 2022-12-5
# last_modified: 2022-12-5

from itertools import product
import math
import numpy as np
from skimage.metrics import structural_similarity
import cv2


def mean_squared_error(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    return float(mse)


def peak_signal_noise_ratio(mse: float) -> float:
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


# Just borrow it from skimage
# will implement soon(for assignment request)
def compare(img1, img2):
    mse = mean_squared_error(img1, img2)
    psnr = peak_signal_noise_ratio(mse)
    ssim = structural_similarity(img1, img2, multichannel=True)
    # print('PSNR：{}，SSIM：{}，MSE：{}'.format(psnr, ssim, mse))
    return psnr, ssim, mse


if __name__ == "__main__":
    for i in range(1, 5):
        globals()[f"input_image{i}"] = cv2.imread(f'./result/input_image_{i}.png')

    # Recording gaussian noise images
    for i in range(1, 5):
        for types in ["channel", "full"]:
            globals()[f"gaussian_img_{types}_{i}"] = cv2.imread(f'./result/gaussian_img_{types}_{i}.png')

    # recording saltpepper noise images
    for i in range(1, 5):
        globals()[f"sp_img_full_{i}"] = cv2.imread(f'./result/sp_img_full_{i}.png',
                                                   )
    for i in range(1, 5):
        globals()[f"low_sp_img_full_{i}"] = cv2.imread(f'./result/low_sp_img_full_{i}.png')

    for i in range(1, 5):
        for f_type, noise in product(["mean", "median", "median_adaptive"], ["gaussian", "sp", "low_sp"]):
            # for noise in ["gaussian", "sp", "low_sp"]:
            for types in ["channel", "full"]:
                if "sp" in noise:
                    types = "full"
                for pad in ["_reflect", ""]:
                    print(f"Reading images in ./result/{f_type}_{noise}_{types}{pad}_{i}.png")
                    globals()[f"{f_type}_{noise}_{types}{pad}_{i}"] = cv2.imread(
                        f'./result/{f_type}_{noise}_{types}{pad}_{i}.png')

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
