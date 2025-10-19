from audioop import add
from tensorly.decomposition import tucker, tensor_train
import tensorly as tl
import os
import numpy as np
import data_utils
import pickle
from tqdm import tqdm
from PIL import Image
import torch
import shutil
import time
from pose_estimation import refine_from_pnp_ransac
from termcolor import colored
import cv2
import scipy
import numba as nb
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer



def img_chunk(img: torch.Tensor, num_split_h, num_split_w):
    # Split img of shape H x W x C to a chunk tensor of shape num_split_h x num_split_w x Hc x Wc x C
    chunk_list = []
    for img_hor_chunk in torch.chunk(img, num_split_h, dim=0):
        chunk_list.append(torch.stack([*torch.chunk(img_hor_chunk, num_split_w, dim=1)], dim=0))
    chunk_tensor = torch.stack(chunk_list, dim=0)  # (num_split_h, num_split_w, Hc, Wc, C)
    return chunk_tensor


def img_dechunk(chunk_tensor: torch.Tensor):
    # Reconstruct an image of shape H x W x C from a chunk tensor of shape num_split_h x num_split_w x Hc x Wc x C
    assert len(chunk_tensor.shape) == 5
    num_split_h, num_split_w, Hc, Wc, C = chunk_tensor.shape
    new_chunk_tensor = chunk_tensor.permute(0, 2, 1, 3, 4)  # num_split_h x Hc x num_split_w x Wc x C
    new_chunk_tensor = new_chunk_tensor.reshape(num_split_h, Hc, -1, C)  # num_split_h x Hc x (num_split_w * Wc) x C
    img = new_chunk_tensor.reshape(-1, num_split_w * Wc, C)  # (num_split_h x Hc) x (num_split_w * Wc) x C

    return img


def max_reflect_filter(event_voxel: np.array, footprint: np.array, mode='nearest'):
    b_f, h_f, w_f = footprint.shape
    h_in, w_in = h_f // 2 + 1, w_f // 2 + 1

    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
    def max_reflect_func(values_ptr, len_values, result, data):
        h_f, w_f = int(len_values ** 0.5), int(len_values ** 0.5)
        h_in, w_in = h_f // 2 + 1, w_f // 2 + 1
        if h_in % 2 == 0 and w_in % 2 == 0:  # Size modulation for larger pixels
            h_in -= 1
            w_in -= 1

        arr_out = carray(values_ptr, (h_f, w_f), dtype=float64)
        arr_in = arr_out[h_f // 2 - h_in // 2: h_f // 2 + h_in // 2 + 1, w_f // 2 - w_in // 2: w_f // 2 + w_in // 2 + 1]
        argmax_in = np.abs(arr_in).argmax()
        argmax_in_i, argmax_in_j = argmax_in // arr_in.shape[1], argmax_in % arr_in.shape[1]
        argmax_out_i, argmax_out_j = argmax_in_i - h_in // 2 + h_f // 2, argmax_in_j - w_in // 2 + w_f // 2
        reflect_out_i, reflect_out_j = (argmax_out_i - h_f // 2) * 2 + h_f // 2, (argmax_out_j - w_f // 2) * 2 + w_f // 2
        min_nn_i = max(reflect_out_i - 1, 0)
        max_nn_i = reflect_out_i + 2
        min_nn_j = max(reflect_out_j - 1, 0)
        max_nn_j = reflect_out_j + 2
        reflect_nn = arr_out[min_nn_i: max_nn_i, min_nn_j: max_nn_j]
        best_idx = np.argmin(np.abs(reflect_nn - arr_in[h_in // 2, w_in // 2]))
        best_i, best_j = best_idx // w_in, best_idx % w_in
        result[0] = reflect_nn[best_i, best_j]

        return 1

    result = scipy.ndimage.generic_filter(event_voxel, function=scipy.LowLevelCallable(max_reflect_func.ctypes), footprint=footprint, mode=mode)
    return result


def max_reflect_fast_filter(event_voxel: np.array, footprint: np.array, blend_mask):
    # Fist compute local maximum locations
    result_voxel = np.copy(event_voxel)
    B, H, W = event_voxel.shape
    min_thres = 1e-6
    abs_event_voxel = np.abs(event_voxel)
    max_event_voxel = cv2.dilate(np.transpose(abs_event_voxel, (1, 2, 0)), np.ones((footprint.shape[1], footprint.shape[2]), dtype=np.uint8))
    max_event_voxel = np.transpose(max_event_voxel, (2, 0, 1))
    max_event_mask = (np.abs(max_event_voxel - abs_event_voxel) < min_thres) & (max_event_voxel > min_thres)  # Locations whose local maximum is itself
    f_h, f_w = footprint.shape[1:]
    max_event_locs = np.stack(np.where(max_event_mask & blend_mask), axis=-1)
    reflect_mtx = np.fliplr(np.eye(f_h))
    for idx in max_event_locs:  # Apply reflection operation
        b_idx, h_idx, w_idx = idx
        valid = (h_idx - f_h // 2 >= 0) & (h_idx + f_h // 2 + 1<= H) & (w_idx - f_w // 2 >= 0) & (w_idx + f_w // 2 + 1 <= W)
        if valid:
            tgt_event_region = event_voxel[b_idx, h_idx - f_h // 2: h_idx + f_h // 2 + 1, w_idx - f_w // 2: w_idx + f_w // 2 + 1]
            result_voxel[b_idx, h_idx - f_h // 2: h_idx + f_h // 2 + 1, w_idx - f_w // 2: w_idx + f_w // 2 + 1] = \
                np.fliplr(np.flipud(tgt_event_region))
    return result_voxel


def median_fast_filter(event_voxel: np.array, blend_locs, footprint: np.array):
    # Fist compute local maximum locations
    result_voxel = np.copy(event_voxel)
    B, H, W = event_voxel.shape
    f_l = footprint.shape[0]
    tgt_voxel = np.copy(event_voxel[:, blend_locs[:, 0], blend_locs[:, 1]])
    tgt_voxel = scipy.ndimage.median_filter(tgt_voxel, footprint=np.ones([f_l, 1], dtype=bool), mode='nearest')
    result_voxel[:, blend_locs[:, 0], blend_locs[:, 1]] = tgt_voxel
    return result_voxel


def apply_voxel_weight(event_voxel: np.array, med_kernel_size=5, reflect_kernel_size=23, mean_filtering=True,
    mean_kernel_size=5, mask_method='med', mask_const=1.0, dilation_size=3):
    """
    Weigh event voxels using ratios between positive and negative events.

    Args:
        event_voxel: (B, H, W) numpy array containing event voxels
        med_kernel_size: Size of median filtering kernel
        reflect_kernel_size: Size of reflection kernel size
        mean_filtering: If True, applies mean filtering prior to applying other filters
        mean_kernel_size: Size of mean kernel size
        mask_methd: Method for selecting which regions to mask out
        mask_const: Constant value applied to the scale statistic
        dilation_size: Size of dilation to apply for sum_mask
    Returns:
        weighted_voxel: (B, H, W) numpy array containing weighted event voxels
    """
    if mean_filtering:
        input_voxel = scipy.ndimage.uniform_filter(event_voxel, size=(1, mean_kernel_size, mean_kernel_size))
    else:
        input_voxel = event_voxel

    sum_voxel = np.abs(event_voxel).sum(0)
    if mask_method == 'med':
        med = np.median(sum_voxel[sum_voxel != 0])
        meddev = np.median(np.abs(sum_voxel[sum_voxel != 0] - med))
        thres_mask = sum_voxel > med + mask_const * meddev
    elif mask_method == 'mean':
        mean = np.mean(sum_voxel[sum_voxel != 0])
        std = np.std(np.abs(sum_voxel[sum_voxel != 0]))
        thres_mask = sum_voxel > mean + mask_const * std

    blend_mask = scipy.ndimage.binary_dilation(thres_mask, np.ones([dilation_size, dilation_size], dtype=bool))
    blend_locs = np.stack(np.where(blend_mask), axis=-1)  # (N_blend, 2)
    blend_mask = np.expand_dims(blend_mask, 0)
    fpt = np.ones([med_kernel_size, 1, 1], dtype=bool)
    if med_kernel_size != 1:
        median_voxel = median_fast_filter(input_voxel, blend_locs=blend_locs, footprint=fpt)
    else:
        median_voxel = input_voxel
    max_fpt = np.ones([1, reflect_kernel_size, reflect_kernel_size], dtype=bool)
    max_val_voxel = max_reflect_fast_filter(input_voxel, footprint=max_fpt, blend_mask=blend_mask)
    filtered_voxel = 0.5 * max_val_voxel + 0.5 * median_voxel

    weighted_voxel = blend_mask * filtered_voxel + np.bitwise_not(blend_mask) * event_voxel

    return weighted_voxel
