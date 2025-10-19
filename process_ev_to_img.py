import aedat
import numpy as np
import cv2
import sys
import os
import shutil
from collections import namedtuple
from parse_utils import parse_ini, save_ini, parse_value
from e2vid import E2VID
from e2vid.utils.voxelgrid import VoxelGrid
from privacy_utils import apply_voxel_weight
import argparse
from data_utils import get_img_timestamp
from ast import literal_eval
import torch

# General config parsing
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Config file to use for running experiments", default=None, type=str)
parser.add_argument("--event_path", help="Path to events.dat", default=None, type=str)
parser.add_argument("--data_root", help="Root directory for saving data", default="./log", type=str)
parser.add_argument('--override', default=None, help='Arguments for overriding config')
parser.add_argument('--gray_ev', action='store_true', help='Save event accumulation as grayscale')
parser.add_argument('--joint_ev', action='store_true', help='Save event accumulations alongsize reconstructions')
parser.add_argument('--joint_img', action='store_true', help='Save event accumulations alongsize images')
parser.add_argument('--joint_filter', action='store_true', help='Jointly save filtered and non-filtered reconstructions')
args = parser.parse_args()

event_path = args.event_path
data_root = args.data_root
cfg_dir = args.config

if not os.path.exists(data_root):
    print(f"Making Directory {data_root}...")
    os.makedirs(data_root)
else:
    print(f"Removing Directory {data_root}...")
    shutil.rmtree(data_root)
    print(f"Making Directory {data_root}...")
    os.makedirs(data_root)

cfg = parse_ini(cfg_dir)

if args.override is not None:
    equality_split = args.override.split('=')
    num_equality = len(equality_split)
    assert num_equality > 0
    if num_equality == 2:
        override_dict = {equality_split[0]: parse_value(equality_split[1])}
    else:
        keys = [equality_split[0]]  # First key
        keys += [equality.split(',')[-1] for equality in equality_split[1:-1]]  # Other keys
        values = [equality.replace(',' + key, '') for equality, key in zip(equality_split[1:-1], keys[1:])]  # Get values other than last field
        values.append(equality_split[-1])  # Get last value
        values = [value.replace('[', '').replace(']', '') for value in values]

        override_dict = {key: parse_value(value) for key, value in zip(keys, values)}

    cfg_dict = cfg._asdict()

    Config = namedtuple('Config', tuple(set(cfg._fields + tuple(override_dict.keys()))))
    
    cfg_dict.update(override_dict)

    cfg = Config(**cfg_dict)

img_root = os.path.join(data_root, 'images')
os.makedirs(img_root)

num_unroll = getattr(cfg, 'num_unroll', 10)
voxel_weight = getattr(cfg, 'voxel_weight', False)
if args.joint_filter:  # Enforce voxel_weight to be true if joint_filter is on
    voxel_weight = True
img_file = open(os.path.join(data_root, 'images.txt'), 'w')
separate_blocks = getattr(cfg, 'separate_blocks', True)

events = np.memmap(event_path, mode='r', dtype=np.float64)
events = events.reshape(-1, 4)

count_window = cfg.count_window
slice_size = count_window * num_unroll
if separate_blocks:
    num_blocks = events.shape[0] // (count_window * num_unroll)
else:
    shift_size = getattr(cfg, 'shift_size', count_window * 2)
    num_blocks = events.shape[0] // shift_size
print(f"Reconstructing images for {num_blocks} blocks...")

# Setup for event to image reconstruction module
reconstructor = E2VID(cfg)  # Used for reconstructing references

# Voxel weights for privacy preservation
voxel_weight = getattr(cfg, 'voxel_weight', False)
if voxel_weight:
    print("Using voxel weighting for privacy preservation...")

grid = VoxelGrid(reconstructor.model.num_bins, cfg.width, cfg.height, upsample_rate=cfg.upsample_rate)

if args.joint_img:
    img_timestamp_name = args.event_path.replace("events.dat", "raw_images.txt")
    raw_img_root = args.event_path.replace("events.dat", "raw_images")
    raw_img_timestamp = open(img_timestamp_name, 'r').readlines()
    raw_img_name_list = []
    raw_timestamp_list = []
    for timestamp_idx in range(len(raw_img_timestamp)):
        raw_timestamp = raw_img_timestamp[timestamp_idx]
        timestamp = literal_eval(raw_timestamp.strip().split(' ')[0])
        img_name = raw_timestamp.strip().split(' ')[-1].split('/')[-1]
        raw_img_name_list.append(os.path.join(raw_img_root, img_name))
        raw_timestamp_list.append(timestamp)
    raw_timestamp_arr = np.array(raw_timestamp_list)

for block_idx in range(num_blocks):
    if separate_blocks:
        query_events = events[block_idx * slice_size: (block_idx + 1) * slice_size]
    else:
        query_events = events[block_idx * shift_size: block_idx * shift_size + slice_size]
    max_roll_idx = query_events.shape[0] // count_window + 1 if query_events.shape[0] % count_window != 0 else query_events.shape[0] // count_window

    reconstructor.image_reconstructor.last_states_for_each_channel = {'grayscale': None}  # re-initialize state

    if voxel_weight:
        grid_list = []
        for roll_idx in range(max_roll_idx):
            query_events_grid, _ = grid.events_to_voxel_grid(query_events[count_window * roll_idx: count_window * (roll_idx + 1), [2, 0, 1, 3]])
            query_events_grid = grid.normalize_voxel(query_events_grid)
            grid_list.append(query_events_grid)

        total_grid = np.concatenate(grid_list, axis=0)
        time_gap = query_events[-1, 2] - query_events[0, 2]
        weighted_total_grid = apply_voxel_weight(total_grid, getattr(cfg, 'med_kernel_size', 5), getattr(cfg, 'reflect_kernel_size', 23),
            getattr(cfg, 'mean_filtering', True), getattr(cfg, 'mean_kernel_size', 5), 
            getattr(cfg, 'mask_method', 'med'), getattr(cfg, 'mask_const', 1.))
        block_size = weighted_total_grid.shape[0] // max_roll_idx

        for roll_idx in range(max_roll_idx):
            query_recon_img = reconstructor(weighted_total_grid[roll_idx * block_size: (roll_idx + 1) * block_size])

    else:
        for roll_idx in range(max_roll_idx):
            query_events_grid, _ = grid.events_to_voxel_grid(query_events[count_window * roll_idx: count_window * (roll_idx + 1), [2, 0, 1, 3]])
            query_events_grid = grid.normalize_voxel(query_events_grid)
            query_recon_img = reconstructor(query_events_grid)

    if args.joint_ev:
        query_recon_img = np.stack([query_recon_img] * 3, axis=-1)
        if args.gray_ev:
            query_ev_img = np.ones([*query_recon_img.shape], dtype=np.uint8) * 122
            query_ev_img[query_events_grid.sum(0) > 0] = np.array([[255, 255, 255]])
            query_ev_img[query_events_grid.sum(0) < 0] = np.array([[0, 0, 0]])
        else:
            query_ev_img = np.zeros([*query_recon_img.shape], dtype=np.uint8)
            query_ev_img[query_events_grid.sum(0) > 0] = np.array([[0, 0, 255]])
            query_ev_img[query_events_grid.sum(0) < 0] = np.array([[255, 0, 0]])
        margin = 255 * np.ones([query_recon_img.shape[0], query_recon_img.shape[1] // 8, 3], dtype=np.uint8)
        query_joint_img = np.concatenate([query_ev_img, margin, query_recon_img], axis=1)
        cv2.imwrite(os.path.join(img_root, f"recon_{str(block_idx).rjust(8, '0')}.png"), query_joint_img)
    elif args.joint_filter:
        query_recon_img = np.stack([query_recon_img] * 3, axis=-1)

        for roll_idx in range(max_roll_idx):
            query_events_grid, _ = grid.events_to_voxel_grid(query_events[count_window * roll_idx: count_window * (roll_idx + 1), [2, 0, 1, 3]])
            query_events_grid = grid.normalize_voxel(query_events_grid)
            query_orig_img = reconstructor(query_events_grid)

        query_orig_img = np.stack([query_orig_img] * 3, axis=-1)
        margin = 255 * np.ones([query_recon_img.shape[0], query_recon_img.shape[1] // 8, 3], dtype=np.uint8)
        query_joint_img = np.concatenate([query_orig_img, margin, query_recon_img], axis=1)
        cv2.imwrite(os.path.join(img_root, f"recon_{str(block_idx).rjust(8, '0')}.png"), query_joint_img)
    elif args.joint_img:
        best_idx = np.abs(raw_timestamp_arr - query_events[-1, 2]).argmin()
        query_raw_img = cv2.imread(raw_img_name_list[best_idx])
        query_ev_img = np.zeros([*query_raw_img.shape], dtype=np.uint8)
        query_ev_img[query_events_grid.sum(0) > 0] = np.array([[0, 0, 255]])
        query_ev_img[query_events_grid.sum(0) < 0] = np.array([[255, 0, 0]])
        margin = 255 * np.ones([query_raw_img.shape[0], query_raw_img.shape[1] // 8, 3], dtype=np.uint8)
        query_joint_img = np.concatenate([query_raw_img, margin, query_ev_img], axis=1)
        cv2.imwrite(os.path.join(img_root, f"recon_{str(block_idx).rjust(8, '0')}.png"), query_joint_img)
    else:
        query_recon_img = np.stack([query_recon_img] * 3, axis=-1)
        cv2.imwrite(os.path.join(img_root, f"recon_{str(block_idx).rjust(8, '0')}.png"), query_recon_img)
    timestamp = query_events[-1, 2]
    img_file.write(f"{timestamp} images/recon_{str(block_idx).rjust(8, '0')}.png\n")
    print(f"IMAGE {block_idx} saved...")

img_file.close()
