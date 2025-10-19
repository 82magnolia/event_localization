from glob import glob
from ast import literal_eval
import numpy as np
import torch
from torch_scatter import scatter_max
from event_representations import parse_cfg_image
from random import shuffle
import os


def get_scene_names(dataset):
    if dataset == 'ev_rooms':
        scene_names = sorted(glob(f"./data/ev_rooms/*/"))
        scene_names = [s.strip('/').split('/')[-1] for s in scene_names]
    return scene_names


def get_img_names(dataset, scene_name):
    if dataset in ['ev_rooms']:
        if '_human' in scene_name:
            acq_condition = scene_name.split('_')[-1]
            orig_scene_name = scene_name.replace(f"_{acq_condition}", "")
            img_names = sorted(glob(f"./data/{dataset}/{scene_name}/images/*")) \
                + sorted(glob(f"./data/ev_rooms/{orig_scene_name}/images/*"))
        elif '_dark' in scene_name:
            acq_condition = scene_name.split('_')[-1]
            orig_scene_name = scene_name.replace(f"_{acq_condition}", "")
            img_names = sorted(glob(f"./data/{dataset}/{scene_name}/images/*")) \
                + sorted(glob(f"./data/ev_rooms/{orig_scene_name}/images/*"))
        else:
            img_names = sorted(glob(f"./data/{dataset}/{scene_name}/images/*"))
    return img_names


def get_scale(dataset, scene_name):
    if dataset in ['ev_rooms']:
        if 'dark' in scene_name or 'human' in scene_name:  # If the maps are created in 'normal' conditions
            acq_condition = scene_name.split('_')[-1]
            orig_scene_name = scene_name.replace(f"_{acq_condition}", "")
            scale = float(open(f"./data/ev_rooms/{orig_scene_name}/scale.txt").readline().strip())
        else:
            scale = float(open(f"./data/{dataset}/{scene_name}/scale.txt").readline().strip())
    return scale


def get_points_3d(dataset, scene_name, scale):
    """
    Parse SfM 3D points into a dictionary containing (xyz coordinates, rgb color, image id, and point 2d idx)
    
    Args:
        dataset: Name of dataset
        scene_name: Name of scene
        scale: Magnification scale of the SfM model compared to meter scale
    
    Returns:
        points_3d: Dictionary with 3D point ids as keys and values containing (xyz coordinates, rgb color, image id, and point 2d idx)
    """
    if dataset in ['ev_rooms']:
        skip_rows = 3
        if 'dark' in scene_name or 'human' in scene_name:  # If the maps are created in 'normal' conditions
            acq_condition = scene_name.split('_')[-1]
            orig_scene_name = scene_name.replace(f"_{acq_condition}", "")
            points_3d_name = f"./data/ev_rooms/{orig_scene_name}/sparse/points3D.txt"
        else:
            points_3d_name = f"./data/{dataset}/{scene_name}/sparse/points3D.txt"
        raw_points_3d = open(points_3d_name, 'r').readlines()[skip_rows:]
        points_3d = {}
        for raw_point in raw_points_3d:
            raw_point_data = [literal_eval(val) for val in raw_point.strip().split(' ')]
            point_id = raw_point_data[0]
            
            data_dict = {}
            data_dict['xyz'] = raw_point_data[1:4]
            data_dict['rgb'] = raw_point_data[4:7]

            data_dict['xyz'] = [val / scale for val in data_dict['xyz']]

            # Save tuples containing (image idx, point 2d idx)
            data_dict['img_2d'] = []
            min_idx = 8 // 2
            max_idx = len(raw_point_data) // 2
            for idx in range(min_idx, max_idx):
                data_dict['img_2d'].append((raw_point_data[2 * idx], raw_point_data[2 * idx + 1]))

            points_3d[point_id] = data_dict

    return points_3d


def get_img_labels(dataset, scene_name, scale):
    """
    Parse SfM image labels into a dictionary containing (translation, rotation, image id, and 2d points)

    Args:
        dataset: Name of dataset
        scene_name: Name of scene
        scale: Magnification scale of the SfM model compared to meter scale

    Returns:
        img_labels: Dictionary with image names as keys and values containing (translation, rotation, image id, and 2d points)
        id2img_name: Dictionary that maps image id to image name
    """
    if dataset in ['ev_rooms']:
        skip_rows = 4
        if 'dark' in scene_name or 'human' in scene_name:  # If the maps are created in 'normal' conditions
            acq_condition = scene_name.split('_')[-1]
            orig_scene_name = scene_name.replace(f"_{acq_condition}", "")
            img_labels_name = f"./data/ev_rooms/{orig_scene_name}/sparse/images.txt"
        else:
            img_labels_name = f"./data/{dataset}/{scene_name}/sparse/images.txt"
        raw_img_labels = open(img_labels_name, 'r').readlines()
        img_labels = {}
        id2img_name = {}
        min_idx = skip_rows // 2
        max_idx = len(raw_img_labels) // 2
        for raw_img_idx in range(min_idx, max_idx):
            raw_img = raw_img_labels[raw_img_idx * 2]
            raw_img_data = [literal_eval(val) for val in raw_img.strip().split(' ')[:-1]]
            raw_img_data.append(raw_img.strip().split(' ')[-1])
            img_id = raw_img_data[0]
            img_name = raw_img_data[-1]
            
            data_dict = {}
            data_dict['trans'] = raw_img_data[5:8]
            data_dict['rot'] = raw_img_data[1:5]
            data_dict['img_id'] = img_id

            data_dict['trans'] = [val / scale for val in data_dict['trans']]

            # Save tuples containing (point 2d coordinates, point 3d idx)
            data_dict['points_2d'] = []
            data_dict['points_3d_idx'] = []
            raw_point = raw_img_labels[raw_img_idx * 2 + 1]
            if raw_point == '\n':
                continue
            else:
                raw_point_data = [literal_eval(val) for val in raw_point.strip().split(' ')]
            local_min_idx = 0
            local_max_idx = len(raw_point_data) // 3
            for idx in range(local_min_idx, local_max_idx):
                data_dict['points_2d'].append([raw_point_data[3 * idx], raw_point_data[3 * idx + 1]])
                data_dict['points_3d_idx'].append(raw_point_data[3 * idx + 2])

            img_labels[img_name] = data_dict
            id2img_name[img_id] = img_name
        
    return img_labels, id2img_name


def get_camera_intrinsics(dataset, scene_name):
    """
    Parse SfM camera labels into an intrinsic matrix

    Args:
        dataset: Name of dataset
        scene_name: Name of scene

    Returns:
        intrinsic_mtx: Numpy array of shape (3, 3) containing camera intrinsics
        distortion_coeff: Distortion coefficient containing radial distortion parameters 
    """
    if dataset in ['ev_rooms']:
        skip_rows = 3
        if 'dark' in scene_name or 'human' in scene_name:  # If the maps are created in 'normal' conditions
            acq_condition = scene_name.split('_')[-1]
            orig_scene_name = scene_name.replace(f"_{acq_condition}", "")
            cam_labels_name = f"./data/ev_rooms/{orig_scene_name}/sparse/cameras.txt"
        else:
            cam_labels_name = f"./data/{dataset}/{scene_name}/sparse/cameras.txt"
        raw_cam_labels = open(cam_labels_name, 'r').readlines()
        values = raw_cam_labels[skip_rows].strip().split(' ')
        intrinsic_mtx = np.zeros([3, 3])
        intrinsic_mtx[0, 0] = values[4]
        intrinsic_mtx[1, 1] = values[4]
        intrinsic_mtx[0, 2] = values[5]
        intrinsic_mtx[1, 2] = values[6]
        intrinsic_mtx[2, 2] = 1
        distortion_coeff = float(values[7])
    
    return intrinsic_mtx, distortion_coeff


def get_split(dataset, scene_name, split_type=None, ref_sample_rate=1, exp_mode='orig'):
    """
    Split .txt files specifying query and reference images and return lists containing each split

    Args:
        dataset: Name of dataset
        scene_name: Name of scene
        split_type: Type of split to test on
        ref_sample_rate: Sample rate of reference images
        exp_mode: Experiment mode to use for evaluation

    Returns:
        query_list: List containing query image names
        ref_list: List containing reference image names 
    """
    if dataset in ['ev_rooms']:
        img_dir = f"./data/{dataset}/{scene_name}/images/*"
        img_list = [os.path.basename(l) for l in glob(img_dir)]
        query_ratio = 0.3
        if 'dark' in scene_name or 'human' in scene_name:  # If the data acquisition is also conducted in 'normal' conditions
            if exp_mode == 'orig':
                shuffle(img_list)
                acq_condition = scene_name.split('_')[-1]
                query_list = img_list[:int(len(img_list) * query_ratio)]
                orig_scene_name = scene_name.replace(f"_{acq_condition}", "")
                ref_dir = f"./data/ev_rooms/{orig_scene_name}/images/*"
                ref_list = img_list[int(len(img_list) * query_ratio):] + [os.path.basename(l) for l in glob(ref_dir)]
            elif exp_mode == 'no_shuffle':
                acq_condition = scene_name.split('_')[-1]
                query_list = img_list
                orig_scene_name = scene_name.replace(f"_{acq_condition}", "")
                ref_dir = f"./data/ev_rooms/{orig_scene_name}/images/*"
                ref_list = [os.path.basename(l) for l in glob(ref_dir)]
        else:
            if exp_mode == 'orig':
                shuffle(img_list)
                query_list = img_list[:int(len(img_list) * query_ratio)]
                ref_list = img_list[int(len(img_list) * query_ratio):]
            elif exp_mode == 'no_shuffle':
                img_list = sorted(img_list)
                query_list = img_list[:int(len(img_list) * query_ratio)]
                ref_list = img_list[int(len(img_list) * query_ratio):]

    if ref_sample_rate != 1:
        ref_list = [r for idx, r in enumerate(ref_list) if idx % ref_sample_rate == 0]

    return query_list, ref_list


def filter_img_list(img_names, img_labels, query_list, ref_list):
    """
    Filter image labels with query list and reference list

    Args:
        img_names: List containing names of images including directory 
        img_labels: Dictionary containing image labels with image names as keys
        query_list: List containing query image names
        ref_list: List containing reference image names
    
    Returns:
        query_img_names: List containing names of query images including directory
        ref_img_names: List containing names of reference images including directory
        query_img_labels: Dictionary containing query image labels
        ref_img_labels: Dictionary containing reference image labels
    """
    query_img_names = sorted(list(filter(lambda name: name.split('/')[-1] in query_list and name.split('/')[-1] in img_labels.keys(), img_names)),
        key=lambda x: x.split('/')[-1])  # Sort with file names excluding directory name
    ref_img_names = sorted(list(filter(lambda name: name.split('/')[-1] in ref_list and name.split('/')[-1] in img_labels.keys(), img_names)),
        key=lambda x: x.split('/')[-1])
    query_img_labels = dict(filter(lambda val: val[0] in query_list and val[0] in img_labels.keys(), img_labels.items()))
    ref_img_labels = dict(filter(lambda val: val[0] in ref_list and val[0] in img_labels.keys(), img_labels.items()))

    return query_img_names, ref_img_names, query_img_labels, ref_img_labels


# COLMAP conversion code excerpted from https://github.com/Fyusion/LLFF/
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def convert_trans(trans, format, device='cpu'):
    if format == 'numpy':
        return np.array(trans).reshape(1, 3)
    elif format == 'torch':
        return torch.tensor(trans).reshape(1, 3).float().to(device)


def convert_rot(rot, format, device='cpu', return_type='rot_mat'):
    qvec = np.array(rot)
    if return_type == 'rot_mat':
        r = qvec2rotmat(qvec)
    elif return_type == 'qvec':
        r = qvec

    if format == 'numpy':
        return r
    elif format == 'torch':
        return torch.tensor(r).float().to(device)


# Event utils
def get_img_timestamp(dataset, scene_name, raw=False):
    """
    Parse image timestamps into a dictionary containing timestamp for each image

    Args:
        dataset: Name of dataset
        scene_name: Name of scene
        raw: If True, loads timestamps for original DAVIS frames in ev_rooms

    Returns:
        img_timestamps: Dictionary containing timestamps for each image
    """
    if dataset in ['ev_rooms']:
        if raw:
            img_timestamp_name = f"./data/{dataset}/{scene_name}/raw_images.txt"
        else:
            img_timestamp_name = f"./data/{dataset}/{scene_name}/images.txt"
        raw_img_timestamp = open(img_timestamp_name, 'r').readlines()
        img_timestamps = {}
        for timestamp_idx in range(len(raw_img_timestamp)):
            raw_timestamp = raw_img_timestamp[timestamp_idx]
            timestamp = literal_eval(raw_timestamp.strip().split(' ')[0])
            img_name = raw_timestamp.strip().split(' ')[-1].split('/')[-1]
            img_timestamps[img_name] = timestamp
    return img_timestamps


def get_events(dataset, scene_name, keep_resolution=True):
    """
    Parse event .txt files into a numpy array containing events

    Args:
        dataset: Name of dataset
        scene_name: Name of scene

    Returns:
        events: (N, 4) numpy array containing (x, y, t, p) events
    """
    if dataset in ['ev_rooms']:
        event_name = f"./data/{dataset}/{scene_name}/events.dat"
        events = np.memmap(event_name, mode='r', dtype=np.float64)
        events = events.reshape(-1, 4)

        if not keep_resolution:
            ORIG_H, ORIG_W = 260, 346
            TGT_H, TGT_W = 180, 240
            events[:, 1] *= TGT_W / ORIG_W
            events[:, 2] *= TGT_H / ORIG_H
    else:
        raise NotImplementedError("Other datasets not supported")
    return events


def slice_events(event, timestamp, method='count', count_window=30000, time_window=0.05):
    # Slice event using fixed counts or time window
    time_idx  = np.searchsorted(event[:, 2], timestamp)
    if method == 'count':
        min_idx = max(0, time_idx - count_window)
        return event[min_idx:time_idx]
    elif method == 'time':
        min_time = timestamp - time_window
        min_idx = np.searchsorted(event[:, 2], min_time)
        return event[min_idx:time_idx]


def make_event_image(event, dataset, rep_list, resolution=None):
    # Convert event to an image-like representation
    if resolution is not None:
        H, W = resolution
        event_tensor = torch.tensor(event)
        event_tensor[:, 0] *= W / ORIG_W
        event_tensor[:, 1] *= H / ORIG_H
    else:
        H, W = ORIG_H, ORIG_W
        event_tensor = torch.tensor(event)

    event_image = parse_cfg_image(event_tensor, H, W, rep_list)
    return event_image


def kpts2points_2d(kpts, points_2d, points_3d_idx):
    # Associate closest 2D points for each keypoint match
    if isinstance(kpts, torch.Tensor):
        dist = (kpts.unsqueeze(1) - points_2d.unsqueeze(0)).norm(dim=-1)  # (N_kpts, N_2D)
    else:
        dist_mtx = (kpts.reshape(kpts.shape[0], 1, *kpts.shape[1:]) - points_2d.reshape(1, points_2d.shape[0], *points_2d.shape[1:]))  # (N_kpts, N_2D, 2)
        dist = np.linalg.norm(dist_mtx, 2, axis=-1)
    kpts_points_2d_idx = dist.argmin(-1)  # (N_kpts, )

    kpts_points_2d = points_2d[kpts_points_2d_idx]  # (N_valid, )
    kpts_points_3d_idx = points_3d_idx[kpts_points_2d_idx]

    # Exclude indices below 0
    if isinstance(kpts, torch.Tensor):
        valid_match_idx = (kpts_points_3d_idx > 0) & (dist.min(-1).values < 5)
    else:
        valid_match_idx = (kpts_points_3d_idx > 0) & (dist.min(-1) < 5)
    kpts_points_2d = kpts_points_2d[valid_match_idx]
    kpts_points_3d_idx = kpts_points_3d_idx[valid_match_idx]

    return kpts_points_2d, kpts_points_3d_idx, valid_match_idx
