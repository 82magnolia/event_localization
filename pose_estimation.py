import torch
import cv2
try:
    import kornia as K
    import kornia.feature as KF
except ImportError:
    pass
from superglue_models.matching import Matching
from superglue_models.superpoint import SuperPoint
import numpy as np
import data_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import ssl
# Disabled to allow Pytorch hub download (currently not working as torch version is old)
ssl._create_default_https_context = ssl._create_unverified_context



class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()
        self.normalize = nn.functional.normalize
        
    def forward(self, x):
        x = self.normalize(x)
        return x


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, 
    visualize_match_dict, log_dir, match_vis_idx, points_2d=None, show_keypoints=False, margin=10):
    
    draw_matches = visualize_match_dict['draw_matches']
    draw_points_2d = visualize_match_dict['draw_points_2d']
    draw_match_kpts=visualize_match_dict['draw_match_kpts']
    draw_img_kpts = visualize_match_dict['draw_img_kpts']
    if draw_img_kpts:
        img_kpts = visualize_match_dict['img_kpts']
        raise NotImplementedError("TODO: Image keypoint visualization")

    image0 = image0.cpu().numpy()
    image1 = image1.cpu().numpy()
    kpts0 = kpts0.cpu().numpy()
    kpts1 = kpts1.cpu().numpy()
    mkpts0 = mkpts0.cpu().numpy()
    mkpts1 = mkpts1.cpu().numpy()
    
    if points_2d is not None:
        points_2d = points_2d.cpu().numpy()
    
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    image0 = (image0 * 255).astype(np.uint8)
    image1 = (image1 * 255).astype(np.uint8)
    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for idx, ((x0, y0), (x1, y1), c) in enumerate(zip(mkpts0, mkpts1, color)):
        c = c.tolist()
        if draw_matches:
            cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                    color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        if draw_img_kpts:
            x_2d, y_2d = img_kpts[idx]
            x_2d, y_2d = int(x_2d), int(y_2d)
            c_2d = [255, 0, 0]
            cv2.circle(out, (x_2d + margin + W0, y_2d), 2, c_2d, -1,
                    lineType=cv2.LINE_AA)

        if draw_match_kpts:
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                    lineType=cv2.LINE_AA)
        if draw_points_2d:
            x_2d, y_2d = points_2d[idx]
            x_2d, y_2d = int(x_2d), int(y_2d)
            c_2d = [0, 250, 0]
            cv2.circle(out, (x_2d + margin + W0, y_2d), 2, c_2d, -1,
                    lineType=cv2.LINE_AA)
    
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.savefig(os.path.join(log_dir, f"match_{match_vis_idx}.png"))


def ev_to_grayscale(
    image: torch.Tensor, rgb_weights: torch.Tensor = [1 / 3, 1 / 3, 1 / 3]) -> torch.Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/ev_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = ev_to_grayscale(input) # 2x1x4x5
    """
    r: torch.Tensor = image[..., 0:1, :, :]
    g: torch.Tensor = image[..., 1:2, :, :]
    b: torch.Tensor = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights
    return w_r * r + w_g * g + w_b * b


def refine_from_pnp_ransac(query_ev_img, cam_intrinsic, distortion_coeff, cand_names, cand_ev_imgs, \
    points_3d, ref_img_labels, match_model=None, visualize_match=False, visualize_match_dict=None, \
    no_ransac=False, rgb_weights=[1 / 3, 1 / 3, 1 / 3], log_dir=None, match_vis_idx=0):
    """
    Refine top-k poses by using feature matching, here both rotation and translation is optimized

    Args:
        query_ev_img: (C, H, W) torch tensor containing query event image
        cam_intrinsic: (3, 3) numpy array containing camera intrinsic matrix
        distortion_coeff: Float containing radial distortion
        cand_names: Name of reference images that are selected as candidate views
        cand_ev_imgs: (N_c, C, H, W) torch tensor containing candidate event images
        points_3d: Dictionary of 3D points
        ref_img_labels: Dictionary of reference image labels 
        match_model: Model used for matching features
        visualize_match: If True, visualize matching results
        visualize_match_dict: Dictionary containing configs for visualizing matches
        no_ransac: If True, skips the RANSAC process
        rgb_weights: Weights for blending the three channels to one
        log_dir: Directory for saving matches
        match_vis_idx: Index for saving matches
    
    Returns:
        best_idx: Index among the k reference images with the most feature matches
        best_trans: (1, 3) torch tensor containing translation of best view
        best_rot: (3, 3) torch tensor containing rotation of best view
        top_k_trans: (K, 3) torch tensor containing retrieved candidate translations
        top_k_rot: (K, 3, 3) torch tensor containing retrieved candidate rotations
        rf_trans: (1, 3) torch tensor containing refined translation
        rf_rot: (3, 3) torch tensor containing refined rotation
    """
    num_k = len(cand_names)
    device = query_ev_img.device
    tgt_img = query_ev_img.repeat(cand_ev_imgs.shape[0], 1, 1, 1)

    # Extract candidate poses
    top_k_trans = [data_utils.convert_trans(ref_img_labels[name]['trans'], 'torch', device) for name in cand_names]
    top_k_rot = [data_utils.convert_rot(ref_img_labels[name]['rot'], 'torch', device) for name in cand_names]

    top_k_trans = torch.cat(top_k_trans, dim=0)  # (K, 3)
    top_k_rot = torch.stack(top_k_rot, dim=0)  # (K, 3, 3)
    top_k_trans = -(top_k_trans.unsqueeze(1) @ top_k_rot).reshape(num_k, 3)

    if match_model is None:
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        matching = Matching(config).eval().to(device)
    else:
        matching = match_model
    
    tgt_img = ev_to_grayscale(tgt_img, rgb_weights)
    cand_ev_imgs = ev_to_grayscale(cand_ev_imgs, rgb_weights)
    keys = ['keypoints', 'scores', 'descriptors']
    last_data = matching.superpoint({'image': tgt_img[0:1]})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = tgt_img[0:1]

    matches = {'batch_indexes': [], 'keypoints0': [], 'keypoints1': [], 'confidence': []}
    for idx in range(num_k):
        with torch.no_grad():
            pred = matching({**last_data, 'image1': cand_ev_imgs[idx: idx + 1]})
        kpts0 = last_data['keypoints0'][0]
        kpts1 = pred['keypoints1'][0]
        pred_matches = pred['matches0'][0]
        confidence = pred['matching_scores0'][0]

        valid = pred_matches > -1
        num_keypts = valid.sum()
        mkpts0 = kpts0[valid]
        if valid.sum() != 0:
            mkpts1 = kpts1[pred_matches[valid]]
        else:
            mkpts1 = mkpts0

        confidence = confidence[valid]

        if visualize_match:
            if visualize_match_dict['visualize_all']:
                conf_thres = visualize_match_dict['conf_thres']
                vis_idx = confidence > conf_thres
                vis_conf = confidence[vis_idx]
                vis_mkpts0 = mkpts0[vis_idx]
                vis_mkpts1 = mkpts1[vis_idx]
                color = cm.jet(vis_conf.cpu().numpy())
                make_matching_plot_fast(tgt_img[0, 0], cand_ev_imgs[idx, 0], kpts0, kpts1, vis_mkpts0, vis_mkpts1, \
                    color, visualize_match_dict=visualize_match_dict, log_dir=log_dir, match_vis_idx=match_vis_idx)

        matches['batch_indexes'].append(idx * torch.ones(num_keypts, dtype=torch.int))
        matches['keypoints0'].append(mkpts0)
        matches['keypoints1'].append(mkpts1)
        matches['confidence'].append(confidence)
    matches['batch_indexes'] = torch.cat(matches['batch_indexes'], dim=0)
    matches['keypoints0'] = torch.cat(matches['keypoints0'], dim=0)
    matches['keypoints1'] = torch.cat(matches['keypoints1'], dim=0)
    matches['confidence'] = torch.cat(matches['confidence'], dim=0)
    batch_idx = matches['batch_indexes']

    # Choose view with largest number of matches
    batch_confidence = torch.tensor([matches['confidence'][batch_idx == idx].sum() for idx in range(num_k)])
    best_idx = batch_confidence.argmax()
    best_name = cand_names[best_idx]

    best_trans = top_k_trans[best_idx: best_idx + 1]
    best_rot = top_k_rot[best_idx]

    # Note that matches['keypoints'] are in (x, y) order, similar to torch.nn.grid_sample
    best_tgt_kpts = matches['keypoints0'][batch_idx == best_idx]  # (N_f, 2)
    best_cand_kpts = matches['keypoints1'][batch_idx == best_idx]  # (N_f, 2)
    best_conf = matches['confidence'][batch_idx == best_idx]
    
    best_points_2d = torch.tensor(ref_img_labels[best_name]['points_2d'], dtype=torch.float, device=device) # (N_2D, 2)
    best_points_3d_idx = torch.tensor(ref_img_labels[best_name]['points_3d_idx'], dtype=torch.long, device=device)
    best_match_2d, best_match_3d_idx, valid_match_idx = data_utils.kpts2points_2d(best_cand_kpts, best_points_2d, best_points_3d_idx)
    best_match_3d = [points_3d[idx]['xyz'] for idx in best_match_3d_idx.tolist()]
    best_match_3d = torch.tensor(best_match_3d, dtype=torch.float, device=device)  # (N_3D, 3)

    # Skip optimization for empty best_coord_arr
    if best_match_3d.numel() == 0 or best_conf.shape[0] < 30 or no_ransac:  # These are cases where PnP cannot be applied
        optimize = False
    else:
        optimize = True

    # Visualize for best match
    if visualize_match:
        if not visualize_match_dict['visualize_all']:
            if visualize_match_dict['draw_points_2d']:
                conf_thres = visualize_match_dict.get('conf_thres', 0.)
                vis_idx = matches['confidence'][batch_idx == best_idx][valid_match_idx] > conf_thres
                vis_conf = matches['confidence'][batch_idx == best_idx][valid_match_idx][vis_idx]
                vis_mkpts0 = best_tgt_kpts[valid_match_idx][vis_idx]
                vis_mkpts1 = best_cand_kpts[valid_match_idx][vis_idx]
                vis_points_2d = best_match_2d[vis_idx]
                color = cm.jet(vis_conf.cpu().numpy())
                make_matching_plot_fast(tgt_img[0, 0], cand_ev_imgs[idx, 0], vis_mkpts0, vis_mkpts1, vis_mkpts0, vis_mkpts1, \
                    color, points_2d=vis_points_2d, visualize_match_dict=visualize_match_dict, log_dir=log_dir, match_vis_idx=match_vis_idx)

            else:
                conf_thres = visualize_match_dict.get('conf_thres', 0.)
                vis_idx = matches['confidence'][batch_idx == best_idx] > conf_thres
                vis_conf = matches['confidence'][batch_idx == best_idx][vis_idx]
                vis_mkpts0 = matches['keypoints0'][batch_idx == best_idx][vis_idx]
                vis_mkpts1 = matches['keypoints1'][batch_idx == best_idx][vis_idx]
                color = cm.jet(vis_conf.cpu().numpy())
                make_matching_plot_fast(tgt_img[0, 0], cand_ev_imgs[idx, 0], vis_mkpts0, vis_mkpts1, vis_mkpts0, vis_mkpts1, \
                    color, visualize_match_dict=visualize_match_dict, log_dir=log_dir, match_vis_idx=match_vis_idx)

    # Optimize
    prior_trans = best_trans.clone().detach()
    prior_rot = best_rot.clone().detach()

    if optimize:
        try:
            # RANSAC Loop
            x = best_match_3d.cpu().numpy()  # (N_3D, 3)
            y = best_tgt_kpts[valid_match_idx].cpu().numpy()  # (N_2D, 2)

            sol = cv2.solvePnPRansac(x, y, cam_intrinsic, distCoeffs=np.array([distortion_coeff, 0, 0, 0]), iterationsCount=100, reprojectionError=8., flags=cv2.SOLVEPNP_SQPNP)
            rf_rot = torch.from_numpy(cv2.Rodrigues(sol[1])[0]).float().to(device)  # (3, 3)
            rf_trans = (- rf_rot.T @ torch.from_numpy(sol[2]).float().to(device)).T  # (1, 3)
        except cv2.error as e:
            rf_trans = prior_trans
            rf_rot = prior_rot
    else:
        rf_trans = prior_trans
        rf_rot = prior_rot

    return best_idx, best_trans, best_rot, top_k_trans, top_k_rot, rf_trans, rf_rot


def refine_from_fast_pnp_ransac(query_ev_img, cam_intrinsic, distortion_coeff, cand_names, cand_ev_imgs, \
    points_3d, ref_img_labels, match_model=None, visualize_match=False, visualize_match_dict=None, \
    no_ransac=False, rgb_weights=[1 / 3, 1 / 3, 1 / 3], log_dir=None, match_vis_idx=0, device='cpu'):
    """
    Refine top-k poses by using feature matching, here both rotation and translation is optimized

    Args:
        query_ev_img: (C, H, W) numpy array containing query event image
        cam_intrinsic: (3, 3) numpy array containing camera intrinsic matrix
        distortion_coeff: Float containing radial distortion
        cand_names: Name of reference images that are selected as candidate views
        cand_ev_imgs: (N_c, C, H, W) numpy array containing candidate event images
        points_3d: Dictionary of 3D points
        ref_img_labels: Dictionary of reference image labels 
        match_model: Model used for matching features
        visualize_match: If True, visualize matching results
        visualize_match_dict: Dictionary containing configs for visualizing matches
        no_ransac: If True, skips the RANSAC process
        rgb_weights: Weights for blending the three channels to one
        log_dir: Directory for saving matches
        match_vis_idx: Index for saving matches
        device: Device for mapping tensors
    
    Returns:
        best_idx: Index among the k reference images with the most feature matches
        best_trans: (1, 3) torch tensor containing translation of best view
        best_rot: (3, 3) torch tensor containing rotation of best view
        top_k_trans: (K, 3) torch tensor containing retrieved candidate translations
        top_k_rot: (K, 3, 3) torch tensor containing retrieved candidate rotations
        rf_trans: (1, 3) torch tensor containing refined translation
        rf_rot: (3, 3) torch tensor containing refined rotation
    """
    num_k = len(cand_names)
    tgt_img = query_ev_img

    # Extract candidate poses
    top_k_trans = [data_utils.convert_trans(ref_img_labels[name]['trans'], 'torch', device) for name in cand_names]
    top_k_rot = [data_utils.convert_rot(ref_img_labels[name]['rot'], 'torch', device) for name in cand_names]

    top_k_trans = torch.cat(top_k_trans, dim=0)  # (K, 3)
    top_k_rot = torch.stack(top_k_rot, dim=0)  # (K, 3, 3)
    top_k_trans = -(top_k_trans.unsqueeze(1) @ top_k_rot).reshape(num_k, 3)
    
    tgt_img = tgt_img.mean(0)
    cand_ev_imgs = cand_ev_imgs.mean(1)
    keys = ['keypoints', 'scores', 'descriptors']

    matches = {'batch_indexes': [], 'keypoints0': [], 'keypoints1': [], 'confidence': []}
    kpts0, desc0 = match_model['detector'].detectAndCompute((tgt_img * 255).astype(np.uint8), None)

    for idx in range(num_k):
        kpts1, desc1 = match_model['detector'].detectAndCompute((cand_ev_imgs[idx] * 255).astype(np.uint8), None)

        pred_matches = match_model['matcher'].knnMatch(desc0.astype(np.float32), desc1.astype(np.float32), 2)
        
        matched0 = []
        matched1 = []
        nn_match_ratio = 0.7 # Nearest neighbor matching ratio
        for m_match, n_match in pred_matches:
            if m_match.distance < nn_match_ratio * n_match.distance:
                matched0.append(np.array(kpts0[m_match.queryIdx].pt))
                matched1.append(np.array(kpts1[m_match.trainIdx].pt))

        confidence = np.ones([len(matched0),])
        if visualize_match:
            if visualize_match_dict['visualize_all']:
                vis_mkpts0 = matched0
                vis_mkpts1 = matched1
                vis_conf = confidence
                color = cm.jet(vis_conf.cpu().numpy())
                make_matching_plot_fast(tgt_img[0, 0], cand_ev_imgs[idx, 0], matched0, matched1, vis_mkpts0, vis_mkpts1, \
                    color, visualize_match_dict=visualize_match_dict, log_dir=log_dir, match_vis_idx=match_vis_idx)

        matches['batch_indexes'].append(idx * np.ones([len(matched0),], dtype=np.int))
        matches['keypoints0'].append(matched0)
        matches['keypoints1'].append(matched1)
        matches['confidence'].append(confidence)
    matches['batch_indexes'] = np.concatenate(matches['batch_indexes'], axis=0)
    matches['keypoints0'] = np.concatenate(matches['keypoints0'], axis=0)
    matches['keypoints1'] = np.concatenate(matches['keypoints1'], axis=0)
    matches['confidence'] = np.concatenate(matches['confidence'], axis=0)
    batch_idx = matches['batch_indexes']

    # Choose view with largest number of matches
    batch_confidence = np.array([matches['confidence'][batch_idx == idx].sum() for idx in range(num_k)])
    best_idx = batch_confidence.argmax()
    best_name = cand_names[best_idx]

    best_trans = top_k_trans[best_idx: best_idx + 1]
    best_rot = top_k_rot[best_idx]

    # Note that matches['keypoints'] are in (x, y) order, similar to torch.nn.grid_sample
    best_tgt_kpts = matches['keypoints0'][batch_idx == best_idx]  # (N_f, 2)
    best_cand_kpts = matches['keypoints1'][batch_idx == best_idx]  # (N_f, 2)
    best_conf = matches['confidence'][batch_idx == best_idx]
    
    best_points_2d = np.array(ref_img_labels[best_name]['points_2d']) # (N_2D, 2)
    best_points_3d_idx = np.array(ref_img_labels[best_name]['points_3d_idx'])

    if len(best_cand_kpts) == 0:
        optimize = False
    else:
        best_match_2d, best_match_3d_idx, valid_match_idx = data_utils.kpts2points_2d(best_cand_kpts, best_points_2d, best_points_3d_idx)
        best_match_3d = np.array([points_3d[idx]['xyz'] for idx in best_match_3d_idx.tolist()])  # (N_3D, 3)

        # Skip optimization for empty best_coord_arr
        if len(best_match_3d) == 0 or best_conf.shape[0] < 5 or no_ransac:  # These are cases where PnP cannot be applied
            optimize = False
        else:
            optimize = True

    # Visualize for best match
    if visualize_match:
        if not visualize_match_dict['visualize_all']:
            if visualize_match_dict['draw_points_2d']:
                vis_conf = matches['confidence'][batch_idx == best_idx][valid_match_idx]
                vis_mkpts0 = best_tgt_kpts[valid_match_idx]
                vis_mkpts1 = best_cand_kpts[valid_match_idx]
                vis_points_2d = best_match_2d
                color = cm.jet(vis_conf.cpu().numpy())
                make_matching_plot_fast(tgt_img[0, 0], cand_ev_imgs[idx, 0], vis_mkpts0, vis_mkpts1, vis_mkpts0, vis_mkpts1, \
                    color, points_2d=vis_points_2d, visualize_match_dict=visualize_match_dict, log_dir=log_dir, match_vis_idx=match_vis_idx)

            else:
                conf_thres = visualize_match_dict.get('conf_thres', 0.)
                vis_conf = matches['confidence'][batch_idx == best_idx]
                vis_mkpts0 = matches['keypoints0'][batch_idx == best_idx]
                vis_mkpts1 = matches['keypoints1'][batch_idx == best_idx]
                color = cm.jet(vis_conf.cpu().numpy())
                make_matching_plot_fast(tgt_img[0, 0], cand_ev_imgs[idx, 0], vis_mkpts0, vis_mkpts1, vis_mkpts0, vis_mkpts1, \
                    color, visualize_match_dict=visualize_match_dict, log_dir=log_dir, match_vis_idx=match_vis_idx)

    # Optimize
    prior_trans = best_trans.clone().detach()
    prior_rot = best_rot.clone().detach()

    if optimize:
        try:
            # RANSAC Loop
            x = best_match_3d  # (N_3D, 3)
            y = best_tgt_kpts[valid_match_idx]  # (N_2D, 2)

            sol = cv2.solvePnPRansac(x, y, cam_intrinsic, distCoeffs=np.array([distortion_coeff, 0, 0, 0]), iterationsCount=100, reprojectionError=8., flags=cv2.SOLVEPNP_SQPNP)
            rf_rot = torch.from_numpy(cv2.Rodrigues(sol[1])[0]).float().to(device)  # (3, 3)
            rf_trans = (- rf_rot.T @ torch.from_numpy(sol[2]).float().to(device)).T  # (1, 3)
        except cv2.error as e:
            rf_trans = prior_trans
            rf_rot = prior_rot

    else:
        rf_trans = prior_trans
        rf_rot = prior_rot

    return best_idx, best_trans, best_rot, top_k_trans, top_k_rot, rf_trans, rf_rot


def get_matcher(cfg, device='cpu'):
    # Return appropriate matching module
    match_model_type = getattr(cfg, 'match_model_type', 'SuperGlue')
    if match_model_type == 'SuperGlue':
        superglue_cfg = {
                'superpoint': {
                    'nms_radius': getattr(cfg, 'nms_radius', 4),
                    'keypoint_threshold': getattr(cfg, 'keypoint_threshold', 0.005),
                    'max_keypoints': getattr(cfg, 'max_keypoints', -1)
                },
                'superglue': {
                    'weights': getattr(cfg, 'weights', 'indoor'),
                    'sinkhorn_iterations': getattr(cfg, 'sinkhorn_iterations', 20),
                    'match_threshold': getattr(cfg, 'match_threshold', 0.2),
                }
            }
        match_model = Matching(superglue_cfg)
        if getattr(cfg, 'load_superpoint', None) is not None:
            match_model.superpoint.load_state_dict(torch.load(cfg.load_superpoint))
        match_model = match_model.eval().to(device)
    elif match_model_type == 'LoFTR':
        match_model = KF.LoFTR(pretrained=getattr(cfg, 'weights', 'indoor')).to(device)
    elif match_model_type == 'SIFT':
        match_model = {}
        match_model['detector'] = cv2.SIFT_create()
        match_model['matcher'] = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    elif match_model_type == 'AKAZE':
        match_model = {}
        match_model['detector'] = cv2.AKAZE_create()
        match_model['matcher'] = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    elif match_model_type == 'BRISK':
        match_model = {}
        match_model['detector'] = cv2.BRISK_create()
        match_model['matcher'] = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    elif match_model_type == 'ORB':
        match_model = {}
        match_model['detector'] = cv2.ORB_create()
        match_model['matcher'] = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    else:
        raise ValueError("Invalid model type")
    
    return match_model


def get_superpoint(cfg, device='cpu'):
    superpoint_cfg = {
                'superpoint': {
                    'nms_radius': getattr(cfg, 'nms_radius', 4),
                    'keypoint_threshold': getattr(cfg, 'keypoint_threshold', 0.005),
                    'max_keypoints': getattr(cfg, 'max_keypoints', -1)
                }
        }
    superpoint_model = SuperPoint(superpoint_cfg.get('superpoint', {}))

    if getattr(cfg, 'load_superpoint', None) is not None:
        superpoint_model.load_state_dict(torch.load(cfg.load_superpoint))
    
    superpoint_model = superpoint_model.eval().to(device)
    return superpoint_model


def get_netvlad(cfg, device='cpu'):
    # Return pre-trained NetVlad layer
    netvlad_model = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True)

    if getattr(cfg, 'load_netvlad', None) is not None:
        netvlad_model.load_state_dict(torch.load(cfg.load_netvlad))

        # Additionally add predictor if it exists on model directory
        if os.path.exists(cfg.load_netvlad.replace('model.pth', 'predictor.pth')):
            print("Additionally loading predictor!")
            predictor_dir = cfg.load_netvlad.replace('model.pth', 'predictor.pth')
            predictor = nn.Sequential(nn.Linear(4096, 1024, bias=False),
                                            nn.BatchNorm1d(1024),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(1024, 4096)) # output layer
            predictor.load_state_dict(torch.load(predictor_dir))
            predictor.eval().to(device)
            netvlad_model.eval().to(device)
            normalizer = Normalizer().to(device)

            model = nn.Sequential(
                netvlad_model,
                predictor,
                normalizer
            )
        else:
            model = netvlad_model.eval().to(device)

    else:
        model = netvlad_model.eval().to(device)

    return model

class BrewNetVLAD(nn.Module):  # Brewed implementation of NetVLAD
    def __init__(self, base_model, net_vlad, pca_layer, dim=4096):
        super(BrewNetVLAD, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.pca_layer = pca_layer

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        _, x = self.base_model(x)
        vlad_x = self.net_vlad(x)
        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        # reduction
        N, D = vlad_x.size()
        vlad_x = vlad_x.view(N, D, 1, 1)
        vlad_x = self.pca_layer(vlad_x).view(N, -1)
        vlad_x = F.normalize(vlad_x, p=2, dim=-1)  # L2 normalize
        return vlad_x

class MobileNetVLAD(nn.Module):  # Brewed implementation of NetVLAD
    def __init__(self, device):
        super(MobileNetVLAD, self).__init__()
        mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.base_model = mobilenet.features.to(device)

    def forward(self, x):
        x = self.base_model(x)
        # [IMPORTANT] normalize
        x = F.normalize(x, p=2, dim=2)  # intra-normalization
        x = x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(x, p=2, dim=1)  # L2 normalize

        return vlad_x
