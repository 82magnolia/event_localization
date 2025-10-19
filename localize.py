import torch
import cv2
import os
import time
from typing import NamedTuple
import warnings
from log_utils import save_logger, PoseLogger
import data_utils
from pose_estimation import get_matcher, get_netvlad, refine_from_pnp_ransac, MobileNetVLAD
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np
from e2vid import E2VID
from e2vid.utils.voxelgrid import VoxelGrid
from privacy_utils import PrivacyTester, ReconTester, apply_voxel_weight
from train_voxel_regression import UpsampleConvLayer, FrontalInverter, UNet


warnings.filterwarnings("ignore", category=UserWarning) 


def localize(cfg: NamedTuple, log_dir: str):
    """
    Main function for performing localization.

    Args:
        cfg: Config file
        log_dir: Directory in which logs will be saved
    
    Returns:
        None
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dataset = cfg.dataset
    supported_datasets = ['ev_rooms']  # Currently supported datasets

    if dataset not in supported_datasets:
        raise ValueError("Invalid dataset")

    # Get scene names
    if dataset in supported_datasets:
        scene_names = getattr(cfg, 'scene_names', None)
        if scene_names is None:
            scene_names = data_utils.get_scene_names(dataset)
        if isinstance(scene_names, str):
            scene_names = [scene_names]

    logger= PoseLogger(log_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Dataset configs
    split_type = getattr(cfg, 'split_type', None)
    slice_method = getattr(cfg, 'slice_method', 'count')
    count_window = getattr(cfg, 'count_window', 30000)
    time_window = getattr(cfg, 'time_window', 0.05)

    # Algorithm configs
    top_k_candidate = getattr(cfg, 'top_k_candidate', 5)
    refine_mode = getattr(cfg, 'refine_mode', 'match')
    print_retrieval = getattr(cfg, 'print_retrieval', False)
    rep_list = getattr(cfg, 'rep_list', ['s_p', 'b'])
    use_recon_ref = getattr(cfg, 'use_recon_ref', True)
    num_unroll = getattr(cfg, 'num_unroll', 10)  # Number of unrolling for generating candidate views
    hide_enc_conv = getattr(cfg, 'hide_enc_conv', False)
    hide_first_dec = getattr(cfg, 'hide_first_dec', False)

    # Noise injection
    noise_watermark = getattr(cfg, 'noise_watermark', False)
    if noise_watermark:
        watermark_path = cfg.load_model.replace('model_ckpts/model.pth', 'watermark.pt')
        print(f"Using watermark! Loading watermark from {watermark_path}")
        additive_noise = torch.load(watermark_path).numpy()
    else:
        additive_noise = None

    # Match Visualization dictionary
    vis_match_dict = {'conf_thres': getattr(cfg, 'conf_thres', 0.0), 'visualize_all': getattr(cfg, 'visualize_all', False), \
        'draw_matches': getattr(cfg, 'draw_matches', True), 'draw_points_2d': getattr(cfg, 'draw_points_2d', True), \
        'draw_match_kpts': getattr(cfg, 'draw_match_kpts', True), 'draw_img_kpts': getattr(cfg, 'draw_img_kpts', False)}

    # Set matching module
    matcher = get_matcher(cfg, device=device)

    netvlad_backbone = getattr(cfg, 'netvlad_backbone', 'vgg')

    if netvlad_backbone == 'vgg':
        # Set retrieval module
        netvlad = get_netvlad(cfg, device=device)

        # Transforms prior to NetVLAD
        netvlad_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                                            std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
    elif netvlad_backbone == 'mobilenet':
        netvlad = MobileNetVLAD(device)
        netvlad_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    # Setup for event to image reconstruction module
    reconstructor = E2VID(cfg)  # Used for reconstructing references
    query_reconstructor = E2VID(cfg)  # Used for reconstructing query

    # Voxel weights for privacy preservation
    voxel_weight = getattr(cfg, 'voxel_weight', False)
    if voxel_weight:
        print("Using voxel weighting for privacy preservation...")

    grid = VoxelGrid(reconstructor.model.num_bins, cfg.width, cfg.height, upsample_rate=cfg.upsample_rate)

    # Main localization loop
    for scene_name in scene_names:
        print(f"Scene Name: {scene_name}")

        print("STEP 1: NetVLAD feature computation and COLMAP parsing")
        scale = data_utils.get_scale(dataset, scene_name)
        full_img_names = data_utils.get_img_names(dataset, scene_name)
        points_3d = data_utils.get_points_3d(dataset, scene_name, scale)
        camera_intrinsic, distortion_coeff = data_utils.get_camera_intrinsics(dataset, scene_name)
        img_labels, id2img_name = data_utils.get_img_labels(dataset, scene_name, scale)
        query_list, ref_list = data_utils.get_split(dataset, scene_name, split_type, getattr(cfg, 'ref_sample_rate', 1), getattr(cfg, 'exp_mode', 'orig'))
        full_query_img_names, full_ref_img_names, query_img_labels, ref_img_labels = data_utils.filter_img_list(full_img_names, img_labels, query_list, ref_list)
        sort_ref_labels = sorted(ref_img_labels.keys())

        # Get reference image poses for evaluating retrieval
        ref_trans = [data_utils.convert_trans(ref_img_labels[k]['trans'], 'torch', device) for k in ref_img_labels.keys()]
        ref_rot = [data_utils.convert_rot(ref_img_labels[k]['rot'], 'torch', device) for k in ref_img_labels.keys()]

        ref_trans = torch.cat(ref_trans, dim=0)  # (N_r, 3)
        ref_rot = torch.stack(ref_rot, dim=0)  # (N_r, 3, 3)
        ref_trans = -(ref_trans.unsqueeze(1) @ ref_rot).reshape(-1, 3)

        # Prepare event data
        img_timestamps = data_utils.get_img_timestamp(dataset, scene_name)
        events = data_utils.get_events(dataset, scene_name)

        # Prepare NetVLAD features for reference views

        ref_img_fts = []
        ref_imgs = []
        slice_idx = 0
        vis_idx = 0
        ref_idx = 0
        for ref_idx, full_ref_img_name in enumerate(tqdm(full_ref_img_names)):
            # Track events until target timestamp is found
            ref_img_name = full_ref_img_name.split('/')[-1]  # File name excluding all directory names
            logger.add_filename(full_ref_img_name, scene_name)

            # Read image and generate event image
            if 'human' in scene_name or 'dark' in scene_name:  # Load images from reference views captured in 'normal' conditions
                ref_img = cv2.cvtColor(cv2.imread(full_ref_img_name), cv2.COLOR_BGR2RGB)
                ref_imgs.append(ref_img[..., 0])
                ref_ntvd_img = netvlad_transform(Image.fromarray(ref_img.astype(np.uint8)))
            else:
                ref_img = cv2.cvtColor(cv2.imread(full_ref_img_name), cv2.COLOR_BGR2RGB)
                ref_timestamp = img_timestamps[ref_img_name]

                ref_events = data_utils.slice_events(events, ref_timestamp, method=slice_method, count_window=count_window * num_unroll)
                max_roll_idx = ref_events.shape[0] // count_window + 1 if ref_events.shape[0] % count_window != 0 else ref_events.shape[0] // count_window

                reconstructor.image_reconstructor.last_states_for_each_channel = {'grayscale': None}  # re-initialize state

                for roll_idx in range(max_roll_idx):
                    ref_events_grid, _ = grid.events_to_voxel_grid(ref_events[count_window * roll_idx: count_window * (roll_idx + 1), [2, 0, 1, 3]])
                    ref_events_grid = grid.normalize_voxel(ref_events_grid)
                    if noise_watermark:
                        ref_events_grid += additive_noise
                    ref_recon_img = query_reconstructor(ref_events_grid)
                ref_imgs.append(ref_recon_img)
                ref_recon_img = np.stack([ref_recon_img] * 3, axis=-1)
                ref_ntvd_img = netvlad_transform(Image.fromarray(ref_recon_img.astype(np.uint8)))
            with torch.no_grad():
                feature = netvlad(ref_ntvd_img.unsqueeze(0).to(device))[0].cpu()
            ref_img_fts.append(feature)

        ref_img_fts = torch.stack(ref_img_fts, dim=0).to(device)  # (N_r, F)

        valid_trial = 0
        well_posed = 0

        for query_idx, full_query_img_name in enumerate(full_query_img_names):
            # Track events until target timestamp is found
            query_img_name = full_query_img_name.split('/')[-1]  # File name excluding all directory names
            valid_trial += 1            
            logger.add_filename(full_query_img_name, scene_name)

            # Assume all translation to be in form (1, 3) and rotation to be in form (3, 3)
            gt_trans = img_labels[query_img_name]['trans']
            gt_rot = img_labels[query_img_name]['rot']

            gt_trans = data_utils.convert_trans(gt_trans, 'torch', device)
            gt_rot = data_utils.convert_rot(gt_rot, 'torch', device)
            gt_trans = - gt_trans @ gt_rot  # COLMAP convention for getting camera center

            # Read image and generate event image
            query_img = cv2.cvtColor(cv2.imread(full_query_img_name), cv2.COLOR_BGR2RGB)
            query_timestamp = img_timestamps[query_img_name]

            query_events = data_utils.slice_events(events, query_timestamp, method=slice_method, count_window=count_window * num_unroll)
            max_roll_idx = query_events.shape[0] // count_window + 1 if query_events.shape[0] % count_window != 0 else query_events.shape[0] // count_window

            query_reconstructor.image_reconstructor.last_states_for_each_channel = {'grayscale': None}  # re-initialize state

            if voxel_weight:
                grid_list = []
                for roll_idx in range(max_roll_idx):
                    query_events_grid, _ = grid.events_to_voxel_grid(query_events[count_window * roll_idx: count_window * (roll_idx + 1), [2, 0, 1, 3]])
                    query_events_grid = grid.normalize_voxel(query_events_grid)
                    grid_list.append(query_events_grid)

                total_grid = np.concatenate(grid_list, axis=0)
                time_gap = query_events[-1, 2] - query_events[0, 2]
                weighted_total_grid = apply_voxel_weight(total_grid, getattr(cfg, 'med_kernel_size', 5), getattr(cfg, 'reflect_kernel_size', 23), 
                    getattr(cfg, 'mean_filtering', True), getattr(cfg, 'mean_kernel_size', 5), getattr(cfg, 'mask_method', 'med'), 
                    getattr(cfg, 'mask_const', 1.0), getattr(cfg, 'dilation_size', 3))
                block_size = weighted_total_grid.shape[0] // max_roll_idx
                
                for roll_idx in range(max_roll_idx):
                    input_voxel = weighted_total_grid[roll_idx * block_size: (roll_idx + 1) * block_size]
                    if noise_watermark:
                        input_voxel += additive_noise
                    query_recon_img = query_reconstructor(input_voxel)

            else:
                for roll_idx in range(max_roll_idx):
                    query_events_grid, _ = grid.events_to_voxel_grid(query_events[count_window * roll_idx: count_window * (roll_idx + 1), [2, 0, 1, 3]])
                    query_events_grid = grid.normalize_voxel(query_events_grid)
                    if noise_watermark:
                        query_events_grid += additive_noise
                    query_recon_img = query_reconstructor(query_events_grid)

            start_time = time.time()

            print(f"Image Name: {full_query_img_name}")
            print("STEP 2: Reference pose retrieval")
            query_recon_img = np.stack([query_recon_img] * 3, axis=-1)
            query_ntvd_img = netvlad_transform(Image.fromarray(query_recon_img.astype(np.uint8)))
            query_recon_img = torch.from_numpy(query_recon_img).permute(2, 0, 1) / 255

            # Query event NetVLAD feature computation & candidate view selection
            with torch.no_grad():
                query_ev_ft = netvlad(query_ntvd_img.unsqueeze(0).to(device))[0]
            cost_mtx = (ref_img_fts - query_ev_ft.unsqueeze(0)).norm(dim=-1)  # (N_r, )
            min_inds = cost_mtx.argsort()[:top_k_candidate]

            # Event image generation for candidate views
            if use_recon_ref:
                cand_imgs = []
                cand_names = [sort_ref_labels[idx] for idx in min_inds]
                for vis_idx, min_idx in enumerate(min_inds):
                    cand_recon_img = ref_imgs[min_idx]
                    cand_recon_img = np.stack([cand_recon_img] * 3, axis=-1)
                    cand_imgs.append(torch.from_numpy(cand_recon_img).permute(2, 0, 1) / 255)
                cand_imgs = torch.stack(cand_imgs, dim=0).to(device)

            # Pose Refinement
            print(f"STEP 3: Pose refinement with {refine_mode}")                
            if refine_mode == 'pnp_ransac':
                best_idx, best_trans, best_rot, top_k_trans, top_k_rot, rf_trans, rf_rot \
                    = refine_from_pnp_ransac(query_recon_img.to(device), camera_intrinsic, distortion_coeff, cand_names, \
                    cand_imgs, points_3d, ref_img_labels, matcher, getattr(cfg, 'visualize_match', False), visualize_match_dict=vis_match_dict, \
                    no_ransac=getattr(cfg, 'no_ransac', False), rgb_weights=getattr(cfg, 'rgb_weights', [1 / 3, 1 / 3, 1 / 3]), log_dir=log_dir, match_vis_idx=query_idx)

            # Measure time for localization
            localization_time = time.time() - start_time
            print(f"Elapsed time: {localization_time}")
             
            # Evaluate estimated rotations with ground truth
            top_k_t_error = (top_k_trans - gt_trans).norm(dim=-1)

            top_k_r_error = torch.matmul(torch.transpose(top_k_rot, dim0=-2, dim1=-1), gt_rot.unsqueeze(0))
            top_k_r_error = torch.diagonal(top_k_r_error, dim1=-2, dim2=-1).sum(-1)
            top_k_r_error[top_k_r_error < -1] = -2 - top_k_r_error[top_k_r_error < -1]
            top_k_r_error[top_k_r_error > 3] = 6 - top_k_r_error[top_k_r_error > 3]
            top_k_r_error = torch.rad2deg(torch.abs(torch.arccos((top_k_r_error - 1) / 2)))

            t_error = (rf_trans.cpu() - gt_trans.cpu().squeeze()).norm().item()
            r_error = torch.matmul(torch.transpose(rf_rot, dim0=-2, dim1=-1), gt_rot.unsqueeze(0))
            r_error = torch.diagonal(r_error, dim1=-2, dim2=-1).sum(-1)
            if r_error < -1:
                r_error = -2 - r_error
            elif r_error > 3:
                r_error = 6 - r_error
            r_error = torch.rad2deg(torch.abs(torch.arccos((r_error - 1) / 2))).item()

            best_t_error = (best_trans.cpu() - gt_trans.cpu().squeeze()).norm().item()
            best_r_error = torch.matmul(torch.transpose(best_rot, dim0=-2, dim1=-1), gt_rot.unsqueeze(0))
            best_r_error = torch.diagonal(best_r_error, dim1=-2, dim2=-1).sum(-1)
            if best_r_error < -1:
                best_r_error = -2 - best_r_error
            elif best_r_error > 3:
                best_r_error = 6 - best_r_error
            best_r_error = torch.rad2deg(torch.abs(torch.arccos((best_r_error - 1) / 2))).item()

            logger.add_error(t_error, r_error, full_query_img_name, scene_name)
            logger.add_estimate(full_query_img_name, rf_trans.squeeze().cpu().numpy(), rf_rot.cpu().numpy())

            print("=============== CURRENT RESULTS ===============")
            if print_retrieval:
                print("retrieved t-error: ", top_k_t_error.min().item())
                print("retrieved r-error: ", top_k_r_error.min().item())

                orcl_t_error = (ref_trans - gt_trans).norm(dim=-1)
                orcl_r_error = torch.matmul(torch.transpose(ref_rot, dim0=-2, dim1=-1), gt_rot.unsqueeze(0))
                orcl_r_error = torch.diagonal(orcl_r_error, dim1=-2, dim2=-1).sum(-1)
                orcl_r_error[orcl_r_error < -1] = -2 - orcl_r_error[orcl_r_error < -1]
                orcl_r_error[orcl_r_error > 3] = 6 - orcl_r_error[orcl_r_error > 3]
                orcl_r_error = torch.rad2deg(torch.abs(torch.arccos((orcl_r_error - 1) / 2)))
                print("oracle t-error: ", orcl_t_error.min().item())
                print("oracle r-error: ", orcl_r_error.min().item())
                print("best t-error: ", best_t_error)
                print("best r-error: ", best_r_error)

            print("t-error: ", t_error)
            print("r-error: ", r_error)

            if (t_error < 0.1) and (r_error < 5):
                well_posed += 1
            print("Accuracy: ", well_posed / valid_trial)
            print("===============================================")

    # Calculate statistics and save logger
    logger.calc_statistics('scene')
    logger.calc_statistics('total')
    save_logger(os.path.join(log_dir, getattr(cfg, 'log_name', 'result.pkl')), logger)
