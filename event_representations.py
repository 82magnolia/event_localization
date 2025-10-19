import numpy as np
import torch
from torch_scatter import scatter_max


def normalize_ev_acc(ev_acc, mode='all_pos'):
    # ev_acc is assumed to be a tensor of shape (B, 1, H, W)
    B, _, H, W = ev_acc.shape
    flat_ev_acc = ev_acc.reshape(B, -1)
    qt_range = [0.99, 0.999, 0.9999]
    clip_val = torch.zeros(B)
    for q in qt_range:
        new_clip_val = torch.maximum(torch.quantile(flat_ev_acc, q, dim=-1).abs(), torch.quantile(flat_ev_acc, 1 - q, dim=-1).abs())  # (B, )
        clip_val[clip_val == 0] = new_clip_val[clip_val == 0]  # Replace clip values with new nonzero ones

    clip_val = clip_val.reshape(B, 1, 1, 1)

    # Clip regions over each clip value
    ev_acc[ev_acc > clip_val] = clip_val.repeat(1, 1, H, W)[ev_acc > clip_val]
    ev_acc[ev_acc < -clip_val] = -clip_val.repeat(1, 1, H, W)[ev_acc < -clip_val]

    # Normalize with respect to clip ranges
    non_zero_clip = clip_val.reshape(B) != 0  # (B, )

    if mode == 'all_pos':  # Make all values to be within [0, 1]
        ev_acc[non_zero_clip] = (ev_acc[non_zero_clip] + clip_val[non_zero_clip]) / (2 * clip_val[non_zero_clip])
    elif mode == 'zero_hold':  # Normalize with zero at center and in range [-1, 1]
        ev_acc[non_zero_clip] = ev_acc[non_zero_clip] / clip_val[non_zero_clip]
    elif mode == 'half_hold':  # Normalize with zero at center and in range [-0.5, 0.5]
        ev_acc[non_zero_clip] = ev_acc[non_zero_clip] / (2 * clip_val[non_zero_clip])

    return ev_acc


# Each representation returns a (H, W) image representation
def parse_cfg_image(event_tensor, H, W, rep_list):
    # Parse ref_list and return a representation

    def _parse(rep_name):
        if rep_name == 'b_p':
            return binary_image(event_tensor, H, W, polarity=True)
        elif rep_name == 'b':
            return binary_image(event_tensor, H, W)
        elif rep_name == 't_p':
            return timestamp_image(event_tensor, H, W, polarity=True)
        elif rep_name == 't':
            return timestamp_image(event_tensor, H, W)
        elif rep_name == 'bg':
            return binary_greyscale_image(event_tensor, H, W)
        elif rep_name == 'bg_r':
            return binary_greyscale_image(event_tensor, H, W, reverse=True)
        elif rep_name == 'bg_n':
            return binary_greyscale_image(event_tensor, H, W, neutral=True)
        elif rep_name == 's_p':
            return sort_image(event_tensor, H, W, polarity=True)
        elif rep_name == 's':
            return sort_image(event_tensor, H, W, polarity=False)
        elif rep_name == 'c_p':
            return count_image(event_tensor, H, W, polarity=True)
        elif rep_name == 'c_sp':
            return count_image(event_tensor, H, W, polarity=True, sum_polarity=True)
        elif rep_name == 'c_nsp':
            return count_image(event_tensor, H, W, polarity=True, sum_polarity=True, sim_normalize=True)
        elif rep_name == 'c_wsp':
            return count_image(event_tensor, H, W, polarity=True, weighted_sum_polarity=True)
        elif rep_name == 'c_wnsp':
            return count_image(event_tensor, H, W, polarity=True, weighted_sum_polarity=True, sim_normalize=True)
        elif rep_name == 'c':
            return count_image(event_tensor, H, W)
        elif rep_name == 'e_p':
            return timestamp_image(event_tensor, H, W, polarity=True, exp=True)
        elif rep_name == 'e':
            return timestamp_image(event_tensor, H, W, exp=True)

    event_img_list = []
    for rep in rep_list:
        event_rep = _parse(rep)
        if isinstance(event_rep, torch.Tensor):
            event_img_list.append(event_rep)
        else:
            event_img_list += [*event_rep]
    
    event_img = torch.stack(event_img_list, dim=2)
    event_img = event_img.permute(2, 0, 1)
    event_img = event_img.float()

    return event_img


def binary_image(event_tensor, H, W, polarity=False):
    if polarity:
        pos = event_tensor[event_tensor[:, 3] > 0]
        neg = event_tensor[event_tensor[:, 3] < 0]
        
        pos_coords = pos.long()
        neg_coords = neg.long()

        pos_event_image = torch.zeros([H, W])
        neg_event_image = torch.zeros([H, W])

        pos_event_image[(pos_coords[:, 1], pos_coords[:, 0])] = 1.0
        neg_event_image[(neg_coords[:, 1], neg_coords[:, 0])] = 1.0

        return pos_event_image, neg_event_image
    else:
        coords = event_tensor[:, :2].long()
        event_image = torch.zeros([H, W])
        event_image[(coords[:, 1], coords[:, 0])] = 1.0

        return event_image


def timestamp_image(event_tensor, H, W, polarity=False, exp=False):
    EXP_TAU = 0.3
    if polarity:
        pos = event_tensor[event_tensor[:, 3] > 0]
        neg = event_tensor[event_tensor[:, 3] < 0]
        start_time = event_tensor[0, 2]
        time_length = event_tensor[-1, 2] - event_tensor[0, 2]

        # Positive, negative timestamp image
        norm_pos_time = (pos[:, 2] - start_time) / time_length
        norm_neg_time = (neg[:, 2] - start_time) / time_length
        pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
        neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
        pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
        pos_out = pos_out.reshape(H, W)
        neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
        neg_out = neg_out.reshape(H, W)
        
        if exp:
            pos_out = torch.exp(-(1 - pos_out) / EXP_TAU)
            neg_out = torch.exp(-(1 - neg_out) / EXP_TAU)

        return pos_out, neg_out
    else:
        start_time = event_tensor[0, 2]
        time_length = event_tensor[-1, 2] - event_tensor[0, 2]

        norm_time = (event_tensor[:, 2] - start_time) / time_length
        event_idx = event_tensor[:, 0].long() + event_tensor[:, 1].long() * W
        
        event_out, _ = scatter_max(norm_time, event_idx, dim=-1, dim_size=H * W)
        event_out = event_out.reshape(H, W)

        if exp:
            event_out = torch.exp(-(1 - event_out) / EXP_TAU)

        return event_out


def binary_greyscale_image(event_tensor, H, W, reverse=False, neutral=False):
    if neutral:
        coords = event_tensor[:, :2].long()
        neut_event_image = torch.ones([H, W]) * 0.5
        neut_event_image[(coords[:, 1], coords[:, 0])] = 1.0 
        return neut_event_image
    else:
        pos = event_tensor[event_tensor[:, 3] > 0]
        neg = event_tensor[event_tensor[:, 3] < 0]
        pos_coords = pos.long()
        neg_coords = neg.long()

        if reverse:
            event_image = torch.ones([H, W]) * 0.5
            event_image[(pos_coords[:, 1], pos_coords[:, 0])] = 1.0
            event_image[(neg_coords[:, 1], neg_coords[:, 0])] = 0.0
            return event_image
        else:
            rev_event_image = torch.ones([H, W]) * 0.5
            rev_event_image[(pos_coords[:, 1], pos_coords[:, 0])] = 0.0
            rev_event_image[(neg_coords[:, 1], neg_coords[:, 0])] = 1.0
            return rev_event_image


def sort_image(event_tensor, H, W, polarity=False):
    if polarity:
        TIME_SCALE = 1000000
        time_idx = (event_tensor[:, 2] * TIME_SCALE).long()
        pos_time_idx = time_idx[event_tensor[:, 3] > 0]
        neg_time_idx = time_idx[event_tensor[:, 3] < 0]

        pos_mem, pos_cnt = torch.unique_consecutive(pos_time_idx, return_counts=True)
        pos_time_idx = torch.repeat_interleave(torch.arange(pos_mem.shape[0]), pos_cnt)

        neg_mem, neg_cnt = torch.unique_consecutive(neg_time_idx, return_counts=True)
        neg_time_idx = torch.repeat_interleave(torch.arange(neg_mem.shape[0]), neg_cnt)
        
        event_tensor[:, 2] = time_idx
        pos = event_tensor[event_tensor[:, 3] > 0]
        neg = event_tensor[event_tensor[:, 3] < 0]

        if pos.shape[0] == 0:
            pos = torch.zeros(1, 4)
            pos[:, -1] = 1
        if neg.shape[0] == 0:
            neg = torch.zeros(1, 4)
            neg[:, -1] = 1

        # Get pos sort
        pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
        pos_scatter_result, pos_scatter_idx = scatter_max(pos[:, 2], pos_idx, dim=-1, dim_size=H * W)

        pos_idx_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
        pos_idx_mask[pos_scatter_idx[pos_scatter_idx < pos_idx.shape[0]]] = True
        tmp_pos = pos[pos_idx_mask]

        pos_final_mem, pos_final_cnt = torch.unique_consecutive(tmp_pos[:, 2], return_counts=True)
        # One is added to ensure that sorted values are greater than 1
        pos_final_scatter = torch.repeat_interleave(torch.arange(pos_final_mem.shape[0]), pos_final_cnt).float() + 1

        if pos_final_scatter.max() != pos_final_scatter.min():
            pos_final_scatter = (pos_final_scatter - pos_final_scatter.min()) / (pos_final_scatter.max() - pos_final_scatter.min())
        else:
            pos_final_scatter.fill_(0.0)
        
        pos_sort = torch.zeros(H, W)
        pos_coords = tmp_pos[:, :2].long()

        pos_sort[(pos_coords[:, 1], pos_coords[:, 0])] = pos_final_scatter
        pos_sort = pos_sort.reshape(H, W)

        # Get neg_sort
        neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
        neg_scatter_result, neg_scatter_idx = scatter_max(neg[:, 2], neg_idx, dim=-1, dim_size=H * W)

        neg_idx_mask = torch.zeros(neg.shape[0], dtype=torch.bool)
        neg_idx_mask[neg_scatter_idx[neg_scatter_idx < neg_idx.shape[0]]] = True
        tmp_neg = neg[neg_idx_mask]

        neg_final_mem, neg_final_cnt = torch.unique_consecutive(tmp_neg[:, 2], return_counts=True)
        # One is added to ensure that sorted values are greater than 1
        neg_final_scatter = torch.repeat_interleave(torch.arange(neg_final_mem.shape[0]), neg_final_cnt).float() + 1
        if neg_final_scatter.max() != neg_final_scatter.min():
            neg_final_scatter = (neg_final_scatter - neg_final_scatter.min()) / (neg_final_scatter.max() - neg_final_scatter.min())
        else:
            neg_final_scatter.fill_(0.0)

        neg_sort = torch.zeros(H, W)
        neg_coords = tmp_neg[:, :2].long()

        neg_sort[(neg_coords[:, 1], neg_coords[:, 0])] = neg_final_scatter
        neg_sort = neg_sort.reshape(H, W)

        return pos_sort, neg_sort
    
    else:
        idx = event_tensor[:, 0].long() + event_tensor[:, 1].long() * W
        scatter_result, scatter_idx = scatter_max(event_tensor[:, 2], idx, dim=-1, dim_size=H * W)

        idx_mask = torch.zeros(event_tensor.shape[0], dtype=torch.bool)
        idx_mask[scatter_idx[scatter_idx < idx.shape[0]]] = True
        event_tensor = event_tensor[idx_mask]

        final_mem, final_cnt = torch.unique_consecutive(event_tensor[:, 2], return_counts=True)
        # One is added to ensure that sorted values are greater than 1
        final_scatter = torch.repeat_interleave(torch.arange(final_mem.shape[0]), final_cnt).float() + 1
        if final_scatter.max() != final_scatter.min():
            final_scatter = (final_scatter - final_scatter.min()) / (final_scatter.max() - final_scatter.min())
        else:
            final_scatter.fill_(0.0)

        event_sort = torch.zeros(H, W)
        coords = event_tensor[:, :2].long()

        event_sort[(coords[:, 1], coords[:, 0])] = final_scatter
        event_sort = event_sort.reshape(H, W)

        return event_sort


def count_image(event_tensor, H, W, polarity=False, sum_polarity=False, weighted_sum_polarity=False, sim_normalize=False):
    if polarity:
        pos = event_tensor[event_tensor[:, 3] > 0]
        neg = event_tensor[event_tensor[:, 3] < 0]

        # Get pos, neg counts
        pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
        neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)

        if sum_polarity:
            if sim_normalize:
                sum_image = (pos_count - neg_count).float().reshape(1, 1, H, W)
                sum_image = normalize_ev_acc(sum_image).squeeze()
                return sum_image
            else:
                sum_image = (pos_count - neg_count).float()
                clip_val = max(torch.quantile(sum_image, 0.99).abs(), torch.quantile(sum_image, 0.01).abs())
                sum_image = torch.clip(sum_image, min=-clip_val, max=clip_val)
                sum_image = (sum_image + clip_val) / (2 * clip_val)
            return sum_image
        elif weighted_sum_polarity:
            Ne = event_tensor.shape[0]
            ev_weights = torch.arange(Ne, device=event_tensor.device) / Ne
            pos_weights = ev_weights[event_tensor[:, 3] > 0]
            neg_weights = ev_weights[event_tensor[:, 3] < 0]

            weight_pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W, weights=pos_weights).reshape(H, W)
            weight_neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W, weights=neg_weights).reshape(H, W)

            weight_sum_image = weight_pos_count - weight_neg_count
            
            if sim_normalize:
                weight_sum_image = weight_sum_image.float().reshape(1, 1, H, W)
                weight_sum_image = normalize_ev_acc(weight_sum_image).squeeze()
            else:
                weight_sum_image = torch.clip(weight_sum_image, min=-10, max=10)
                weight_sum_image = (weight_sum_image + 5) / 10
            return weight_sum_image
        else:
            pos_count = pos_count.float()
            neg_count = neg_count.float()
            pos_clip_val = torch.quantile(pos_count, 0.99)
            neg_clip_val = torch.quantile(neg_count, 0.99).abs()
            clip_val = max(pos_clip_val, neg_clip_val)
            if clip_val != 0:
                pos_count = torch.clip(pos_count, max=clip_val) / clip_val
                neg_count = torch.clip(neg_count, max=clip_val) / clip_val

            return pos_count, neg_count
    else:
        event_count = torch.bincount(event_tensor[:, 0].long() + event_tensor[:, 1].long() * W, minlength=H * W).reshape(H, W)
        return event_count
