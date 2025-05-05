import torch
from torch import nn

from einops import repeat, rearrange
from scipy.spatial.transform import Rotation as R
import numpy as np


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class VectorInstanceMemory(nn.Module):

    def __init__(self,
                 dim_in, number_ins, bank_size, mem_len, mem_select_dist_ranges
                 ):
        super().__init__()
        self.max_number_ins = 3 * number_ins # make sure this is not exceeded at initial training when results could be quite random
        self.bank_size = bank_size
        self.mem_len = mem_len
        self.dim_in = dim_in
        self.mem_select_dist_ranges = mem_select_dist_ranges

        p_enc_1d = PositionalEncoding1D(dim_in)
        fake_tensor = torch.zeros((1, 1000, dim_in)) # suppose all sequences are shorter than 1000
        self.cached_pe = p_enc_1d(fake_tensor)[0]

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def set_bank_size(self, bank_size):
        self.bank_size = bank_size

    def init_memory(self, bs):
        self.mem_bank = torch.zeros((self.bank_size, bs, self.max_number_ins, self.dim_in), dtype=torch.float32).cuda()
        self.mem_bank_seq_id = torch.zeros((self.bank_size, bs, self.max_number_ins), dtype=torch.long).cuda()
        self.mem_bank_trans = torch.zeros((self.bank_size, bs,  3),dtype=torch.float32).cuda()
        self.mem_bank_rot = torch.zeros((self.bank_size, bs, 3, 3),dtype=torch.float32).cuda()
        self.batch_mem_embeds_dict = {}
        self.batch_mem_relative_pe_dict = {}
        self.batch_key_padding_dict = {}
        self.curr_rot = torch.zeros((bs,3,3),dtype=torch.float32).cuda()
        self.curr_trans = torch.zeros((bs,3),dtype=torch.float32).cuda()
        self.gt_lines_info = {}

        # memory recording information
        self.instance2mem = [{} for _ in range(bs)]
        self.num_ins = [0 for _ in range(bs)]
        self.active_mem_ids = [None for _ in range(bs)]
        self.valid_track_idx = [None for _ in range(bs)]
        self.random_bev_masks = [None for _ in range(bs)]
        init_entry_length = torch.tensor([0]*self.max_number_ins).long()
        self.mem_entry_lengths = [init_entry_length.clone() for _ in range(bs)]

    def update_memory(self, batch_i, is_first_frame, propagated_ids, prev_out, num_tracks, 
                      seq_idx, timestep):
        if is_first_frame:
            mem_instance_ids = torch.arange(propagated_ids.shape[0])
            track2mem_info = {i: i for i in range(len(propagated_ids))}
            num_instances = len(propagated_ids)
        else:
            track2mem_info_prev = self.instance2mem[batch_i]
            track2mem_info = {}
            num_instances = self.num_ins[batch_i]
            for pred_i, propagated_id in enumerate(propagated_ids):
                if propagated_id < num_tracks: # existing tracks
                    track2mem_info[pred_i] = track2mem_info_prev[propagated_id.item()]
                else: # newborn instances
                    track2mem_info[pred_i] = num_instances
                    num_instances += 1
            mem_instance_ids = torch.tensor([track2mem_info[item] for item in range(len(propagated_ids))]).long()
        
        assert num_instances < self.max_number_ins, 'Number of instances larger than mem size!'

        #NOTE: put information into the memory, need to detach the scores to block gradient backprop 
        # from future time steps
        prev_embeddings = prev_out['hs_embeds'][batch_i]
        prev_scores = prev_out['scores'][batch_i]
        prev_scores, prev_labels = prev_scores.max(-1)
        prev_scores = prev_scores.sigmoid().detach()
        
        mem_lens_per_ins = self.mem_entry_lengths[batch_i][mem_instance_ids]

        # insert information into mem bank
        for ins_idx, mem_id in enumerate(mem_instance_ids):
            if mem_lens_per_ins[ins_idx] < self.bank_size:
                self.mem_bank[mem_lens_per_ins[ins_idx], batch_i, mem_id] = prev_embeddings[propagated_ids[ins_idx]]
                self.mem_bank_seq_id[mem_lens_per_ins[ins_idx], batch_i, mem_id] = seq_idx
            else:
                self.mem_bank[:self.bank_size-1, batch_i, mem_id] = self.mem_bank[1:self.bank_size, batch_i, mem_id]
                self.mem_bank[-1, batch_i, mem_id] = prev_embeddings[propagated_ids[ins_idx]]
                self.mem_bank_seq_id[:self.bank_size-1, batch_i, mem_id] = self.mem_bank_seq_id[1:self.bank_size, batch_i, mem_id]
                self.mem_bank_seq_id[-1, batch_i, mem_id] = seq_idx

        if self.curr_t < self.bank_size:
            self.mem_bank_rot[self.curr_t, batch_i] = self.curr_rot[batch_i]
            self.mem_bank_trans[self.curr_t, batch_i] = self.curr_trans[batch_i]
        else:
            self.mem_bank_rot[:self.bank_size-1, batch_i] = self.mem_bank_rot[1:, batch_i].clone()
            self.mem_bank_rot[-1, batch_i] = self.curr_rot[batch_i]
            self.mem_bank_trans[:self.bank_size-1, batch_i] = self.mem_bank_trans[1:, batch_i].clone()
            self.mem_bank_trans[-1, batch_i] = self.curr_trans[batch_i]

        # Update the mem recording information
        self.instance2mem[batch_i] = track2mem_info
        self.num_ins[batch_i] = num_instances
        self.mem_entry_lengths[batch_i][mem_instance_ids] += 1
        self.active_mem_ids[batch_i] = mem_instance_ids.long().to(propagated_ids.device)
        active_mem_entry_lens = self.mem_entry_lengths[batch_i][self.active_mem_ids[batch_i]]
        self.valid_track_idx[batch_i] = torch.where(active_mem_entry_lens >= 1)[0]

        #print('Active memory ids:', self.active_mem_ids[batch_i])
        #print('Memory entry lens:', active_mem_entry_lens)
        #print('Valid track idx:', self.valid_track_idx[batch_i])

    def prepare_transformation_batch(self,history_e2g_trans,history_e2g_rot,curr_e2g_trans,curr_e2g_rot):
        history_g2e_matrix = torch.stack([torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device),]*len(history_e2g_trans), dim=0)
        history_g2e_matrix[:, :3, :3] = torch.transpose(history_e2g_rot, 1, 2)
        history_g2e_matrix[:, :3, 3] = -torch.bmm(torch.transpose(history_e2g_rot, 1, 2), history_e2g_trans[..., None]).squeeze(-1)

        curr_g2e_matrix = torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

        curr_e2g_matrix = torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device)
        curr_e2g_matrix[:3, :3] = curr_e2g_rot
        curr_e2g_matrix[:3, 3] = curr_e2g_trans

        history_e2g_matrix = torch.stack([torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device),]*len(history_e2g_trans), dim=0)
        history_e2g_matrix[:, :3, :3] = history_e2g_rot
        history_e2g_matrix[:, :3, 3] = history_e2g_trans

        history_curr2prev_matrix = torch.bmm(history_g2e_matrix, repeat(curr_e2g_matrix,'n1 n2 -> r n1 n2', r=len(history_g2e_matrix)))
        history_prev2curr_matrix = torch.bmm(repeat(curr_g2e_matrix, 'n1 n2 -> r n1 n2', r=len(history_e2g_matrix)), history_e2g_matrix)
        
        return history_curr2prev_matrix, history_prev2curr_matrix

    def clear_dict(self,):
        self.batch_mem_embeds_dict = {}
        self.batch_mem_relative_pe_dict = {}
        self.batch_key_padding_dict = {}

    def trans_memory_bank(self, query_prop, b_i, metas):
        seq_id = metas['local_idx']
        
        active_mem_ids = self.active_mem_ids[b_i]
        mem_entry_lens = self.mem_entry_lengths[b_i][active_mem_ids]
        num_track_ins = len(active_mem_ids)
        valid_mem_len = min(self.curr_t, self.mem_len)
        valid_bank_size = min(self.curr_t, self.bank_size)
        mem_trans = self.mem_bank_trans[:, b_i]
        mem_rots = self.mem_bank_rot[:, b_i]

        if self.training:
            # Note: at training time, bank_size must be the same as mem_len, no selection needed
            assert self.mem_len == self.bank_size, 'at training time, bank_size must be the same as mem_len'
            mem_embeds = self.mem_bank[:, b_i, active_mem_ids]
            mem_seq_ids = self.mem_bank_seq_id[:, b_i, active_mem_ids]
        else:
            # at test time, the bank size can be much longer, and we need the selection strategy
            mem_embeds = torch.zeros_like(self.mem_bank[:self.mem_len, b_i, active_mem_ids])
            mem_seq_ids = torch.zeros_like(self.mem_bank_seq_id[:self.mem_len, b_i, active_mem_ids])

        # Put information into mem embeddings and pos_ids, prepare for attention-fusion
        # Also prepare the pose information for the query propagation
        all_pose_select_indices = []
        all_select_indices = []
        for idx, active_idx in enumerate(active_mem_ids):
            effective_len = mem_entry_lens[idx]
            valid_mem_trans = mem_trans[:valid_bank_size]
            trunc_eff_len = min(effective_len, self.bank_size)
            valid_pose_ids = torch.arange(valid_bank_size-trunc_eff_len, valid_bank_size)
            #print('ins {}, valid pose ids {}'.format(idx, valid_pose_ids))
            if effective_len <= self.mem_len:
                select_indices = torch.arange(effective_len)
            else:
                select_indices = self.select_memory_entries(valid_mem_trans[-trunc_eff_len:], metas)
            pose_select_indices = valid_pose_ids[select_indices]
            mem_embeds[:len(select_indices), idx] = self.mem_bank[select_indices, b_i, active_idx]
            mem_seq_ids[:len(select_indices), idx] = self.mem_bank_seq_id[select_indices, b_i, active_idx]
            all_pose_select_indices.append(pose_select_indices)
            all_select_indices.append(select_indices)
        
        # prepare mem padding mask
        key_padding_mask = torch.ones((self.mem_len, num_track_ins)).bool().cuda()
        padding_trunc_loc = torch.clip(mem_entry_lens, max=self.mem_len)
        for ins_i in range(num_track_ins):
            key_padding_mask[:padding_trunc_loc[ins_i], ins_i] = False
        key_padding_mask = key_padding_mask.T

        # prepare relative seq idx gap
        relative_seq_idx = torch.zeros_like(mem_embeds[:,:,0]).long()
        relative_seq_idx[:valid_mem_len] = seq_id - mem_seq_ids[:valid_mem_len]
        relative_seq_pe = self.cached_pe[relative_seq_idx].to(mem_embeds.device)

        # prepare relative pose information for each active instance
        curr2prev_matrix, prev2curr_matrix = self.prepare_transformation_batch(mem_trans[:valid_bank_size],
            mem_rots[:valid_bank_size], self.curr_trans[b_i], self.curr_rot[b_i])
        pose_matrix = prev2curr_matrix.float()[:,:3]
        rot_mat = pose_matrix[..., :3].cpu().numpy()
        rot = R.from_matrix(rot_mat)
        translation = pose_matrix[..., 3] 

        if self.training:
            rot, translation = self.add_noise_to_pose(rot, translation)

        rot_quat = torch.tensor(rot.as_quat()).float().to(pose_matrix.device)
        pose_info = torch.cat([rot_quat, translation], dim=1)
        pose_info_per_ins = torch.zeros((valid_mem_len, num_track_ins, pose_info.shape[1])).to(pose_info.device)

        for ins_idx in range(num_track_ins):
            pose_select_indices = all_pose_select_indices[ins_idx]
            pose_info_per_ins[:len(pose_select_indices), ins_idx] = pose_info[pose_select_indices]

        mem_embeds_new = mem_embeds.clone()
        mem_embeds_valid = rearrange(mem_embeds[:valid_mem_len], 't n c -> (t n) c')
        pose_info_per_ins = rearrange(pose_info_per_ins, 't n c -> (t n) c')
        mem_embeds_prop = query_prop(
            mem_embeds_valid,
            pose_info_per_ins
        )
        mem_embeds_new[:valid_mem_len] = rearrange(mem_embeds_prop, '(t n) c -> t n c', t=valid_mem_len)

        self.batch_mem_embeds_dict[b_i] = mem_embeds_new.clone().detach()
        self.batch_mem_relative_pe_dict[b_i] = relative_seq_pe
        self.batch_key_padding_dict[b_i] = key_padding_mask
    
    def add_noise_to_pose(self, rot, trans):
        rot_euler = rot.as_euler('zxy')
        # 0.08 mean is around 5-degree, 3-sigma is 15-degree
        noise_euler = np.random.randn(*list(rot_euler.shape)) * 0.08
        rot_euler += noise_euler
        noisy_rot = R.from_euler('zxy', rot_euler)

        # error within 0.25 meter
        noise_trans = torch.randn_like(trans) * 0.25
        noise_trans[:, 2] = 0
        noisy_trans = trans + noise_trans

        return noisy_rot, noisy_trans

    def select_memory_entries(self, mem_trans, curr_meta):
        history_e2g_trans = mem_trans[:, :2].cpu().numpy()
        curr_e2g_trans = np.array(curr_meta['ego2global_translation'][:2])
        dists = np.linalg.norm(history_e2g_trans - curr_e2g_trans[None, :], axis=1)

        sorted_indices = np.argsort(dists)
        sorted_dists = dists[sorted_indices]
        covered = np.zeros_like(sorted_indices).astype(np.bool)
        selected_ids = []
        for dist_range in self.mem_select_dist_ranges[::-1]:
            outter_valid_flags = (sorted_dists >= dist_range) & ~covered
            if outter_valid_flags.any():
                pick_id = np.where(outter_valid_flags)[0][0]     
                covered[pick_id:] = True
            else:
                inner_valid_flags = (sorted_dists < dist_range) & ~covered
                if inner_valid_flags.any():
                    pick_id = np.where(inner_valid_flags)[0][-1]
                    covered[pick_id] = True
                else:
                    # return the mem_len closest one, but in the order of far -> close
                    return np.array(sorted_indices[:4][::-1])
            selected_ids.append(pick_id)

        selected_mem_ids = sorted_indices[np.array(selected_ids)]
        return selected_mem_ids
        
    
            
