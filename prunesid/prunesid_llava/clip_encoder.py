import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
import math

def batch_similarity_nms(similarity_matrix, scores, threshold):
    """
    Batched per-group greedy NMS without materializing [B,K,N,N].

    similarity_matrix: [B, N, N]
    scores: [B, N, K], already zero outside each token's group
    threshold: [B]

    This keeps the original greedy order and suppression rule exactly; only the
    redundant K-fold similarity expansion is removed.
    """
    new_scores = scores.clone()
    return_scores = new_scores.clone()
    batch_size, N, group = new_scores.shape
    batch_idx = torch.arange(batch_size, device=new_scores.device).unsqueeze(1).expand(-1, group)
    group_idx = torch.arange(group, device=new_scores.device).unsqueeze(0).expand(batch_size, -1)
    keep_counts = torch.zeros(batch_size, group, dtype=torch.long, device=new_scores.device)
    given_score = 1000

    while True:
        max_values, max_idx = new_scores.max(dim=1)  # [B,K]
        active = max_values > 0
        if not bool(active.any()):
            break

        active_b, active_g = torch.where(active)
        active_token = max_idx[active_b, active_g]
        return_scores[active_b, active_token, active_g] = given_score
        new_scores[active_b, active_token, active_g] = 0
        keep_counts[active_b, active_g] += 1
        given_score -= 1

        # Gather one similarity row per (batch, group). Scores outside the group
        # are already zero, so cross-group entries cannot affect the result.
        sim_rows = similarity_matrix[batch_idx, max_idx]  # [B,K,N]
        suppress = sim_rows > threshold.view(batch_size, 1, 1)
        new_scores.masked_fill_(suppress.transpose(1, 2), 0)

    return keep_counts, return_scores

    

def batch_pca(features, min_components=32):
    standard_features = torch.sigmoid(features.to(torch.float32)).transpose(2,1)[:,:,1:]
    U, S, V = torch.pca_lowrank(standard_features, q=min_components)
    V = torch.abs(V)
    belong_components = torch.argmax(V, dim=-1)
    return V, belong_components


def group_tokens(features, min_components=32, group_method="dgsm"):
    """Stage-1 grouping with PSCA, DGSM-CDKM, CDKM-AISM, or DGSM+AISM."""
    if group_method in (None, "psca", "pca"):
        return batch_pca(features, min_components=min_components)
    if group_method in ("dgsm", "dgsm_cdkm", "cdkm"):
        from prunesid.clustering import batch_dgsm_cdkm
        return batch_dgsm_cdkm(features, min_components=min_components, drop_cls=True)
    if group_method in ("aism", "cdkm_aism"):
        from prunesid.clustering import batch_cdkm_aism
        return batch_cdkm_aism(
            features, min_components=min_components, drop_cls=True, init_mode="kmeans++"
        )
    if group_method == "dgsm_aism":
        from prunesid.clustering import batch_cdkm_aism
        return batch_cdkm_aism(
            features, min_components=min_components, drop_cls=True, init_mode="dgsm"
        )
    raise ValueError(
        f"Unknown group_method={group_method!r}; use 'psca', 'dgsm', 'aism', or 'dgsm_aism'"
    )


class CLIPVisionTower_PruneSID(nn.Module):


    @torch.no_grad()
    def forward(self, images):
        # images: torch.Tensor [batch_size, 3, 336, 336]
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-2]
            hidden_states = image_forward_outs.hidden_states[-2] # [1, 577, 1024]

            need_token_num = self.need_token_num if self.need_token_num else 192
            group_method = getattr(self, "group_method", "dgsm")
            # K aligned with PSCA: need_token_num / 4
            projector_lengths, belong_components = group_tokens(
                hidden_states,
                min_components=int(need_token_num / 4),
                group_method=group_method,
            ) # [batch_size, 576, group_num], [batch_size, 576]
            cls_idx = 0
            cls_attention = attn_weights[:, :, cls_idx, cls_idx + 1 :]
            cls_attention_sum = cls_attention.sum(dim=1) # [batch_size, 576]
            projector_scores = cls_attention_sum.unsqueeze(-1).repeat(1,1, projector_lengths.shape[-1]).to(projector_lengths.dtype) # [batch_size, 576, group_num]
            projector_mask = belong_components.unsqueeze(-1).repeat(1,1, projector_lengths.shape[-1])
            index_map = torch.arange(projector_lengths.shape[-1], device=projector_lengths.device).unsqueeze(0).unsqueeze(0).repeat(projector_lengths.shape[0],projector_lengths.shape[1],1)
            weights_mask = torch.where(projector_mask != index_map)
            projector_lengths[weights_mask] = 0
            projector_scores[weights_mask] = 0
           
            normalized_states = F.normalize(hidden_states[:,1:,:], p=2,dim=-1)
            similarity = torch.bmm(normalized_states, normalized_states.transpose(2,1)).to(torch.float32) # [batch_size， 576， 576]
            triu_mask = torch.triu(torch.ones_like(similarity), diagonal=1).bool()
            sim_mean = (similarity * (triu_mask)).sum(-1).sum(-1) / triu_mask.sum(-1).sum(-1)
            ratio = need_token_num / 32

            
            keep_nms_counts, projector_scores = batch_similarity_nms(
                similarity, projector_scores, ratio * sim_mean
            )
            group_counts = (projector_mask == index_map).sum(dim=1) # [batch_size, group]
            group_lower_bound = torch.ones_like(group_counts, device=group_counts.device)
            group_lower_bound = torch.min(torch.cat([group_lower_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
            group_upper_bound = torch.ones_like(group_counts, device=group_counts.device) * 5 * math.ceil(need_token_num / 64)
            group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
            group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), keep_nms_counts.unsqueeze(0)], dim=0), dim=0)[0]
            while torch.any(group_upper_bound.sum(-1) < need_token_num):
                group_upper_bound[torch.where(group_upper_bound.sum(-1) < need_token_num)] += 1
                group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]

            other_token_nums = need_token_num - group_lower_bound.sum(-1) - 1
            other_token_nums[other_token_nums < 0] = 0
            norm_group_counts = keep_nms_counts / keep_nms_counts.sum(-1, keepdim=True)
            cumulative_sum = torch.cumsum(norm_group_counts, dim=-1)
            other_token_d = (cumulative_sum * other_token_nums.unsqueeze(-1).expand(-1, group_counts.shape[1])).round().int()
            other_token_d = other_token_d - torch.cat([torch.zeros((other_token_d.shape[0], 1), device=other_token_d.device), other_token_d[:, :-1]], dim=-1)
            group_token_d = other_token_d + group_lower_bound
            group_token_d = torch.min(torch.cat([group_token_d.unsqueeze(0), group_upper_bound.unsqueeze(0)], dim=0), dim=0)[0]
            group_sort_index = torch.argsort(keep_nms_counts, dim=-1, descending=True) # [batch, group]
            filling_group = torch.zeros(group_counts.shape[0], device=group_counts.device).int()
            while torch.any(group_token_d.sum(-1) < other_token_nums+group_lower_bound.sum(-1)):
                need_filling_batch = torch.where(group_token_d.sum(-1) < other_token_nums+group_lower_bound.sum(-1))[0]
                filling_num = torch.min(torch.stack(
                    [group_upper_bound[need_filling_batch,group_sort_index[need_filling_batch, filling_group[need_filling_batch]]] - group_token_d[need_filling_batch,group_sort_index[need_filling_batch, filling_group[need_filling_batch]]],
                    other_token_nums[need_filling_batch]+group_lower_bound[need_filling_batch].sum(-1)-group_token_d[need_filling_batch].sum(-1)]
                ),dim=0)[0]
                
                group_token_d[need_filling_batch, group_sort_index[need_filling_batch,filling_group[need_filling_batch]]] += filling_num
                filling_group[need_filling_batch] += 1
            projector_sort_index = torch.argsort(projector_scores, dim=1, descending=True) #[batch_size, 576, group]
            projector_sort_index = projector_sort_index.transpose(1,2).reshape(-1, similarity.shape[-1]) #[batch_size*group, 576]
            group_token_d = group_token_d.reshape(-1)
            important_indices = []
            for i in range(len(group_token_d)):
                important_indices.append(projector_sort_index[i][:int(group_token_d[i])])

            important_indices = [important_indices[i:i+projector_lengths.shape[-1]] for i in range(0, len(important_indices), projector_lengths.shape[-1])]
            for i in range(len(important_indices)):
                important_indices[i] = torch.cat([torch.tensor([0], device=similarity.device), torch.cat(important_indices[i])+1])
            batch_indices = torch.stack(important_indices)
            batch_indices_expanded = batch_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)) 
            batch_hidden_states = torch.gather(hidden_states, dim=1, index=batch_indices_expanded)

        return batch_hidden_states, batch_indices # torch.Tensor: [batch_size, token_num, hidden_dim] torch.Tensor: [batch_size, dominant_token_num]