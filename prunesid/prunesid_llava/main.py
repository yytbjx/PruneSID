from .clip_encoder import CLIPVisionTower_PruneSID
from .llava_arch import prepare_inputs_labels_for_multimodal_prunesid, encode_images_prunesid, encode_images_prunesid_multi, restore_image_features_sorted

def prunesid_llava(
    model,
    need_token_num=64,
    group_method="dgsm",
    *,
    final_refine="lloyd",
    nms_spatial_radius=None,
    merge_spatial_gamma=0.0,
    merge_cost_normalize=False,
):
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_PruneSID.forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_prunesid
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_prunesid_multi = encode_images_prunesid_multi
        LlavaMetaForCausalLM.encode_images_prunesid = encode_images_prunesid
    vt = model.model.vision_tower
    vt.need_token_num = need_token_num
    vt.group_method = group_method
    vt.final_refine = final_refine
    vt.nms_spatial_radius = nms_spatial_radius
    vt.merge_spatial_gamma = merge_spatial_gamma
    vt.merge_cost_normalize = bool(merge_cost_normalize)
    # Compile DGSM/AISM Numba kernels before timed eval (avoids first-batch stall).
    if group_method in ("dgsm", "dgsm_cdkm", "cdkm", "dgsm_aism", "aism", "cdkm_aism"):
        try:
            from prunesid.clustering.dgsm_cdkm import warmup_dgsm_cdkm
            warmup_dgsm_cdkm(n=64, m=128, k=max(8, int(need_token_num // 4)))
        except Exception:
            pass
    return model
