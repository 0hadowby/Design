import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .imagebind import ImageBindWrapper
from .open_clip_encoder import OpenCLIPVisionTower
from .hf_vision import HFVisionTower
from .siglip_encoder import SigLipVisionTower
from .mlcd_encoder import MLCDVisionTower, MLCDVisionTowerS2
# from .eva_clip.eva_clip_encoder import EvaClipVisionTower
# from .dev_eva_clip.eva_vit import EvaViTWrapper


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if vision_tower is None:
        raise ValueError("vision_tower is None")

    vt_lower = vision_tower.lower()
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)

    # 关键：先判断 siglip，避免本地绝对路径被误判成 CLIP
    if "siglip" in vt_lower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    if (
        is_absolute_path_exists
        or vt_lower.startswith("openai")
        or vt_lower.startswith("laion")
        or "sharegpt4v" in vt_lower
    ):
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if vt_lower.startswith("hf:"):
        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if vision_tower in ["imagebind_huge"]:
        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)

    if vt_lower.startswith("open_clip_hub"):
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if "mlcd-vit-bigg-patch14" in vt_lower:
        if use_s2:
            return MLCDVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return MLCDVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")

