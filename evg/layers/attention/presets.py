from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftAttentionPreset:
    name: str
    latent_h: int
    latent_w: int
    num_frames: int
    text_len: int
    pool_h: int
    pool_w: int
    sparsity_ratio: float
    num_heads: int
    head_dim: int

    @property
    def visual_len(self) -> int:
        return self.num_frames * self.latent_h * self.latent_w


PRESETS: dict[str, DraftAttentionPreset] = {
    "wan2.2-480p": DraftAttentionPreset(
        name="wan2.2-480p",
        latent_h=32,
        latent_w=48,
        num_frames=21,
        text_len=0,
        pool_h=8,
        pool_w=16,
        sparsity_ratio=0.75,
        num_heads=40,
        head_dim=128,
    ),
    "wan2.2-720p": DraftAttentionPreset(
        name="wan2.2-720p",
        latent_h=48,
        latent_w=80,
        num_frames=21,
        text_len=0,
        pool_h=8,
        pool_w=16,
        sparsity_ratio=0.75,
        num_heads=40,
        head_dim=128,
    ),
    "hunyuanvideo-1.5-480p": DraftAttentionPreset(
        name="hunyuanvideo-1.5-480p",
        latent_h=32,
        latent_w=48,
        num_frames=33,
        text_len=256,
        pool_h=8,
        pool_w=16,
        sparsity_ratio=0.9,
        num_heads=24,
        head_dim=128,
    ),
    "hunyuanvideo-1.5-720p": DraftAttentionPreset(
        name="hunyuanvideo-1.5-720p",
        latent_h=48,
        latent_w=80,
        num_frames=33,
        text_len=256,
        pool_h=8,
        pool_w=16,
        sparsity_ratio=0.9,
        num_heads=24,
        head_dim=128,
    ),
}


def get_preset(name: str, full_shape: bool = False) -> DraftAttentionPreset:
    try:
        preset = PRESETS[name]
    except KeyError as exc:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown Draft Attention preset '{name}'. Known presets: {known}") from exc

    if full_shape:
        return preset

    return DraftAttentionPreset(
        name=f"{preset.name}-toy",
        latent_h=8,
        latent_w=16,
        num_frames=2,
        text_len=min(preset.text_len, 16),
        pool_h=2,
        pool_w=4,
        sparsity_ratio=preset.sparsity_ratio,
        num_heads=min(preset.num_heads, 4),
        head_dim=min(preset.head_dim, 32),
    )
