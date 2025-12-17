from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
import torchvision.utils as vutils

from src.models.mdm_model import MaskedDiffusionModel


@torch.no_grad()
def make_xt_from_xclean(
    model: MaskedDiffusionModel,
    x_clean: Tensor,
    t_value: Optional[int],
    obs_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Build x_t from x_clean via the model's forward masking schedule and a fixed t.

    Parameters
    ----------
    model:
        MaskedDiffusionModel
    x_clean:
        FloatTensor [B,1,H,W] with values in {0,1}
    t_value:
        int or None (if None uses T//2)
    obs_mask:
        Optional mask [B,1,H,W], where 1 = observed, 0 = missing.
        If provided, missing positions are forced to MASK in xt_states.

    Returns
    -------
    xt_states:
        LongTensor [B,1,H,W] in {0,1,MASK}
    xt_vis:
        FloatTensor [B,1,H,W], where MASK positions are set to 0.5 for visualization.
    """
    if x_clean.dim() != 4:
        raise ValueError(
            f"make_xt_from_xclean: expected x_clean [B,1,H,W], got {tuple(x_clean.shape)}"
        )

    B, C, _, _ = x_clean.shape
    if C != 1:
        raise ValueError(
            f"make_xt_from_xclean: expected x_clean with C=1, got C={C}"
        )

    if t_value is None:
        t_value = model.schedule.num_timesteps // 2

    t = torch.full(
        (B,),
        fill_value=int(t_value),
        dtype=torch.long,
        device=x_clean.device,
    )

    xt_states, _ = model.schedule.forward_mask(x_clean, t)  # [B,1,H,W] long

    if obs_mask is not None:
        if obs_mask.shape != x_clean.shape:
            raise ValueError(
                "make_xt_from_xclean: obs_mask shape must match x_clean shape"
            )
        missing = (obs_mask.to(x_clean.device) == 0)
        if missing.any():
            xt_states = xt_states.clone()
            xt_states[missing] = model.mask_token_id

    xt_vis = xt_states.float()
    mask_positions = (xt_states == model.mask_token_id)
    if mask_positions.any():
        xt_vis = xt_vis.clone()
        xt_vis[mask_positions] = 0.5

    return xt_states.long(), xt_vis


@torch.no_grad()
def reconstruct_from_xt(
    model: MaskedDiffusionModel,
    xt_states: Tensor,
    t: Tensor,
) -> Tensor:
    """
    Reconstruct x0 from a given xt using the model's UNet and SUBS-like inference.

    xt_states: LongTensor [B,1,H,W] in {0,1,MASK}
    t:         LongTensor [B]
    """
    xt_one_hot = model._encode_states(xt_states)  # [B,3,H,W]
    logits = model.unet(xt_one_hot, t)           # [B,3,H,W]
    x_recon = model.infer_from_xt(xt_states=xt_states, logits=logits)  # [B,1,H,W]
    return x_recon


@torch.no_grad()
def save_triplet_grid(
    x_top: Tensor,
    x_mid: Tensor,
    x_bottom: Tensor,
    path: str,
    nrow: int,
) -> None:
    """
    Save a 3-row grid: top/mid/bottom batches stacked along batch dimension.
    """
    all_imgs = torch.cat([x_top, x_mid, x_bottom], dim=0)
    vutils.save_image(all_imgs, path, nrow=nrow, normalize=False)

