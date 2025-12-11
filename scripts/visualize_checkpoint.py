import argparse
import os
import sys
import torch
import torchvision.utils as vutils
import yaml
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mdm_unet import MDMUNet, MDMUNetConfig
from src.models.mdm_scheduler import MaskSchedule, MaskScheduleConfig
from src.models.mdm_model import MaskedDiffusionModel, MaskedDiffusionConfig
from src.utils.seed import set_seed


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Local implementation of dataloader since src.data is missing/unfound
def create_mnist_dataloaders(
    root,
    batch_size,
    p_missing,
    train_val_split=0.9,
    num_workers=0,
    download=True,
    **kwargs,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),
        ]
    )

    # Use test set for visualization
    dataset = datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=download,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    class DictWrapper:
        def __init__(self, loader):
            self.loader = loader
            self.iterator = iter(loader)

        def __iter__(self):
            self.iterator = iter(self.loader)
            return self

        def __next__(self):
            try:
                x, y = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                x, y = next(self.iterator)

            # dataset returns (x, y)
            return {"x_clean": x}

    return None, DictWrapper(loader)


def subs_argmax(model, xt, logits):
    """
    SUBS-инференс (argmax) по логитам и текущему состоянию xt.
    xt:     [B,1,H,W] в {0,1,MASK}
    logits: [B,C,H,W]
    """
    mask_id = model.mask_token_id
    neg_inf = model.cfg.neg_infinity

    B, C, H, W = logits.shape

    logits = logits.clone()
    logits[:, mask_id, :, :] += neg_inf

    # Normalize -> log_probs
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # Force unmasked pixels to stay as they are
    unmasked = xt != mask_id  # [B,1,H,W]
    if unmasked.any():
        target = xt.squeeze(1).long()  # [B,H,W]

        # создаём тензор с -inf везде, кроме 0 в таргет-классе
        forced = torch.full_like(log_probs, fill_value=neg_inf)  # [B,C,H,W]

        forced_flat = (
            forced.permute(0, 2, 3, 1).contiguous().view(-1, C)
        )  # [B*H*W,C]
        target_flat = target.view(-1)
        row_idx = torch.arange(forced_flat.shape[0], device=logits.device)
        forced_flat[row_idx, target_flat] = 0.0

        forced = forced_flat.view(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]

        unmasked_broadcast = unmasked.expand_as(log_probs)
        log_probs[unmasked_broadcast] = forced[unmasked_broadcast]

    probs = log_probs.exp()
    # Argmax over class 0 and 1
    return probs[:, :2, :, :].argmax(dim=1, keepdim=True).float()


def main():
    parser = argparse.ArgumentParser(description="Visualize MDM checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mdm_baseline.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualizations",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples / reconstructions to visualize",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading config from {args.config}")
    cfg_dict = load_config(args.config)

    set_seed(42)

    # Detect device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # -------- Build model --------
    model_cfg = cfg_dict.get("model", {})

    unet_config = MDMUNetConfig(
        image_channels=model_cfg.get("image_channels", 3),
        base_channels=model_cfg.get("base_channels", 64),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 2])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        time_emb_dim_mult=model_cfg.get("time_emb_dim_mult", 4),
    )
    unet = MDMUNet(unet_config)

    schedule_config = MaskScheduleConfig(
        num_timesteps=model_cfg.get("num_timesteps", 100),
        alpha_min=model_cfg.get("alpha_min", 0.01),
        schedule_type=model_cfg.get("schedule_type", "linear"),
    )
    schedule = MaskSchedule(schedule_config)

    mdm_cfg = MaskedDiffusionConfig(
        neg_infinity=cfg_dict.get("model", {}).get("neg_infinity", -1e9)
    )
    model = MaskedDiffusionModel(
        unet=unet,
        schedule=schedule,
        cfg=mdm_cfg,
    )
    model.to(device)
    model.eval()

    # -------- Load checkpoint --------
    print(f"Loading checkpoint from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found!")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    epoch = checkpoint.get("epoch", -1)
    print(f"Loaded checkpoint from epoch {epoch}")

    # -------- 1. Unconditional samples --------
    print("Generating unconditional samples...")
    samples = model.sample(num_samples=args.num_samples, device=device)
    samples = samples.clamp(0.0, 1.0)  # [B,1,28,28]

    # -------- 2. Reconstructions --------
    print("Generating reconstructions...")

    data_cfg = cfg_dict.get("data", {})
    _, val_loader = create_mnist_dataloaders(
        root=data_cfg.get("root", "data"),
        batch_size=args.num_samples,
        p_missing=0.0,
        train_val_split=0.9,
        num_workers=0,
        download=True,
    )

    val_batch = next(iter(val_loader))
    x_clean = val_batch["x_clean"].to(device)  # [B,1,28,28]

    # Reconstruct from t = T/2 (partial mask)
    t_mid_val = model.schedule.num_timesteps // 2
    t_mid = torch.full(
        (x_clean.shape[0],),
        t_mid_val,
        device=device,
        dtype=torch.long,
    )

    # Forward mask
    xt_mid, mask_mid = model.schedule.forward_mask(x_clean, t_mid)  # [B,1,H,W], [B,1,H,W]

    # Encode
    xt_one_hot = model._encode_states(xt_mid)  # [B,3,H,W]
    logits = model.unet(xt_one_hot, t_mid)     # [B,3,H,W]

    # Inference (SUBS argmax)
    recon_mid = subs_argmax(model, xt_mid, logits)  # [B,1,H,W]

    # Prepare masked visualization (0.5 для MASK)
    xt_vis = xt_mid.float()
    xt_vis[xt_mid == model.mask_token_id] = 0.5

    # -------- 3. Single composite image --------
    # Стек: 1-я строка — unconditional samples,
    #       2-я — clean,
    #       3-я — masked (xt),
    #       4-я — reconstructions.
    # Всего 4 * B изображений, nrow = B.

    B = args.num_samples
    # На случай, если из DataLoader пришло меньше изображений (крайний батч)
    B_actual = min(B, x_clean.shape[0], samples.shape[0])
    samples = samples[:B_actual]
    x_clean = x_clean[:B_actual]
    xt_vis = xt_vis[:B_actual]
    recon_mid = recon_mid[:B_actual]

    composite = torch.cat(
        [samples, x_clean, xt_vis, recon_mid],
        dim=0,
    )  # [4B,1,H,W]

    composite_path = os.path.join(
        args.output_dir,
        f"epoch{epoch:03d}_summary_t{t_mid_val}.png",
    )
    vutils.save_image(
        composite,
        composite_path,
        nrow=B_actual,
        normalize=False,
    )

    print(f"Saved combined visualization to {composite_path}")


if __name__ == "__main__":
    main()