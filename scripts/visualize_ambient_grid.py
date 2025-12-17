"""
Visualization script for ambient MDM checkpoints.

Generates a 3-row grid:
  - Row 1: Input with p_missing (ambient input, missing pixels shown as black)
  - Row 2: Reconstruction from those inputs
  - Row 3: Unconditional generation from pure noise
"""

import argparse
import os
import sys
import torch
import torchvision.utils as vutils
import yaml

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mdm_unet import MDMUNet, MDMUNetConfig
from src.models.mdm_scheduler import MaskSchedule, MaskScheduleConfig
from src.models.mdm_model import MaskedDiffusionModel, MaskedDiffusionConfig
from src.data.mnist import create_mnist_dataloaders
from src.utils.seed import set_seed


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ambient MDM checkpoint with 3-row grid"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mdm_ambient.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualizations",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--p_missing",
        type=float,
        default=0.5,
        help="Probability of missing pixels for input simulation",
    )
    parser.add_argument(
        "--cond_steps",
        type=int,
        default=50,
        help="Number of steps for conditional sampling (reconstruction)",
    )
    parser.add_argument(
        "--uncond_steps",
        type=int,
        default=50,
        help="Number of steps for unconditional sampling",
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
    print("Building model...")
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
    # Handle both EM checkpoints and regular epoch checkpoints
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        epoch_info = checkpoint.get("em_iter", checkpoint.get("epoch", "unknown"))
        print(f"Loaded checkpoint (iter/epoch: {epoch_info})")
    else:
        # Fallback if state dict is top level
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (direct state dict)")

    # -------- Load data with p_missing --------
    print(f"Loading data with p_missing={args.p_missing}...")
    data_cfg = cfg_dict.get("data", {})
    _, val_loader = create_mnist_dataloaders(
        root=data_cfg.get("root", "data"),
        batch_size=args.num_samples,
        p_missing=args.p_missing,
        train_val_split=0.9,
        num_workers=0,
        download=True,
    )

    val_batch = next(iter(val_loader))
    x_clean = val_batch["x_clean"].to(device)  # [B,1,H,W]
    x_obs = val_batch["x_obs"].to(device)  # [B,1,H,W] (with sentinel -1.0 on missing)
    obs_mask = val_batch["obs_mask"].to(device)  # [B,1,H,W]

    # Limit to num_samples
    B = min(args.num_samples, x_clean.shape[0])
    x_clean = x_clean[:B]
    x_obs = x_obs[:B]
    obs_mask = obs_mask[:B]

    # -------- Row 1: Input visualization (x_obs with missing pixels as black) --------
    print("Preparing input visualization...")
    # Create visualization: observed pixels show actual values (0/1), missing pixels show black (0)
    x_obs_vis = torch.zeros_like(x_obs)
    observed = (obs_mask == 1.0)
    missing = (obs_mask == 0.0)
    
    # Copy observed values (0 or 1)
    if observed.any():
        x_obs_vis[observed] = x_obs[observed].clamp(0.0, 1.0)
    
    # Set missing pixels to black (0) - they remain 0 from initialization

    # -------- Row 2: Reconstruction from inputs --------
    print(f"Generating reconstructions with {args.cond_steps} steps...")
    with torch.no_grad():
        x_recon = model.conditional_sample(
            x_obs=x_obs,
            obs_mask=obs_mask,
            num_steps=args.cond_steps,
            device=device,
        ).clamp(0.0, 1.0)

    # -------- Row 3: Unconditional generation from pure noise --------
    print(f"Generating unconditional samples with {args.uncond_steps} steps...")
    with torch.no_grad():
        x_uncond = model.sample(
            num_samples=B,
            device=device,
            num_steps=args.uncond_steps,
        ).clamp(0.0, 1.0)

    # -------- Create 3-row grid --------
    print("Creating 3-row grid visualization...")
    all_imgs = torch.cat([x_obs_vis, x_recon, x_uncond], dim=0)  # [3B,1,H,W]

    # Generate output filename
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    output_path = os.path.join(
        args.output_dir,
        f"{checkpoint_name}_ambient_grid_pmiss{args.p_missing}.png",
    )

    vutils.save_image(
        all_imgs,
        output_path,
        nrow=B,
        normalize=False,
    )

    print(f"Saved 3-row grid visualization to: {output_path}")
    print(f"  Row 1: Input (p_missing={args.p_missing}, missing pixels shown as black)")
    print(f"  Row 2: Reconstruction from inputs")
    print(f"  Row 3: Unconditional generation from pure noise")


if __name__ == "__main__":
    main()
