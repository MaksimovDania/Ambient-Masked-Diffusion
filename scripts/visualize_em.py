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


def create_mnist_dataloaders(
    root,
    batch_size,
    p_missing,
    train_val_split=0.9,
    num_workers=0,
    download=True,
    **kwargs,
):
    """
    Creates dataloader for visualization.
    IMPORTANT: We simulate 'ambient' missing data here by using p_missing.
    """
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

    class AmbientWrapper:
        def __init__(self, loader, p_missing):
            self.loader = loader
            self.iterator = iter(loader)
            self.p_missing = p_missing

        def __iter__(self):
            self.iterator = iter(self.loader)
            return self

        def __next__(self):
            try:
                x, y = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                x, y = next(self.iterator)

            # Generate ambient mask
            if self.p_missing > 0:
                obs_mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p_missing))
            else:
                obs_mask = torch.ones_like(x)

            # Create x_obs (observed part of x, rest is sentinel/ignored)
            # In our conditional_sample, we expect x_obs to be 0/1 where obs_mask=1.
            # Where obs_mask=0, the value doesn't matter (we'll set it to -1 for clarity).
            x_obs = x.clone()
            x_obs[obs_mask == 0] = -1.0

            return {
                "x_clean": x,
                "x_obs": x_obs,
                "obs_mask": obs_mask
            }

    return None, AmbientWrapper(loader, p_missing)


def main():
    parser = argparse.ArgumentParser(description="Visualize EM MDM checkpoint with iterative sampling")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mdm_em.yaml",
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
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--p_missing",
        type=float,
        default=0.5,
        help="Percentage of missing pixels to simulate for conditional sampling",
    )
    parser.add_argument(
        "--cond_steps",
        type=int,
        default=50,
        help="Number of steps for conditional sampling",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["conditional", "unconditional"],
        default="conditional",
        help="Mode: 'conditional' (reconstruction) or 'unconditional' (generation from scratch)",
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
    # Handle both EM checkpoints (mdm_em_iterX.pt) and regular epoch checkpoints
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        epoch_info = checkpoint.get("em_iter", checkpoint.get("epoch", "unknown"))
        print(f"Loaded checkpoint (iter/epoch: {epoch_info})")
    else:
        # Fallback if state dict is top level
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (direct state dict)")

    # -------- Data for Conditional Sampling --------
    if args.mode == "conditional":
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
        
        x_clean = val_batch["x_clean"].to(device)
        x_obs = val_batch["x_obs"].to(device)
        obs_mask = val_batch["obs_mask"].to(device)
        
        print(f"Generating conditional samples (reconstruction) with {args.cond_steps} steps...")
        
        with torch.no_grad():
            x_hat = model.conditional_sample(
                x_obs=x_obs,
                obs_mask=obs_mask,
                num_steps=args.cond_steps,
                device=device
            )

        # -------- Visualization (Conditional) --------
        # Row 1: Ground Truth (Clean)
        # Row 2: Input (Observed only, missing masked as grey/0.5)
        # Row 3: Output (Inpainted/Restored)
        
        # Prepare visual for input:
        # obs_mask=1 -> x_obs (0 or 1)
        # obs_mask=0 -> 0.5 (grey)
        x_in_vis = x_obs.clone()
        x_in_vis[obs_mask == 0] = 0.5
        
        composite = torch.cat([x_clean, x_in_vis, x_hat], dim=0) # [3*B, 1, H, W]
        
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join(
            args.output_dir, 
            f"{ckpt_name}_cond_sample_pmiss{args.p_missing}.png"
        )
        
        vutils.save_image(
            composite,
            out_path,
            nrow=args.num_samples,
            normalize=False
        )
        print(f"Saved reconstruction visualization to {out_path}")

    else:
        # -------- Unconditional Sampling --------
        print(f"Generating unconditional samples (from scratch) with {args.cond_steps} steps...")
        
        # Create dummy inputs (batch size B, all masked)
        B = args.num_samples
        H, W = 28, 28
        
        # All zeros for x_obs (doesn't matter)
        x_obs = torch.zeros(B, 1, H, W, device=device)
        # All zeros for obs_mask (nothing observed -> generate everything)
        obs_mask = torch.zeros(B, 1, H, W, device=device)
        
        with torch.no_grad():
            x_hat = model.conditional_sample(
                x_obs=x_obs,
                obs_mask=obs_mask,
                num_steps=args.cond_steps,
                device=device
            )
            
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join(
            args.output_dir, 
            f"{ckpt_name}_uncond_sample.png"
        )
        
        vutils.save_image(
            x_hat,
            out_path,
            nrow=args.num_samples // 2 if args.num_samples > 8 else args.num_samples,
            normalize=False
        )
        print(f"Saved unconditional samples to {out_path}")

if __name__ == "__main__":
    main()
