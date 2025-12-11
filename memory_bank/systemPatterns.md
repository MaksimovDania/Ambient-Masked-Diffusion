# System Patterns

## Architecture
The system follows a modular design typical for PyTorch research projects.

### 1. Model (`src/models/`)
-   **`MaskedDiffusionModel`** (`mdm_model.py`): The top-level class.
    -   Manages the diffusion process (forward masking) and the reverse process (denoising/sampling).
    -   **Forward Process**: Defined by `MaskSchedule`. Masks pixels based on a linear schedule.
    -   **Reverse Process**: Uses `MDMUNet` to predict the original value ({0, 1}) of masked pixels.
    -   **Loss**: Cross-Entropy loss calculated *only* on the masked positions (MLM-style).
-   **`MDMUNet`** (`mdm_unet.py`): A U-Net architecture adapted for 28x28 images.
    -   Inputs: One-hot encoded image (channels for 0, 1, MASK) + Time Embedding.
    -   Outputs: Logits for classes {0, 1}.
    -   Components: `ResidualBlock`, `TimeEmbedding`, `SinusoidalPosEmb`.
-   **`MaskSchedule`** (`mdm_scheduler.py`):
    -   Defines $\alpha_t$, the probability of a pixel remaining unmasked at time $t$.
    -   Linear decay schedule: $\alpha_t = 1 - t/T$.

### 2. Data (`src/data/`)
-   **`MNISTDataset`**: Custom wrapper (likely in `mnist.py`, referenced in `trainer.py` indirectly via loader).
-   **Preprocessing**: Images are binarized (threshold 0.5) to {0, 1}.
-   **Augmentation**: Current config shows no complex augmentation, just binarization.

### 3. Training (`src/training/`)
-   **`Trainer`** (`trainer.py`): Handles the training loop.
    -   Iterates through epochs and batches.
    -   Computes loss using `model.compute_loss`.
    -   Backpropagates and updates weights.
    -   Evaluates on validation set.
    -   Saves checkpoints (`best` and per-epoch).

### 4. Configuration (`configs/`)
-   Experiments are defined in `.yaml` files.
-   Key parameters: `p_missing` (for ambient setting), `batch_size`, `lr`, `num_timesteps`, `channel_mults`.

## Design Decisions
-   **Discrete Diffusion**: Using a "masking" diffusion instead of Gaussian noise. States are discrete {0, 1, MASK}.
-   **MLM Objective**: Loss is only computed on tokens that were masked, enforcing the model to learn in-painting/unmasking.
-   **One-Hot Input**: The UNet takes a 3-channel input (one-hot for 0, 1, MASK) to explicitly represent the "unknown" state.

