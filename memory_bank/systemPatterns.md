# System Patterns

## Architecture
The system follows a modular design typical for PyTorch research projects.

### 1. Model (`src/models/`)
-   **`MaskedDiffusionModel`** (`mdm_model.py`): The top-level class.
    -   Manages the diffusion process (forward masking) and the reverse process (denoising/sampling).
    -   **Forward Process**: Defined by `MaskSchedule`. Masks pixels based on a linear schedule.
    -   **Reverse Process**: Uses `MDMUNet` to predict the original value ({0, 1}) of masked pixels.
    -   **Loss**:
        -   **Baseline**: Cross-Entropy loss on pixels masked by the forward process.
        -   **Ambient**: Cross-Entropy loss *only* on observed pixels (`obs_mask == 1`). Missing pixels are treated as permanently masked and ignored in the loss.
-   **`MDMUNet`** (`mdm_unet.py`): A U-Net architecture adapted for 28x28 images.
    -   Inputs: One-hot encoded image (channels for 0, 1, MASK) + Time Embedding.
    -   Outputs: Logits for classes {0, 1}.
    -   **Time Embedding**: Standard sinusoidal positional encoding + MLP.
-   **`MaskSchedule`** (`mdm_scheduler.py`):
    -   Defines $\alpha_t$, the probability of a pixel remaining unmasked at time $t$.
    -   Linear decay schedule: $\alpha_t = 1 - t/T$.

### 2. Data (`src/data/`)
-   **`MNISTDataset`**: Custom wrapper.
-   **Preprocessing**: Images are binarized (threshold 0.5) to {0, 1}.
-   **Ambient Simulation**: `create_mnist_dataloaders` generates `obs_mask` based on `p_missing`.

### 3. Training (`src/training/` & `scripts/`)
-   **`train_baseline_mdm.py`**: Uses `Trainer` class for clean data.
-   **`train__mdm_ambient.py`**: Standalone script for ambient training.
    -   Handles `obs_mask`.
    -   Saves visualizations (clean, masked, reconstruction) each epoch.

### 4. Visualization (`scripts/visualize_checkpoint.py`)
-   Standalone script to evaluate trained models.
-   Generates unconditional samples.
-   Generates reconstructions from partially masked states (e.g., $t=T/2$) using SUBS inference (argmax).

## Design Decisions
-   **Discrete Diffusion**: Using a "masking" diffusion instead of Gaussian noise. States are discrete {0, 1, MASK}.
-   **Ambient Loss**: The core idea of Ambient Diffusion is implemented by masking the loss function to ignore unobserved pixels.
-   **One-Hot Input**: The UNet takes a 3-channel input (one-hot for 0, 1, MASK) to explicitly represent the "unknown" state.
