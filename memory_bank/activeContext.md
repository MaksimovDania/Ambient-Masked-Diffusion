# Active Context

## Current Status
-   **Project Stage**: Baseline Experimentation.
-   **Active Experiment**: `mdm_mnist_baseline` (Clean MNIST, no missing data during training).
-   **Latest Run**: `2025-12-11 02:50:18` (approx).
    -   Training ran for 3-4 epochs.
    -   Checkpoints saved: `mdm_epoch_1.pt` to `mdm_epoch_3.pt`, `mdm_best.pt`.
    -   Sampling script execution confirmed.

## Recent Issues
-   **Diverging Loss** (Resolved?): The training loss was increasing significantly with each epoch in previous runs.
    -   **Diagnosis**: The issue was identified in the `SinusoidalPosEmb` class in `src/models/mdm_unet.py`. The line `t = t / (t.max().clamp(min=1.0))` normalized timesteps by the *batch maximum*, causing the same timestep to have different embedding values across batches. This inconsistent signal prevented the model from learning the noise level, leading to conflicting gradients and divergence.
    -   **Fix**: The `mdm_unet.py` file has been updated (now ~510 lines) with a correct, standard implementation of `TimeEmbedding` that does not normalize by batch max.

## Next Steps
1.  **Verify Fix**: Run the training again with the new `mdm_unet.py` to confirm convergence.
2.  **Implement Ambient Mode**: Once baseline works, test with `p_missing > 0`.
