# Active Context

## Current Status
-   **Project Stage**: Ambient Experimentation.
-   **Active Experiment**: `mdm_mnist_ambient` (50% missing data).
-   **Latest Updates**:
    -   **Ambient Training**: Currently running (`train__mdm_ambient.py`).
    -   **Loss**: Converging well. Epoch 4 AvgLoss ~0.1325, ValLoss ~0.1317.
    -   **Reconstruction**: Verified at Epoch 4. The model successfully reconstructs digits from heavily masked inputs (simulating $t=T/2$ + ambient missing pixels). The visual quality is high, with sharp and accurate digits.
    -   **Implementation**: Validated correct `compute_loss` logic (intersection of diffusion mask and observation mask).

## Recent Issues
-   **Diverging Loss** (Resolved): The fix in `mdm_unet.py` handles time embeddings correctly.

## Next Steps
1.  **Complete Training**: Let the ambient training run finish (10 epochs planned).
2.  **Evaluation**:
    -   Visualize final results.
    -   Consider quantitative metrics if needed.
