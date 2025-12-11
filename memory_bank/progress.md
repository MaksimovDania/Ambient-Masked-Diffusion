# Progress

## Completed
-   [x] **Project Structure**: Set up folders for `src`, `configs`, `scripts`, `outputs`.
-   [x] **Data Pipeline**: MNIST downloading and loading.
-   [x] **Model Implementation**:
    -   [x] `MaskedDiffusionModel` wrapper with ambient loss support.
    -   [x] `MDMUNet` architecture (fixed time embeddings).
    -   [x] `MaskSchedule` linear schedule.
-   [x] **Training Scripts**:
    -   [x] `train_baseline_mdm.py` (baseline).
    -   [x] `train__mdm_ambient.py` (ambient).
-   [x] **Utilities**:
    -   [x] `visualize_checkpoint.py`.
-   [x] **Configuration**: YAML config support.
-   [x] **Debugging**: Resolved time embedding divergence issue.
-   [x] **Ambient Experiments**:
    -   [x] Run `train__mdm_ambient.py` with `p_missing=0.5`.
    -   [x] Verify reconstruction quality on missing data (confirmed high quality at Epoch 4).

## In Progress
-   [ ] **Evaluation**:
    -   [ ] Wait for final training epoch.
    -   [ ] Final visual inspection.

## Todo
-   [ ] **Refinement**:
    -   [ ] Tune hyperparameters (LR, batch size, model size).
    -   [ ] Experiment with different masking schedules (e.g., cosine).
