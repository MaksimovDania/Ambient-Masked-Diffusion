# Progress

## Completed
-   [x] **Project Structure**: Set up folders for `src`, `configs`, `scripts`, `outputs`.
-   [x] **Data Pipeline**: MNIST downloading and loading (implied working).
-   [x] **Model Implementation**:
    -   [x] `MaskedDiffusionModel` wrapper.
    -   [x] `MDMUNet` architecture (updated to robust version).
    -   [x] `MaskSchedule` linear schedule.
-   [x] **Training Loop**: `Trainer` class implemented and functional.
-   [x] **Configuration**: YAML config support.
-   [x] **Baseline Run**: First run of `train_baseline_mdm.py` completed (diverged).
-   [x] **Debugging**: Identified cause of divergence (incorrect time embedding normalization).

## In Progress
-   [ ] **Verification**: Re-run baseline training to ensure fix works.

## Todo
-   [ ] **Ambient Implementation**:
    -   [ ] Verify `p_missing` logic in dataset/loader.
    -   [ ] Run experiments with missing data.
-   [ ] **Evaluation**:
    -   [ ] Implement FID or other metrics? (Optional, visual inspection used currently).
-   [ ] **Refinement**:
    -   [ ] Tune hyperparameters (LR, batch size, model size).
    -   [ ] Experiment with different masking schedules (e.g., cosine).
