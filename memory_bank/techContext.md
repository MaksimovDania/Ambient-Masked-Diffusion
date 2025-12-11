# Tech Context

## Technology Stack
-   **Language**: Python 3.11
-   **Deep Learning Framework**: PyTorch (compatible with CPU and MPS on macOS).
-   **Data Processing**: standard PyTorch `DataLoader`, `torchvision` for MNIST.
-   **Configuration**: `PyYAML`.
-   **Logging**: Standard Python `logging`.

## Development Environment
-   **OS**: macOS (Darwin 25.1.0).
-   **Shell**: zsh.
-   **Virtual Environment**: `venv` (located in project root).
-   **Hardware**: Apple Silicon (MPS acceleration available).

## Constraints & Dependencies
-   **Compute**: Local training on Mac. Small models (UNet) and datasets (MNIST) are chosen to fit within these constraints.
-   **Data**: MNIST dataset (downloaded locally to `data/`).
-   **External Libraries**:
    -   `torch`, `torchvision`
    -   `pyyaml`
    -   `numpy` (implied)

