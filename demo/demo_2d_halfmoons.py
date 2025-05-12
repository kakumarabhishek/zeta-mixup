import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path to allow imports from root.
sys.path.append(str(Path(__file__).parent.parent))

from demo_utils import make_moons_dataset
from mixup import mixup_data
from zeta_mixup import zeta_mixup


@dataclass
class Config:
    """
    Configuration parameters for the visualization.
    """

    SEED: int = 42
    NUM_SAMPLES: int = 512
    BATCH_SIZE: int = 32
    NUM_BATCHES: int = NUM_SAMPLES // BATCH_SIZE
    GAMMA_VALS: List[float] = (3.0, 2.8, 2.6, 2.4, 2.2, 2.0)
    FIGSIZE: Tuple[int, int] = (20, 2)
    CMAP: str = "coolwarm_r"
    SHOW_COLORBARS: bool = True
    SAVE_FORMAT: str = "png"
    OUTPUT_DIR: Path = Path("./demo_visualizations")


def prepare_data(
    config: Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare the half-moons dataset and convert to appropriate tensor formats.

    Returns:
        Tuple containing:
        - X_torch: Original features (N, 2)
        - y_torch: Original labels (N,)
        - X_torch_expanded: Features expanded to 4D (N, 2, 1, 1)
        - y_torch_expanded: Labels expanded to 2D (N, 1)
    """
    X, y, _ = make_moons_dataset(config.NUM_SAMPLES)
    X_torch = torch.from_numpy(X).float()  # Shape: (N, 2)
    y_torch = torch.from_numpy(y).long()  # Shape: (N,). Needs to be long for
    # one-hot encoding.

    # Expand the features and labels to 4D and 2D respectively.
    X_torch_expanded = X_torch.unsqueeze(-1).unsqueeze(
        -1
    )  # Shape: (N, 2, 1, 1).
    y_torch_expanded = y_torch.unsqueeze(-1)  # Shape: (N, 1).

    return X_torch, y_torch, X_torch_expanded, y_torch_expanded


def create_soft_labels_mixup(
    y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    """
    Create soft labels by interpolating between two one-hot encoded labels.

    Args:
        y_a: First set of labels (N, 1)
        y_b: Second set of labels (N, 1)
        lam: Mixing coefficient

    Returns:
        Soft labels (N, 2) where each row sums to 1
    """
    # Convert the labels to one-hot encoded labels.
    # Need to convert to float for label interpolation.
    y_a_onehot = torch.nn.functional.one_hot(
        y_a.squeeze(-1), num_classes=2
    ).float()  # Shape: (N, 2).
    y_b_onehot = torch.nn.functional.one_hot(
        y_b.squeeze(-1), num_classes=2
    ).float()  # Shape: (N, 2).

    # Interpolate between the two one-hot encoded labels to get the final
    # soft labels.
    return lam * y_a_onehot + (1 - lam) * y_b_onehot


def process_batch(
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    gamma_vals: List[float],
    use_cuda: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[float, torch.Tensor],
    Dict[float, torch.Tensor],
]:
    """
    Process a single batch of data through mixup and zeta-mixup.

    Args:
        X_batch: Input features (batch_size, 2, 1, 1)
        y_batch: Input labels (batch_size, 1)
        gamma_vals: List of gamma values for zeta-mixup
        use_cuda: Whether to use GPU

    Returns:
        Tuple containing:
        - X_mixup: Mixed features from mixup
        - y_mixup: Soft labels from mixup
        - X_zeta: Dictionary of mixed features from zeta-mixup
        - y_zeta: Dictionary of soft labels from zeta-mixup
    """
    # Apply mixup to the batch.
    mixed_x, y_a, y_b, lam = mixup_data(X_batch, y_batch, use_cuda=use_cuda)

    # Create soft labels from the mixup.
    y_mixup = create_soft_labels_mixup(y_a, y_b, lam)

    # Apply zeta-mixup to the batch for each gamma.
    X_zeta = {}
    y_zeta = {}
    for gamma in gamma_vals:
        # Apply zeta-mixup to the batch.
        X_zeta[gamma], y_zeta[gamma] = zeta_mixup(
            X_batch,
            y_batch.squeeze(-1),
            num_classes=2,
            gamma=gamma,
        )

    return mixed_x, y_mixup, X_zeta, y_zeta


def create_plot(
    X: np.ndarray,
    y: np.ndarray,
    X_mixup: torch.Tensor,
    y_mixup: torch.Tensor,
    X_zeta: Dict[float, torch.Tensor],
    y_zeta: Dict[float, torch.Tensor],
    config: Config,
) -> None:
    """
    Create and save the visualization plot.

    Args:
        X: Original features
        y: Original labels
        X_mixup: Mixed features from mixup
        y_mixup: Soft labels from mixup
        X_zeta: Dictionary of mixed features from zeta-mixup
        y_zeta: Dictionary of soft labels from zeta-mixup
        config: Configuration parameters
    """
    fig, axs = plt.subplots(
        1, 1 + len(config.GAMMA_VALS) + 1, figsize=config.FIGSIZE
    )

    # Original data.
    axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap=config.CMAP)
    axs[0].set_title("Original Data")
    axs[0].set_xlabel("X1")
    axs[0].set_ylabel("X2")

    # zeta-mixup outputs for each gamma.
    for i, gamma in enumerate(config.GAMMA_VALS):
        axs[1 + i].scatter(
            X_zeta[gamma][:, 0],
            X_zeta[gamma][:, 1],
            c=y_zeta[gamma],
            cmap=config.CMAP,
        )
        axs[1 + i].set_title(rf"$\zeta-mixup\ (\gamma = {gamma})$")
        axs[1 + i].set_xlabel("X1")
        axs[1 + i].set_ylabel("X2")

    # mixup outputs (last subplot).
    scatter = axs[-1].scatter(
        X_mixup[:, 0], X_mixup[:, 1], c=y_mixup, cmap=config.CMAP
    )
    axs[-1].set_title("$\it{mixup}$")
    axs[-1].set_xlabel("X1")
    axs[-1].set_ylabel("X2")

    # Add colorbar to the mixup plot (last subplot).
    if config.SHOW_COLORBARS:
        plt.colorbar(
            scatter,
            ax=axs[-1],
            label="Class 1 Probability",
            ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        )

    for ax in axs:
        ax.set_axis_off()

    plt.tight_layout()

    # Create output directory if it doesn't exist.
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the plot.
    plt.savefig(config.OUTPUT_DIR / f"demo_2d_halfmoons.{config.SAVE_FORMAT}")


def main() -> None:
    """
    Main function to run the visualization.
    """
    # Set random seeds
    config = Config()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Prepare the data.
    X_torch, y_torch, X_torch_expanded, y_torch_expanded = prepare_data(config)

    # Initialize the output tensors.
    X_mixup = torch.zeros_like(X_torch_expanded)
    y_mixup = torch.zeros((X_torch.shape[0], 2))
    X_zeta = {
        gamma: torch.zeros_like(X_torch_expanded)
        for gamma in config.GAMMA_VALS
    }
    y_zeta = {
        gamma: torch.zeros((X_torch.shape[0], 2))
        for gamma in config.GAMMA_VALS
    }

    # Shuffle the data.
    shuffled_idx = torch.randperm(X_torch.shape[0])
    X_torch_expanded = X_torch_expanded[shuffled_idx]
    y_torch_expanded = y_torch_expanded[shuffled_idx]

    # Process the data in batches.
    for i in range(config.NUM_BATCHES):
        low_idx = i * config.BATCH_SIZE
        high_idx = (i + 1) * config.BATCH_SIZE

        # Get the batch data.
        X_batch = X_torch_expanded[low_idx:high_idx]
        y_batch = y_torch_expanded[low_idx:high_idx]

        # Process each batch.
        mixed_x, batch_y_mixup, batch_X_zeta, batch_y_zeta = process_batch(
            X_batch, y_batch, config.GAMMA_VALS
        )

        # Store the results.
        X_mixup[low_idx:high_idx] = mixed_x
        y_mixup[low_idx:high_idx] = batch_y_mixup
        for gamma in config.GAMMA_VALS:
            X_zeta[gamma][low_idx:high_idx] = batch_X_zeta[gamma]
            y_zeta[gamma][low_idx:high_idx] = batch_y_zeta[gamma]

    # Remove extra dimensions for plotting.
    X_mixup = X_mixup.squeeze(-1).squeeze(-1)
    y_mixup = y_mixup[:, 1]
    for gamma in config.GAMMA_VALS:
        X_zeta[gamma] = X_zeta[gamma].squeeze(-1).squeeze(-1)
        y_zeta[gamma] = y_zeta[gamma][:, 1]

    # Create and save the plot.
    create_plot(
        X_torch.numpy(),
        y_torch.numpy(),
        X_mixup,
        y_mixup,
        X_zeta,
        y_zeta,
        config,
    )


if __name__ == "__main__":
    main()
