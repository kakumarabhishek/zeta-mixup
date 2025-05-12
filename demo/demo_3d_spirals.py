import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path
import sys
from matplotlib.animation import FuncAnimation
import skdim

# Add parent directory to path to allow imports from root.
sys.path.append(str(Path(__file__).parent.parent))

from mixup import mixup_data
from zeta_mixup import zeta_mixup


@dataclass
class Config:
    """
    Configuration parameters for the visualization.
    """

    # Data parameters
    SEED: int = 42
    NUM_SAMPLES: int = 8192
    BATCH_SIZE: int = 32
    NUM_BATCHES: int = NUM_SAMPLES // BATCH_SIZE
    GAMMA_VALS: List[float] = field(default_factory=lambda: [2.0, 4.0, 6.0])

    # Visualization parameters
    MARKER_SIZE: float = 1.5
    COLORS: Dict[str, str] = field(
        default_factory=lambda: {
            "original": "#636363",  # Grey-like
            "mixup": "#43a2ca",  # Blue-like
            "zeta_mixup": "#dd1c77",  # Pink-like
        }
    )

    # Camera parameters
    NUM_FRAMES: int = 180
    START_ELEV: float = 0.0
    END_ELEV: float = 360.0
    AZIM: float = 0.0

    # GIF parameters
    # GIF time = NUM_FRAMES / FPS.
    # 60 fps was smooth but too fast.
    FPS: int = 15
    DPI: int = 100
    SAVE_FORMAT: str = "gif"
    OUTPUT_DIR: Path = Path("./demo_visualizations")


def prepare_data(config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare the 3D spiral dataset and convert to PyTorch tensors.

    Returns:
        Tuple containing:
        - X_torch: Original features (N, 3).
        - X_torch_expanded: Features expanded to 4D (N, 3, 1, 1).
    """
    # Generate 3D spiral data.
    benchmark = skdim.datasets.BenchmarkManifolds(random_state=config.SEED)
    X = benchmark.generate(
        name="M13b_Spiral", n=config.NUM_SAMPLES, dim=3, d=1
    )

    # Convert to torch tensors.
    X_torch = torch.from_numpy(X).float()
    X_torch_expanded = X_torch.unsqueeze(-1).unsqueeze(
        -1
    )  # Shape: (N, 3, 1, 1)

    return X_torch, X_torch_expanded


def process_batch(
    X_batch: torch.Tensor,
    gamma_vals: List[float],
) -> Tuple[torch.Tensor, Dict[float, torch.Tensor]]:
    """
    Process a single batch of data through mixup and zeta-mixup.

    Args:
        X_batch: Input features (batch_size, 3, 1, 1)
        gamma_vals: List of gamma values for zeta-mixup

    Returns:
        Tuple containing:
        - X_mixup: Mixed features from mixup
        - X_zeta: Dictionary of mixed features from zeta-mixup
    """
    # Apply mixup (using dummy labels since we don't need them)
    dummy_y = torch.zeros(X_batch.size(0), dtype=torch.long)
    mixed_x, _, _, _ = mixup_data(X_batch, dummy_y, use_cuda=False)

    # Apply zeta-mixup for each gamma
    X_zeta = {}
    for gamma in gamma_vals:
        X_zeta[gamma], _ = zeta_mixup(
            X_batch,
            dummy_y,
            num_classes=1,
            gamma=gamma,
        )

    return mixed_x, X_zeta


def create_animation(
    X: np.ndarray,
    X_mixup: torch.Tensor,
    X_zeta: Dict[float, torch.Tensor],
    config: Config,
    view_type: str,
) -> None:
    """
    Create and save the animation for a specific view type.

    Args:
        X: Original features
        X_mixup: Mixed features from mixup
        X_zeta: Dictionary of mixed features from zeta-mixup
        config: Configuration parameters
        view_type: Type of view ('original', 'mixup', or 'zeta-mixup')
    """
    # Create figure with specific size
    fig = plt.figure(figsize=(4, 4))

    # Add subplot with minimal margins
    ax = fig.add_subplot(
        111, projection="3d", position=[0.02, 0.02, 0.96, 0.96]
    )

    # Plot original data
    scatter_orig = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=config.COLORS["original"],
        s=config.MARKER_SIZE,
        alpha=0.5,
        label="Original",
    )

    # Plot augmented data based on view type
    if view_type == "mixup":
        scatter_aug = ax.scatter(
            X_mixup[:, 0],
            X_mixup[:, 1],
            X_mixup[:, 2],
            c=config.COLORS["mixup"],
            s=config.MARKER_SIZE,
            alpha=0.5,
            label="Mixup",
        )
        title = "$\it{mixup}$"
    elif view_type.startswith("zeta-mixup"):
        gamma = float(view_type.split("_")[-1])
        scatter_aug = ax.scatter(
            X_zeta[gamma][:, 0],
            X_zeta[gamma][:, 1],
            X_zeta[gamma][:, 2],
            c=config.COLORS["zeta_mixup"],
            s=config.MARKER_SIZE,
            alpha=0.5,
            label="Zeta-mixup",
        )
        title = rf"$\zeta-mixup\ (\gamma = {gamma})$"
    else:  # original
        title = "Original Data"

    # Set title with specific position
    fig.suptitle(title, y=0.98, fontsize=12)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Remove axis labels and ticks
    ax.set_axis_off()

    def update(frame):
        # Calculate elevation for this frame
        elev = config.START_ELEV + (
            config.END_ELEV - config.START_ELEV
        ) * frame / (config.NUM_FRAMES - 1)
        ax.view_init(elev=elev, azim=config.AZIM)
        return (
            scatter_orig,
            scatter_aug if view_type != "original" else scatter_orig,
        )

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=config.NUM_FRAMES,
        interval=1000 / config.FPS,
        blit=True,
    )

    # Create output directory if it doesn't exist
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save animation
    anim.save(
        config.OUTPUT_DIR
        / f"demo_3d_spirals_{view_type}.{config.SAVE_FORMAT}",
        writer="pillow",
        fps=config.FPS,
        dpi=config.DPI,
    )

    plt.close(fig)


def main() -> None:
    """
    Main function to run the visualization.
    """
    # Set random seeds
    config = Config()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Prepare data
    X_torch, X_torch_expanded = prepare_data(config)

    # Initialize output tensors
    X_mixup = torch.zeros_like(X_torch_expanded)
    X_zeta = {
        gamma: torch.zeros_like(X_torch_expanded)
        for gamma in config.GAMMA_VALS
    }

    # Shuffle the data
    shuffled_idx = torch.randperm(X_torch.shape[0])
    X_torch_expanded = X_torch_expanded[shuffled_idx]

    # Process data in batches
    for i in range(config.NUM_BATCHES):
        low_idx = i * config.BATCH_SIZE
        high_idx = (i + 1) * config.BATCH_SIZE

        # Get batch data
        X_batch = X_torch_expanded[low_idx:high_idx]

        # Process batch
        mixed_x, batch_X_zeta = process_batch(X_batch, config.GAMMA_VALS)

        # Store results
        X_mixup[low_idx:high_idx] = mixed_x
        for gamma in config.GAMMA_VALS:
            X_zeta[gamma][low_idx:high_idx] = batch_X_zeta[gamma]

    # Remove extra dimensions for plotting
    X_mixup = X_mixup.squeeze(-1).squeeze(-1)
    for gamma in config.GAMMA_VALS:
        X_zeta[gamma] = X_zeta[gamma].squeeze(-1).squeeze(-1)

    # Create animations for each view
    create_animation(X_torch.numpy(), X_mixup, X_zeta, config, "original")
    create_animation(X_torch.numpy(), X_mixup, X_zeta, config, "mixup")
    for gamma in config.GAMMA_VALS:
        create_animation(
            X_torch.numpy(), X_mixup, X_zeta, config, f"zeta-mixup_{gamma}"
        )


if __name__ == "__main__":
    main()
