import numpy as np
import pandas as pd
from sklearn.datasets import make_moons


def make_moons_dataset(
    n_samples: int = 512, noise: float = 0.1
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Make a half-moons dataset with Gaussian noise.

    Args:
        n_samples (int, optional): The number of samples to generate. 
                                   Defaults to 512.
        noise (float, optional): The standard deviation of the Gaussian noise.
                                 Defaults to 0.1.

    Returns:
        X (np.ndarray): The dataset.
        y (np.ndarray): The labels.
        df (pd.DataFrame): The dataset as a pandas DataFrame.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise)
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))

    return X, y, df
