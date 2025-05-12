import numpy as np
from numpy.fft import fft, ifft
import torch


def zeta_mixup_weights(batch_size: int, gamma: float = 2.8) -> torch.Tensor:
    """
    Generates the weight matrix to be used with zeta-mixup.

    This function creates a weight matrix where each row represents the weights for mixing
    multiple samples. The weights are derived from a p-series (with parameter gamma) and
    are normalized to sum to 1 for each row. Thanks to StackOverflow user "obchardon" for suggesting this approach.

    The process involves:
    1. Creating randomized indices for each sample.
    2. Calculating shift values to align indices so that the value 1 is on the diagonal.
    3. Applying FFT-based shifting to each row.
    4. Computing p-series weights.
    5. Normalizing weights so that each row sums to 1.

    Args:
        batch_size (int): Number of samples in the training batch.
        gamma (float, optional): The value of gamma in the p-series. Defaults to 2.8.

    Returns:
        torch.Tensor: A weight matrix of shape (batch_size, batch_size) where:
            - Each row sums to 1.
            - Weights are derived from the p-series.

    Example:
        >>> weights = zeta_mixup_weights(batch_size=4, gamma=2.8)
        >>> print(weights.shape)
        torch.Size([4, 4])
        >>> print(torch.allclose(weights.sum(dim=1), torch.ones(4)))
        True
    """
    # Step 1: Create the randomized indices matrix.
    # Create a randomized array with `np.random.shuffle()`. This will ensure that all
    # rows have numbers `1` through `batch_size` but in a randomized order.
    # To do this, for each row, generate a random permutation of numbers 1 to batch_size.
    # We add 1 to get 1-based indices.
    idxs_orig = np.argsort(np.random.rand(batch_size, batch_size), axis=1) + 1

    # Step 2: Find the position of value 1 in each row.
    # We use `np.where()` for this.
    # This will be used to calculate how much each row needs to be shifted so that the
    # value 1 is on the diagonal.
    _, s = np.where(idxs_orig == 1)

    # Step 3: Calculate shift values. We use left-shifting.
    # Again, the shift value is the left shift needed to bring the value 1 to the diagonal.
    # To do this, we subtract the expected position (0 to batch_size-1) from the position
    # of the value 1 in each row.
    # This aligns the value 1 to the diagonal for each row
    s -= np.r_[0 : idxs_orig.shape[0]]

    # Step 4: Apply FFT-based circular shift
    # Use FFT to efficiently shift each row by its calculated amount
    # The complex exponential term creates the phase shift needed for the rotation
    
    # Step 4A: FFT Transformation
    # `fft(idxs_orig, axis=1)` converts each row of the matrix (i.e., each permutation) 
    # into the frequency domain.
    # This is done for each row independently (i.e., axis=1).
    
    # Step 4B: Phase Shift Calculation
    # `np.exp(2 * 1j * np.pi * s[:, None] * np.r_[0 : idxs_orig.shape[1]][None, :]` 
    # creates a matrix of complex exponential terms that will rotate each frequency component.
    # The term `s[:, None]` is the shift value for each row.
    # The term `np.r_[0 : idxs_orig.shape[1]][None, :]` creates a sequence [0, 1, 2, ..., batch_size-1]
    # for the rotation.
    # The result of this multiplication is a phase shift matrix that will rotate each frequency component 
    # by the corresponding amount in the shift vector `s`.
    
    # Step 4C: Inverse FFT Transformation
    # `ifft(..., axis=1)` converts the shifted frequency domain back to the time domain.
    # This is done for each row independently (i.e., axis=1).
    # The multiplication with the phase shift term rotates each frequency component
    # by the corresponding amount in `s`.
    # The result is a matrix where each row has been circularly shifted by the corresponding amount in `s`.
    
    # Step 4D: Final Shifted Indices
    # `np.real(ifft(...))` takes the real part of the inverse FFT.
    # `round()` ensures that we get integer indices.
    idxs_np = np.real(
        ifft(
            fft(idxs_orig, axis=1)
            * np.exp(
                2
                * 1j
                * np.pi
                / idxs_orig.shape[1]
                * s[:, None]
                * np.r_[0 : idxs_orig.shape[1]][None, :]
            ),
            axis=1,
        ).round()
    )

    # Step 5: Convert the shuffled indices to a PyTorch tensor.
    # This prepares the indices for weight calculation.
    idxs_pt = torch.Tensor(idxs_np)

    # Step 6: Calculate p-series weights.
    # For each index n, compute weight as 1/n^gamma.
    # This creates the p-series weights with the specified gamma parameter.
    weights = torch.reciprocal(torch.pow(idxs_pt, gamma).float())

    # Step 7: Normalize weights so that each row sums to 1.
    # This makes the weights suitable for convex combinations.
    weights /= torch.sum(weights, dim=1)

    return weights


class CE_SoftLabels(torch.nn.Module):
    """
    Cross-entropy loss for soft labels.

    This loss function computes the cross-entropy between predictions and soft target labels.
    It is particularly useful for training with label smoothing or when using data augmentation
    methods that generate soft labels (like mixup or zeta-mixup).

    The loss is computed as:
        L = -mean(sum(soft_targets * log_softmax(predictions), dim=1))

    Example:
        >>> criterion = CE_SoftLabels()
        >>> predictions = torch.randn(3, 5)  # batch_size=3, num_classes=5
        >>> soft_targets = torch.softmax(torch.randn(3, 5), dim=1)  # soft labels
        >>> loss = criterion(predictions, soft_targets)
    """

    def __init__(self):
        super(CE_SoftLabels, self).__init__()

    def forward(
        self, predictions: torch.Tensor, soft_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cross-entropy loss between predictions and soft targets.

        Args:
            predictions (torch.Tensor): Model predictions of shape (batch_size, num_classes).
                These should be raw logits (not softmaxed).
            soft_targets (torch.Tensor): Soft target labels of shape (batch_size, num_classes).
                Each row should sum to 1.

        Returns:
            torch.Tensor: The mean cross-entropy loss across the batch.

        Note:
            Taken from:
            https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/19
        """
        logsoftmax = torch.nn.LogSoftmax()
        return torch.mean(torch.sum(-soft_targets * logsoftmax(predictions), 1))
