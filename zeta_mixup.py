import torch
from utils import zeta_mixup_weights

def zeta_mixup(
    x: torch.Tensor, y: torch.Tensor, num_classes: int, gamma: float = 2.8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs zeta-mixup on a batch of training data.

    Args:
        x (torch.Tensor): The set of inputs from the training batch.
        y (torch.tensor): The set of target labels from the training batch.
        num_classes (int): The number of classes in the dataset.
        weights (torch.Tensor): The weights used to perform zeta-mixup.

    Returns:
        x_new (torch.Tensor): The mixed inputs.
        y_new (torch.Tensor): The mixed targets.
    """
    # Generate the weights for the zeta-mixup.
    weights = zeta_mixup_weights(x.shape[0], gamma)
    # print(x.dtype, weights.dtype)

    # Generate a convex combination of the inputs using the weights.
    x_new = torch.einsum("ijkl,pi->pjkl", x, weights)

    # Convert the original labels to one-hot encoding.
    y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()

    # Interpolate the corresponding labels for the inputs using the weights.
    y_new = torch.einsum("pq,qj->pj", weights, y_onehot)

    return x_new, y_new
