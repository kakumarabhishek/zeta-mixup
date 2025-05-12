import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Compute the mixup data. Return mixed inputs, pairs of targets, and lambda.
    Taken from: https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py#L17

    Args:
        x (torch.Tensor): The set of inputs from the training batch.
        y (torch.tensor): The set of target labels from the training batch.
        alpha (float, optional): The hyperparameter for mixup. Defaults to 1.0.
        use_cuda (bool, optional): Specify if the GPU should be used. Defaults to True.

    Returns:
        mixed_x (torch.Tensor): The mixed inputs.
        y_a (torch.Tensor): The first set of targets.
        y_b (torch.Tensor): The second set of targets.
        lam (float): The value of lambda used for mixing.
    """

    # Choose the value of lambda.
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Infer batch size from the training batch.
    batch_size = x.size()[0]

    # Generate a shuffled index of the training batch and move to the GPU if specified.
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # Mix the inputs based on the shuffled index and lambda.
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Interpolate the corresponding labels for the inputs
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
