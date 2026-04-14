import torch

def dice_score(preds, targets):
    """
    expects 2 torch tensor inputs: preds and targets
    returns scalar value output dice coeff of the two
    """

    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    return (2 * (intersection) / (preds.sum() + targets.sum() + 1e-12)).item()