import torch
import torchmetrics
from torchmetrics.classification import BinaryJaccardIndex, BinaryAUROC

def dice_score(preds, targets):
    """
    expects 2 torch tensor inputs: preds and targets
    preds expeted binary prediction
    returns scalar value output dice coeff of the two
    """
    preds = preds.float()
    targets = targets.float()
    print(preds.shape, targets.shape)  # add this
    intersection = (preds * targets).sum()
    return (2 * (intersection) / (preds.sum() + targets.sum() + 1e-12)).item()

def jaccard_index(preds, targets):
    """"
    wrapper for torch binaryjaccerdinex, return jaccard of preds and 
    targets (two tensors). Assumes preds is binary prediction
    """
    preds = preds.float()
    targets = targets.float()
    metric = BinaryJaccardIndex().to(preds.device)
    return metric(preds, targets)

def auc(preds, targets):
    """"
    Returns auc given tensor inputs of labels
    and probability values (preds). Expects preds is sigmoid value
    (not logits or binary prediction)
    """
    binary_auroc = BinaryAUROC().to(preds.device)
    return binary_auroc(preds, targets)

def balanced_acc(preds, targets):
    """"
    return balanced acc of preds and 
    targets (two tensors). Assumes preds is binary prediction
    """
    preds = preds.float()
    targets = targets.float()
    num_tp = ((preds == 1) & (targets == 1)).sum().float()
    num_tn = ((preds == 0) & (targets == 0)).sum().float()
    num_p = (targets == 1).sum().float()
    num_n = (targets == 0).sum().float()
    return ((num_tp / num_p) + (num_tn / num_n)) / 2


    



