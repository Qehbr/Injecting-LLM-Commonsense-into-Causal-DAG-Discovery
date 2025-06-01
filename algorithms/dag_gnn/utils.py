# project/algorithms/dag_gnn/utils.py
import torch
import torch.nn.functional as F
import numpy as np
import math


# --- Main Utility Functions ---

def nll_categorical(preds, target):
    """
    Computes the negative log-likelihood for categorical data using cross-entropy.
    preds: Logits from the model [batch, nodes, num_classes].
    target: Ground truth class indices [batch, nodes].
    """
    if target.dim() == 3 and target.size(2) == 1:
        target = target.squeeze(-1)

    # CrossEntropyLoss expects target of type Long
    target = target.long()

    # Reshape for cross_entropy: preds -> [batch*nodes, num_classes], target -> [batch*nodes]
    num_nodes = preds.size(1)
    preds_flat = preds.view(-1, preds.size(-1))
    target_flat = target.view(-1)

    loss = F.cross_entropy(preds_flat, target_flat, reduction='mean')
    return loss


def nll_gaussian(preds, target, variance, add_const=False):
    """Computes the negative log-likelihood for continuous (Gaussian) data."""
    neg_log_p = variance + torch.div(torch.pow(preds - target, 2), 2. * np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def kl_gaussian_sem(preds):
    """Computes the KL divergence for the SEM."""
    mu = preds
    kl_div = mu * mu
    return 0.5 * kl_div.sum() / preds.size(0)


def stau(w, tau):
    """Soft-thresholding operator."""
    w1 = F.relu(torch.abs(w) - tau)
    return torch.sign(w) * w1


def update_optimizer(optimizer, original_lr, c_A):
    """Dynamically updates learning rate based on the constraint coefficient c_A."""
    MAX_LR, MIN_LR = 1e-2, 1e-4

    # Avoid log(0) or log(negative)
    if c_A <= 1:
        estimated_lr = original_lr
    else:
        estimated_lr = original_lr / math.log10(c_A)

    lr = np.clip(estimated_lr, MIN_LR, MAX_LR)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr


# --- Acyclicity Constraint Functions ---

def _h_A(A, m):
    """Computes the acyclicity constraint function h(A)."""
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


def matrix_poly(matrix, d):
    """Computes the matrix polynomial for the acyclicity constraint."""
    # Ensure eye is on the same device as the matrix
    x = torch.eye(d, device=matrix.device, dtype=matrix.dtype) + torch.div(matrix, d)
    return torch.matrix_power(x, d)


def preprocess_adj_new(adj):
    """Computes I - A^T."""
    # Ensure eye is on the same device as adj
    identity = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    return identity - adj.transpose(0, 1)


def preprocess_adj_new1(adj):
    """Computes (I - A^T)^-1."""
    # Ensure eye is on the same device as adj
    identity = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    return torch.inverse(identity - adj.transpose(0, 1))


# --- Other Utilities ---

def my_softmax(tensor, axis=1):
    """Applies softmax along a specified axis."""
    exp = torch.exp(tensor)
    sum_exp = torch.sum(exp, axis=axis, keepdim=True)
    return exp / (sum_exp + 1e-10)