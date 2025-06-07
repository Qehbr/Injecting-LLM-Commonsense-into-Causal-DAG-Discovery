import torch
import torch.nn.functional as F
import numpy as np
import math


def nll_categorical(preds, target):
    """
    Computes the negative log-likelihood loss for categorical predictions.

    This is equivalent to the cross-entropy loss.

    Parameters
    ----------
    preds : torch.Tensor
        The predicted logits from the model, with shape
        `(batch_size, num_nodes, num_classes)`.
    target : torch.Tensor
        The ground truth class indices, with shape `(batch_size, num_nodes)`
        or `(batch_size, num_nodes, 1)`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the mean cross-entropy loss.
    """
    if target.dim() == 3 and target.size(2) == 1:
        target = target.squeeze(-1)
    target = target.long()
    preds_flat = preds.view(-1, preds.size(-1))
    target_flat = target.view(-1)
    loss = F.cross_entropy(preds_flat, target_flat, reduction='mean')
    return loss


def nll_gaussian(preds, target, variance, add_const=False):
    """
    Computes the negative log-likelihood for a Gaussian distribution.

    Parameters
    ----------
    preds : torch.Tensor
        The predicted means of the distribution.
    target : torch.Tensor
        The ground truth values.
    variance : float or torch.Tensor
        The logarithm of the variance (log-variance).
    add_const : bool, optional
        If True, adds the constant term to the NLL computation.
        Default is False.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the mean negative log-likelihood.
    """
    neg_log_p = variance + torch.div(torch.pow(preds - target, 2), 2. * np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def kl_gaussian_sem(preds):
    """
    Computes the KL divergence of a Gaussian with a standard normal prior.

    This is used in the context of a Structural Equation Model (SEM), assuming
    the prior is N(0, I) and the posterior is N(mu, I). The KL divergence
    simplifies to 0.5 * ||mu||^2.

    Parameters
    ----------
    preds : torch.Tensor
        The means (mu) of the Gaussian distributions, shape `(batch_size, ...)`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the mean KL divergence per batch item.
    """
    mu = preds
    kl_div = mu * mu
    return 0.5 * kl_div.sum() / preds.size(0)


def stau(w, tau):
    """
    Soft-thresholding operator.

    Also known as the shrinkage operator. It is defined as:
    sign(w) * max(|w| - tau, 0).

    Parameters
    ----------
    w : torch.Tensor
        The input tensor.
    tau : float
        The threshold value.

    Returns
    -------
    torch.Tensor
        The tensor after applying the soft-thresholding operation.
    """
    w1 = F.relu(torch.abs(w) - tau)
    return torch.sign(w) * w1


def _h_A(A, m):
    """
    Computes the acyclicity constraint function for an adjacency matrix.

    This function calculates `h(A) = tr(e^(A*A)) - m`, where `A*A` represents
    element-wise multiplication. A value of 0 for `h(A)` implies that the
    graph represented by A is a Directed Acyclic Graph (DAG).

    Parameters
    ----------
    A : torch.Tensor
        The adjacency matrix of the graph.
    m : int
        The number of nodes in the graph (i.e., the dimension of A).

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the value of the constraint function.
    """
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


def matrix_poly(matrix, d):
    """
    Computes a polynomial approximation of the matrix exponential.

    The approximation is `(I + matrix/d)^d`.

    Parameters
    ----------
    matrix : torch.Tensor
        The input square matrix.
    d : int
        The degree of the polynomial approximation.

    Returns
    -------
    torch.Tensor
        The resulting matrix from the polynomial approximation.
    """
    x = torch.eye(d, device=matrix.device, dtype=matrix.dtype) + torch.div(matrix, d)
    return torch.matrix_power(x, d)


def preprocess_adj_new1(adj):
    """
    Pre-processes an adjacency matrix for use in a structural equation model.

    This computes the inverse of `(I - A^T)`, which is a key component in
    recovering the variables from their noise terms in a linear SEM.

    Parameters
    ----------
    adj : torch.Tensor
        The adjacency matrix `A`.

    Returns
    -------
    torch.Tensor
        The pre-processed matrix `(I - A^T)^-1`.
    """
    identity = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    return torch.inverse(identity - adj.transpose(0, 1))
