import numpy as np


def calculate_custom_shd(learned_cpdag_adj, true_dag_adj):
    """
    Calculates a custom Structural Hamming Distance (SHD) between a learned
    CPDAG and a true DAG.

    This function iterates through all pairs of nodes and adds a cost of 1 for
    each missing edge, extra edge, or edge reversal.

    Parameters
    ----------
    learned_cpdag_adj : np.ndarray
        The adjacency matrix of the learned CPDAG. This matrix uses a special
        encoding:
        - `A[i, j] == 1` and `A[j, i] == -1`: a directed edge i -> j
        - `A[i, j] == -1` and `A[j, i] == 1`: a directed edge j -> i
        - Other non-zero combinations represent different edge types or errors.
    true_dag_adj : np.ndarray
        The adjacency matrix of the ground truth DAG, where `A[i, j] = 1`
        indicates a directed edge i -> j.

    Returns
    -------
    int
        The calculated Structural Hamming Distance.

    Raises
    ------
    ValueError
        If the input matrices do not have the same shape, or if the true
        graph is not a valid DAG (i.e., contains bidirectional edges).
    """
    if learned_cpdag_adj.shape != true_dag_adj.shape:
        raise ValueError("Adjacency matrices must have the same shape for SHD calculation.")
    n_nodes = learned_cpdag_adj.shape[0]
    shd = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            l_ij, l_ji = learned_cpdag_adj[i, j], learned_cpdag_adj[j, i]
            t_ij, t_ji = true_dag_adj[i, j], true_dag_adj[j, i]
            learned_edge = "none"
            if l_ij == 1 and l_ji == -1:
                learned_edge = "i->j"
            elif l_ij == -1 and l_ji == 1:
                learned_edge = "j->i"
            elif l_ij == -1 and l_ji == -1:
                learned_edge = "i--j"
            elif l_ij == 1 and l_ji == 1:
                learned_edge = "i<->j"
            elif not (l_ij == 0 and l_ji == 0):
                shd += 1
                continue
            true_edge = "none"
            if t_ij == 1 and t_ji == 0:
                true_edge = "i->j"
            elif t_ij == 0 and t_ji == 1:
                true_edge = "j->i"
            elif t_ij == 1 and t_ji == 1:
                raise ValueError(f"True graph not DAG: ({i},{j})")
            cost = 0
            if learned_edge == true_edge:
                cost = 0
            elif learned_edge == "i->j":
                cost = 1
            elif learned_edge == "j->i":
                cost = 1
            elif learned_edge == "i--j":
                cost = 1
            elif learned_edge == "i<->j":
                cost = 1
            elif learned_edge == "none" and true_edge != "none":
                cost = 1
            shd += cost
    return shd


def evaluate_dag_metrics(learned_adj_matrix_strict_dag, learned_adj_matrix_raw_cpdag, true_adj_matrix):
    """
    Evaluates a learned causal graph against a ground truth DAG.

    This function computes several key metrics:
    1.  A custom Structural Hamming Distance (SHD).
    2.  Precision, Recall, and F1-score for the strictly directed edges.
    3.  Precision, Recall, and F1-score for the undirected graph skeleton.

    Parameters
    ----------
    learned_adj_matrix_strict_dag : np.ndarray
        A standard adjacency matrix (0s and 1s) representing only the
        directed edges of the learned graph.
    learned_adj_matrix_raw_cpdag : np.ndarray
        The adjacency matrix of the learned CPDAG using the special encoding
        required by `calculate_custom_shd`.
    true_adj_matrix : np.ndarray
        A standard adjacency matrix (0s and 1s) of the ground truth DAG.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics. Keys include
        'shd_custom_cpdag_vs_dag', 'precision_strict_directed',
        'recall_strict_directed', 'f1_strict_directed', 'precision_skeleton',
        'recall_skeleton', 'f1_skeleton', and the underlying true positives,
        false positives, and false negatives for both strict and skeleton
        evaluations.
    """
    metrics_results = {}
    n_nodes = true_adj_matrix.shape[0]

    custom_shd_val = calculate_custom_shd(learned_adj_matrix_raw_cpdag, true_adj_matrix)
    metrics_results["shd_custom_cpdag_vs_dag"] = custom_shd_val

    tp_strict = np.sum((learned_adj_matrix_strict_dag == 1) & (true_adj_matrix == 1))
    fp_strict = np.sum((learned_adj_matrix_strict_dag == 1) & (true_adj_matrix == 0))
    fn_strict = np.sum((learned_adj_matrix_strict_dag == 0) & (true_adj_matrix == 1))

    precision_strict = tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) > 0 else 0.0
    recall_strict = tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) > 0 else 0.0
    f1_strict = 2 * (precision_strict * recall_strict) / (precision_strict + recall_strict) if \
        (precision_strict + recall_strict) > 0 else 0.0

    metrics_results["tp_strict_directed"] = tp_strict
    metrics_results["fp_strict_directed"] = fp_strict
    metrics_results["fn_strict_directed"] = fn_strict
    metrics_results["precision_strict_directed"] = precision_strict
    metrics_results["recall_strict_directed"] = recall_strict
    metrics_results["f1_strict_directed"] = f1_strict

    learned_skeleton_adj = np.zeros_like(learned_adj_matrix_raw_cpdag, dtype=int)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if not (learned_adj_matrix_raw_cpdag[i, j] == 0 and learned_adj_matrix_raw_cpdag[j, i] == 0):
                learned_skeleton_adj[i, j] = 1
                learned_skeleton_adj[j, i] = 1
    true_skeleton_adj = np.zeros_like(true_adj_matrix, dtype=int)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if true_adj_matrix[i, j] == 1 or true_adj_matrix[j, i] == 1:
                true_skeleton_adj[i, j] = 1
                true_skeleton_adj[j, i] = 1
    tp_skel, fp_skel, fn_skel = 0, 0, 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            is_adj_learned_skel = learned_skeleton_adj[i, j] == 1
            is_adj_true_skel = true_skeleton_adj[i, j] == 1
            if is_adj_learned_skel and is_adj_true_skel:
                tp_skel += 1
            elif is_adj_learned_skel and not is_adj_true_skel:
                fp_skel += 1
            elif not is_adj_learned_skel and is_adj_true_skel:
                fn_skel += 1

    precision_skel = tp_skel / (tp_skel + fp_skel) if (tp_skel + fp_skel) > 0 else 0.0
    recall_skel = tp_skel / (tp_skel + fn_skel) if (tp_skel + fn_skel) > 0 else 0.0
    f1_skel = 2 * (precision_skel * recall_skel) / (precision_skel + recall_skel) if \
        (precision_skel + recall_skel) > 0 else 0.0

    metrics_results.update({
        "tp_skeleton": tp_skel, "fp_skeleton": fp_skel, "fn_skeleton": fn_skel,
        "precision_skeleton": precision_skel, "recall_skeleton": recall_skel, "f1_skeleton": f1_skel
    })
    return metrics_results
