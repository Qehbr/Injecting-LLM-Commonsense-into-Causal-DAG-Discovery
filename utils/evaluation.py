# project/utils/evaluation.py
import numpy as np


def calculate_custom_shd(learned_cpdag_adj, true_dag_adj):
    """
    Calculates the Structural Hamming Distance (SHD) between a learned CPDAG and a true DAG.

    Args:
        learned_cpdag_adj (np.ndarray): Adjacency matrix of the learned CPDAG.
            - G[i,j] =  1, G[j,i] = -1  => i -> j
            - G[i,j] = -1, G[j,i] =  1  => j -> i (should be consistent with the above)
            - G[i,j] = -1, G[j,i] = -1  => i -- j (undirected)
            - G[i,j] =  0, G[j,i] =  0  => no edge
            - G[i,j] =  1, G[j,i] =  1  => i <-> j (bi-directed, if applicable by algorithm) - COST = 1 if not matching true
        true_dag_adj (np.ndarray): Adjacency matrix of the true DAG.
            - T[i,j] = 1 => i -> j
            - T[i,j] = 0 => no edge from i to j (T[j,i] could be 1 or 0)

    Returns:
        int: The Structural Hamming Distance.
    """
    if learned_cpdag_adj.shape != true_dag_adj.shape:
        raise ValueError("Adjacency matrices must have the same shape for SHD calculation.")

    n_nodes = learned_cpdag_adj.shape[0]
    shd = 0

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Iterate over unique pairs (upper triangle)
            l_ij, l_ji = learned_cpdag_adj[i, j], learned_cpdag_adj[j, i]
            t_ij, t_ji = true_dag_adj[i, j], true_dag_adj[j, i]

            # Determine learned edge type for pair (i, j)
            learned_edge = "none"
            if l_ij == 1 and l_ji == -1:
                learned_edge = "i->j"
            elif l_ij == -1 and l_ji == 1:
                learned_edge = "j->i"
            elif l_ij == -1 and l_ji == -1:
                learned_edge = "i--j"
            elif l_ij == 1 and l_ji == 1:  # Bi-directed in learned graph
                learned_edge = "i<->j"
            elif l_ij == 0 and l_ji == 0:
                learned_edge = "none"
            else:  # Ambiguous or unexpected pattern from algorithm
                # print(f"Warning: Ambiguous learned pattern between {i} and {j}: ({l_ij}, {l_ji}). Costing 1.")
                shd += 1
                continue

            # Determine true edge type for pair (i, j)
            true_edge = "none"
            if t_ij == 1 and t_ji == 0:
                true_edge = "i->j"
            elif t_ij == 0 and t_ji == 1:
                true_edge = "j->i"
            elif t_ij == 1 and t_ji == 1:  # Should not happen in a true DAG
                raise ValueError(f"True graph is not a DAG: bidirectional edge between {i} and {j} ({t_ij}, {t_ji})")
            # (t_ij == 0 and t_ji == 0) is covered by true_edge = "none"

            # Calculate SHD cost for this pair
            cost = 0
            if learned_edge == true_edge:
                cost = 0
            elif learned_edge == "i->j":  # Learned i->j
                if true_edge == "j->i":
                    cost = 1  # Reversed
                else:
                    cost = 1  # Extra or different type (e.g. true was none)
            elif learned_edge == "j->i":  # Learned j->i
                if true_edge == "i->j":
                    cost = 1  # Reversed
                else:
                    cost = 1  # Extra or different type
            elif learned_edge == "i--j":  # Learned i--j
                if true_edge == "i->j" or true_edge == "j->i":
                    cost = 1  # Undirected instead of directed (missing orientation)
                else:
                    cost = 1  # Extra if true_edge is "none"
            elif learned_edge == "i<->j":  # Learned bi-directed
                cost = 1  # Always a mistake if true_edge is not also bi-directed (which it can't be for a DAG)
            elif learned_edge == "none":  # Learned no edge
                if true_edge != "none": cost = 1  # Missing edge
            # If cost is still 0 here, it implies learned_edge == true_edge which is already handled.
            # Any other unhandled learned_edge type already incremented shd.

            shd += cost
    return shd


def evaluate_dag_metrics(
        learned_adj_matrix_strict_dag,  # DAG from CPDAG (only i->j where CPDAG has i->j)
        learned_adj_matrix_raw_cpdag,  # Raw output from algorithm (CPDAG format)
        true_adj_matrix  # True DAG (0 or 1 for i->j)
):
    """
    Evaluates the learned graph against the true DAG using various metrics.
    """
    metrics_results = {}
    n_nodes = true_adj_matrix.shape[0]

    print("  Calculating Custom SHD (learned CPDAG vs true DAG)...")
    try:
        custom_shd_val = calculate_custom_shd(learned_adj_matrix_raw_cpdag, true_adj_matrix)
        print(f"  Structural Hamming Distance (Custom SHD): {custom_shd_val}")
        metrics_results["shd_custom_cpdag_vs_dag"] = custom_shd_val
    except Exception as e:
        print(f"  Could not calculate Custom SHD: {e}")
        metrics_results["shd_custom_cpdag_vs_dag"] = -1  # Indicate error

    # --- Strict F1-score for DIRECTED edges ---
    print("  Calculating Strict F1-score (Manual, for directed edges)...")
    # learned_adj_matrix_strict_dag has 1 where a directed edge i->j is learned.
    # true_adj_matrix has 1 where a true directed edge i->j exists.
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
    print(f"  TP (Strict Directed): {tp_strict}, FP (Strict Directed): {fp_strict}, FN (Strict Directed): {fn_strict}")
    print(f"  Precision (Strict Directed Edges): {precision_strict:.4f}")
    print(f"  Recall (Strict Directed Edges): {recall_strict:.4f}")
    print(f"  F1-Score (Strict Directed Edges): {f1_strict:.4f}")

    # --- Skeleton F1-score for ADJACENCIES ---
    print("  Calculating Skeleton F1-score (Manual, for adjacencies)...")
    # Learned skeleton: an adjacency exists if not (G[i,j]==0 and G[j,i]==0) for the CPDAG
    learned_skeleton_adj = np.zeros_like(learned_adj_matrix_raw_cpdag, dtype=int)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Iterate over unique pairs
            if not (learned_adj_matrix_raw_cpdag[i, j] == 0 and learned_adj_matrix_raw_cpdag[j, i] == 0):
                learned_skeleton_adj[i, j] = 1
                learned_skeleton_adj[j, i] = 1  # Symmetric for skeleton

    # True skeleton: an adjacency exists if T[i,j]==1 or T[j,i]==1
    true_skeleton_adj = np.zeros_like(true_adj_matrix, dtype=int)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Iterate over unique pairs
            if true_adj_matrix[i, j] == 1 or true_adj_matrix[j, i] == 1:
                true_skeleton_adj[i, j] = 1
                true_skeleton_adj[j, i] = 1  # Symmetric for skeleton

    # Calculate TP, FP, FN for skeleton (comparing upper triangles is sufficient due to symmetry)
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
    print(
        f"  TP (Skeleton Adjacency): {tp_skel}, FP (Skeleton Adjacency): {fp_skel}, FN (Skeleton Adjacency): {fn_skel}")
    print(f"  Precision (Skeleton): {precision_skel:.4f}")
    print(f"  Recall (Skeleton): {recall_skel:.4f}")
    print(f"  F1-Score (Skeleton): {f1_skel:.4f}")
    return metrics_results
