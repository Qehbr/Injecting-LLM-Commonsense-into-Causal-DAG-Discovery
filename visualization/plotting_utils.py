import numpy as np
import networkx as nx

NODE_SIZE = 2500
LABEL_FONT_SIZE = 9
EDGE_WIDTH = 1.5
ARROW_SIZE = 20
FIG_SIZE = (12, 10)
TITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 'small'
DEFAULT_LAYOUT_SEED = 42
DEFAULT_LAYOUT_K_SCALE = 1.5
DEFAULT_LAYOUT_ITERATIONS = 75


def get_shared_layout(adj_matrix_list, variable_names, seed=DEFAULT_LAYOUT_SEED,
                      k_scale=DEFAULT_LAYOUT_K_SCALE, iterations=DEFAULT_LAYOUT_ITERATIONS):
    """
    Computes a consistent node layout for visualizing multiple graphs.

    This function creates a "union graph" containing all edges present in any
    of the provided adjacency matrices. It then computes a layout based on this
    union graph, ensuring that nodes maintain the same position across different
    plots for easier comparison.

    Parameters
    ----------
    adj_matrix_list : list of np.ndarray
        A list of adjacency matrices. Each matrix represents a graph. The
        function will create a union of all adjacencies.
    variable_names : list of str
        The names of the nodes, corresponding to the indices of the matrices.
    seed : int, optional
        The random seed for the spring layout algorithm to ensure
        reproducibility. Default is `DEFAULT_LAYOUT_SEED`.
    k_scale : float, optional
        A scaling factor to adjust the optimal distance between nodes in the
        spring layout. Default is `DEFAULT_LAYOUT_K_SCALE`.
    iterations : int, optional
        The number of iterations for the spring layout algorithm.
        Default is `DEFAULT_LAYOUT_ITERATIONS`.

    Returns
    -------
    dict
        A dictionary where keys are node names and values are their (x, y)
        coordinates. This format is compatible with NetworkX drawing functions.
    """
    n_nodes = len(variable_names)
    if n_nodes == 0:
        return {}

    idx_to_node = {i: name for i, name in enumerate(variable_names)}
    union_graph = nx.Graph()
    union_graph.add_nodes_from(variable_names)

    for adj_matrix in adj_matrix_list:
        if adj_matrix is None or adj_matrix.shape[0] != n_nodes:
            continue

        for r_idx in range(n_nodes):
            for c_idx in range(r_idx + 1, n_nodes):
                u_node_name = idx_to_node.get(r_idx)
                v_node_name = idx_to_node.get(c_idx)
                if u_node_name is None or v_node_name is None:
                    continue
                has_adjacency = False
                if (adj_matrix[r_idx, c_idx] != 0 or adj_matrix[c_idx, r_idx] != 0):
                    has_adjacency = True
                if has_adjacency:
                    union_graph.add_edge(u_node_name, v_node_name)
    pos = None
    if union_graph.number_of_nodes() > 0:
        if union_graph.number_of_edges() > 0:
            if nx.is_connected(union_graph):
                if union_graph.number_of_nodes() < 50:
                    pos = nx.kamada_kawai_layout(union_graph)
                else:
                    pass
            if pos is None:
                k_val = k_scale / np.sqrt(union_graph.number_of_nodes()) if union_graph.number_of_nodes() > 0 else 0.5
                pos = nx.spring_layout(union_graph, seed=seed, k=k_val, iterations=iterations, scale=2.0)
        else:
            pos = nx.circular_layout(union_graph)
    else:
        pos = {}
    return pos
