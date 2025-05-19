# project/visualization/plotting_utils.py
import numpy as np
import networkx as nx

# --- Visualization Constants ---
NODE_SIZE = 2500
LABEL_FONT_SIZE = 9
EDGE_WIDTH = 1.5
ARROW_SIZE = 20
FIG_SIZE = (12, 10)
TITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 'small'  # Matplotlib font size for legend

DEFAULT_LAYOUT_SEED = 42
DEFAULT_LAYOUT_K_SCALE = 1.5  # For spring_layout's k parameter
DEFAULT_LAYOUT_ITERATIONS = 75  # For spring_layout


def get_shared_layout(adj_matrix_list, variable_names, seed=DEFAULT_LAYOUT_SEED,
                      k_scale=DEFAULT_LAYOUT_K_SCALE, iterations=DEFAULT_LAYOUT_ITERATIONS):
    """
    Computes a shared graph layout (Kamada-Kawai or Spring) based on the union of adjacencies
    from a list of adjacency matrices.
    Adjacency matrices can be simple DAGs (0/1) or CPDAGs (0, 1, -1).
    """
    n_nodes = len(variable_names)
    if n_nodes == 0:
        return {}

    idx_to_node = {i: name for i, name in enumerate(variable_names)}
    union_graph = nx.Graph()  # Use an undirected graph for layout purposes
    union_graph.add_nodes_from(variable_names)

    for adj_matrix in adj_matrix_list:
        if adj_matrix is None or adj_matrix.shape[0] != n_nodes:
            print("Warning: Skipping an invalid or mismatched adjacency matrix for layout.")
            continue

        for r_idx in range(n_nodes):
            for c_idx in range(r_idx + 1, n_nodes):  # Iterate over unique pairs
                u_node_name = idx_to_node.get(r_idx)
                v_node_name = idx_to_node.get(c_idx)
                if u_node_name is None or v_node_name is None:
                    continue  # Should not happen if variable_names is correct

                # Check for adjacency in the current matrix
                # For CPDAG: edge if not (G[i,j]==0 and G[j,i]==0)
                # For DAG: edge if G[i,j]==1 or G[j,i]==1
                has_adjacency = False
                if (adj_matrix[r_idx, c_idx] != 0 or adj_matrix[c_idx, r_idx] != 0):
                    has_adjacency = True

                if has_adjacency:
                    union_graph.add_edge(u_node_name, v_node_name)

    pos = None
    if union_graph.number_of_nodes() > 0:
        if union_graph.number_of_edges() > 0:
            if nx.is_connected(union_graph):  # Kamada-Kawai needs a connected graph
                try:
                    if union_graph.number_of_nodes() < 50:  # Heuristic: KK is slow for larger graphs
                        pos = nx.kamada_kawai_layout(union_graph)
                    else:
                        # print("Info: Graph too large or not suitable for Kamada-Kawai, using Spring layout.")
                        pass  # Fall through to spring_layout
                except (nx.NetworkXException, nx.NetworkXError, Exception) as e:
                    # print(f"Kamada-Kawai layout failed ({e}), falling back to Spring layout.")
                    pass  # Fall through to spring_layout
            # else:
            # print("Info: Union graph for layout is not connected, using Spring layout.")

            if pos is None:  # If Kamada-Kawai was skipped or failed
                k_val = k_scale / np.sqrt(union_graph.number_of_nodes()) if union_graph.number_of_nodes() > 0 else 0.5
                pos = nx.spring_layout(union_graph, seed=seed, k=k_val, iterations=iterations, scale=2.0)
        else:  # Nodes but no edges
            pos = nx.circular_layout(union_graph)  # Arrange isolated nodes in a circle
    else:  # No nodes
        pos = {}

    return pos