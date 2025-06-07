import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import os
from .plotting_utils import (
    NODE_SIZE, LABEL_FONT_SIZE, EDGE_WIDTH, ARROW_SIZE,
    FIG_SIZE, TITLE_FONT_SIZE, LEGEND_FONT_SIZE
)


def ensure_visualization_folder(viz_folder="visualizations_output"):
    """
    Ensures that the specified directory for saving visualizations exists.

    If the directory does not exist, it will be created.

    Parameters
    ----------
    viz_folder : str, optional
        The path to the directory to check or create.
        Default is "visualizations_output".

    Returns
    -------
    str
        The path to the ensured directory.
    """
    os.makedirs(viz_folder, exist_ok=True)
    return viz_folder


def visualize_true_graph(true_adj, variable_names, filename, pos, dataset_name_str="",
                         viz_folder="visualizations_output"):
    """
    Generates and saves a visualization of the ground truth DAG.

    Parameters
    ----------
    true_adj : np.ndarray
        The adjacency matrix of the ground truth DAG (0s and 1s).
    variable_names : list of str
        The names of the nodes in the graph.
    filename : str
        The name of the file to save the visualization to.
    pos : dict
        A dictionary mapping node names to their (x, y) positions for plotting.
    dataset_name_str : str, optional
        A string identifying the dataset, to be included in the plot title.
        Default is "".
    viz_folder : str, optional
        The directory where the visualization will be saved.
        Default is "visualizations_output".
    """
    target_folder = ensure_visualization_folder(viz_folder)
    filepath = os.path.join(target_folder, filename)
    n_nodes = len(variable_names)
    idx_to_node = {i: name for i, name in enumerate(variable_names)}
    g_true = nx.DiGraph()
    g_true.add_nodes_from(variable_names)
    edge_list = []
    for r_idx in range(n_nodes):
        for c_idx in range(n_nodes):
            if true_adj[r_idx, c_idx] == 1:
                u_node = idx_to_node.get(r_idx)
                v_node = idx_to_node.get(c_idx)
                if u_node and v_node:
                    edge_list.append((u_node, v_node))
    g_true.add_edges_from(edge_list)
    plt.figure(figsize=FIG_SIZE)
    nx.draw_networkx_nodes(g_true, pos, node_size=NODE_SIZE, node_color='lightgreen', edgecolors='black',
                           linewidths=1.0)
    nx.draw_networkx_labels(g_true, pos, font_size=LABEL_FONT_SIZE, font_weight='bold')
    if g_true.edges():
        nx.draw_networkx_edges(g_true, pos, edgelist=g_true.edges(), edge_color='black', style='solid',
                               arrows=True, arrowstyle='-|>', arrowsize=ARROW_SIZE, width=EDGE_WIDTH,
                               node_size=NODE_SIZE)
    title = f"Ground Truth DAG: {dataset_name_str}" if dataset_name_str else "Ground Truth DAG"
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_learned_graph_nx(learned_adj_raw_cpdag, variable_names, filename, pos, dataset_name_str="",
                               viz_folder="visualizations_output"):
    """
    Generates and saves a visualization of a learned CPDAG.

    This function can render directed, undirected, and bidirected edges based
    on the specific encoding of the input adjacency matrix.

    Parameters
    ----------
    learned_adj_raw_cpdag : np.ndarray
        The adjacency matrix of the learned CPDAG. This matrix uses a special
        encoding:
        - `A[i, j] == 1` and `A[j, i] == -1`: a directed edge i -> j
        - `A[i, j] == -1` and `A[j, i] == -1`: an undirected edge i -- j
        - `A[i, j] == 1` and `A[j, i] == 1`: a bidirected edge i <-> j
    variable_names : list of str
        The names of the nodes in the graph.
    filename : str
        The name of the file to save the visualization to.
    pos : dict
        A dictionary mapping node names to their (x, y) positions for plotting.
    dataset_name_str : str, optional
        A string identifying the algorithm and dataset, for the plot title.
        Default is "".
    viz_folder : str, optional
        The directory where the visualization will be saved.
        Default is "visualizations_output".
    """
    target_folder = ensure_visualization_folder(viz_folder)
    filepath = os.path.join(target_folder, filename)
    n_nodes = len(variable_names)
    idx_to_node = {i: name for i, name in enumerate(variable_names)}
    g_learned = nx.DiGraph()
    g_learned.add_nodes_from(variable_names)
    directed_edges = []
    undirected_edges = []
    bidirected_edges = []
    for r_idx in range(n_nodes):
        for c_idx in range(n_nodes):
            if r_idx == c_idx:
                continue
            u_node = idx_to_node.get(r_idx)
            v_node = idx_to_node.get(c_idx)
            if u_node is None or v_node is None: continue
            l_rc = learned_adj_raw_cpdag[r_idx, c_idx]
            l_cr = learned_adj_raw_cpdag[c_idx, r_idx]
            if l_rc == 1 and l_cr == -1:
                directed_edges.append((u_node, v_node))
            elif r_idx < c_idx:
                if l_rc == -1 and l_cr == -1:
                    undirected_edges.append((u_node, v_node))
                elif l_rc == 1 and l_cr == 1:
                    bidirected_edges.append((u_node, v_node))
    plt.figure(figsize=FIG_SIZE)
    nx.draw_networkx_nodes(g_learned, pos, node_size=NODE_SIZE, node_color='skyblue', edgecolors='black',
                           linewidths=1.0)
    nx.draw_networkx_labels(g_learned, pos, font_size=LABEL_FONT_SIZE, font_weight='bold')
    if directed_edges:
        nx.draw_networkx_edges(g_learned, pos, edgelist=directed_edges, edge_color='darkblue', style='solid',
                               arrows=True, arrowstyle='-|>', arrowsize=ARROW_SIZE, width=EDGE_WIDTH,
                               node_size=NODE_SIZE)
    if undirected_edges:
        nx.draw_networkx_edges(g_learned, pos, edgelist=undirected_edges, edge_color='darkblue', style='solid',
                               arrows=False, width=EDGE_WIDTH, node_size=NODE_SIZE)
    if bidirected_edges:
        nx.draw_networkx_edges(g_learned, pos, edgelist=bidirected_edges, edge_color='purple', style='dashed',
                               arrows=True, arrowstyle='<->', arrowsize=ARROW_SIZE, width=EDGE_WIDTH,
                               node_size=NODE_SIZE)
    algo_name_part = dataset_name_str
    title = f"Learned Graph ({algo_name_part})" if algo_name_part else "Learned Graph (NetworkX)"
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_comparison_graphs(learned_adj_raw_cpdag, true_adj_dag, variable_names, filename, pos, dataset_name_str="",
                                viz_folder="visualizations_output"):
    """
    Generates a detailed visualization comparing a learned graph to the true graph.

    Edges are color-coded and styled to indicate their status:
    - Correctly identified directed edges (True Positives)
    - Correct adjacencies learned as undirected
    - Extra directed or undirected edges (False Positives)
    - Reversed edges
    - Missed true edges (False Negatives)

    Parameters
    ----------
    learned_adj_raw_cpdag : np.ndarray
        The specially encoded adjacency matrix of the learned CPDAG.
    true_adj_dag : np.ndarray
        The standard adjacency matrix of the ground truth DAG.
    variable_names : list of str
        The names of the nodes in the graph.
    filename : str
        The name of the file to save the visualization to.
    pos : dict
        A dictionary mapping node names to their (x, y) positions for plotting.
    dataset_name_str : str, optional
        A string for the plot title, identifying the dataset and algorithm.
        Default is "".
    viz_folder : str, optional
        The directory where the visualization will be saved.
        Default is "visualizations_output".
    """
    target_folder = ensure_visualization_folder(viz_folder)
    filepath = os.path.join(target_folder, filename)
    n_nodes = len(variable_names)
    idx_to_node = {i: name for i, name in enumerate(variable_names)}
    plot_g_nodes = nx.DiGraph()
    plot_g_nodes.add_nodes_from(variable_names)
    plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1] + 1.8))
    nx.draw_networkx_nodes(plot_g_nodes, pos, node_size=NODE_SIZE, node_color='lightcoral', edgecolors='black',
                           linewidths=1.0)
    nx.draw_networkx_labels(plot_g_nodes, pos, font_size=LABEL_FONT_SIZE, font_weight='bold')
    edges_tp_directed = []
    edges_fp_directed = []
    edges_fn_missing_true_directed = []
    edges_reversed = []
    edges_correct_adj_learned_undir = []
    edges_fp_undirected = []
    edges_fp_bidirected = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j: continue
            u_node = idx_to_node.get(i)
            v_node = idx_to_node.get(j)
            if u_node is None or v_node is None: continue
            l_ij, l_ji = learned_adj_raw_cpdag[i, j], learned_adj_raw_cpdag[j, i]
            t_ij, t_ji = true_adj_dag[i, j], true_adj_dag[j, i]
            learned_i_to_j = (l_ij == 1 and l_ji == -1)
            learned_j_to_i = (l_ij == -1 and l_ji == 1)
            is_pair_processing = (i < j)
            learned_undirected_ij = is_pair_processing and (l_ij == -1 and l_ji == -1)
            learned_bidirected_ij = is_pair_processing and (l_ij == 1 and l_ji == 1)
            true_i_to_j = (t_ij == 1 and t_ji == 0)
            true_j_to_i = (t_ji == 1 and t_ij == 0)
            true_no_edge_ij_pair = (t_ij == 0 and t_ji == 0)
            if learned_i_to_j:
                if true_i_to_j:
                    edges_tp_directed.append((u_node, v_node))
                elif true_j_to_i:
                    edges_reversed.append((u_node, v_node))
                elif true_no_edge_ij_pair:
                    edges_fp_directed.append((u_node, v_node))
            elif true_i_to_j:
                is_learned_as_j_to_i = (learned_adj_raw_cpdag[j, i] == 1 and learned_adj_raw_cpdag[i, j] == -1)
                min_idx, max_idx = min(i, j), max(i, j)
                is_learned_as_undirected_pair = (
                        learned_adj_raw_cpdag[min_idx, max_idx] == -1 and learned_adj_raw_cpdag[
                    max_idx, min_idx] == -1)
                if not is_learned_as_j_to_i and not is_learned_as_undirected_pair:
                    edges_fn_missing_true_directed.append((u_node, v_node))
            if is_pair_processing:
                if learned_undirected_ij:
                    if true_i_to_j or true_j_to_i:
                        edges_correct_adj_learned_undir.append((u_node, v_node))
                    elif true_no_edge_ij_pair:
                        edges_fp_undirected.append((u_node, v_node))
                elif learned_bidirected_ij:
                    edges_fp_bidirected.append((u_node, v_node))
    edges_tp_directed = list(set(edges_tp_directed))
    edges_fp_directed = list(set(edges_fp_directed))
    edges_fn_missing_true_directed = list(set(edges_fn_missing_true_directed))
    edges_reversed = list(set(edges_reversed))
    edges_correct_adj_learned_undir = list(set(edges_correct_adj_learned_undir))
    edges_fp_undirected = list(set(edges_fp_undirected))
    edges_fp_bidirected = list(set(edges_fp_bidirected))
    conn_style_arc = 'arc3,rad=0.1'
    nx.draw_networkx_edges(plot_g_nodes, pos, edgelist=edges_fn_missing_true_directed, edge_color='silver',
                           style='dotted', arrows=True,
                           arrowstyle='-|>', arrowsize=ARROW_SIZE - 5, width=EDGE_WIDTH, connectionstyle=conn_style_arc,
                           node_size=NODE_SIZE, label='Missed True Directed Edge (FN)')
    nx.draw_networkx_edges(plot_g_nodes, pos, edgelist=edges_fp_undirected, edge_color='salmon', style='dashed',
                           arrows=False,
                           width=EDGE_WIDTH, node_size=NODE_SIZE, label='Extra Undirected (FP Skeleton)')
    nx.draw_networkx_edges(plot_g_nodes, pos, edgelist=edges_fp_bidirected, edge_color='magenta', style='dotted',
                           arrows=True,
                           arrowstyle='<->', arrowsize=ARROW_SIZE, width=EDGE_WIDTH, node_size=NODE_SIZE,
                           label='Extra Bi-directed (FP)')
    nx.draw_networkx_edges(plot_g_nodes, pos, edgelist=edges_fp_directed, edge_color='salmon', style='dashed',
                           arrows=True,
                           arrowstyle='-|>', arrowsize=ARROW_SIZE, width=EDGE_WIDTH, connectionstyle=conn_style_arc,
                           node_size=NODE_SIZE, label='Extra Directed (FP Directed)')
    nx.draw_networkx_edges(plot_g_nodes, pos, edgelist=edges_correct_adj_learned_undir, edge_color='dodgerblue',
                           style='solid', arrows=False,
                           width=EDGE_WIDTH, node_size=NODE_SIZE, label='Correct Adjacency (Learned Undir., True Dir.)')
    nx.draw_networkx_edges(plot_g_nodes, pos, edgelist=edges_reversed, edge_color='darkorange', style='solid',
                           arrows=True,
                           arrowstyle='-|>', arrowsize=ARROW_SIZE, width=EDGE_WIDTH, connectionstyle=conn_style_arc,
                           node_size=NODE_SIZE, label='Reversed Edge')
    nx.draw_networkx_edges(plot_g_nodes, pos, edgelist=edges_tp_directed, edge_color='forestgreen', style='solid',
                           arrows=True,
                           arrowstyle='-|>', arrowsize=ARROW_SIZE + 2, width=EDGE_WIDTH + 0.3,
                           node_size=NODE_SIZE, label='Correct Directed (TP Directed)')
    legend_elements = [
        Line2D([0], [0], color='forestgreen', lw=2.3, marker='>', markersize=0, label='Correct Directed (TP Dir)'),
        Line2D([0], [0], color='dodgerblue', lw=2, label='Correct Adjacency (Learned Undir)'),
        Line2D([0], [0], color='salmon', lw=2, linestyle='--', marker='>', markersize=0,
               label='Extra Directed (FP Dir)'),
        Line2D([0], [0], color='salmon', lw=2, linestyle='--', label='Extra Undirected (FP Skel)'),
        Line2D([0], [0], color='magenta', lw=2, linestyle=':', marker='>', markersize=0,
               label='Extra Bi-directed (FP BiDir)'),
        Line2D([0], [0], color='darkorange', lw=2, marker='>', markersize=0, label='Reversed Edge'),
        Line2D([0], [0], color='silver', lw=2, linestyle=':', marker='>', markersize=0,
               label='Missed True Directed (FN Dir)')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2,
               fontsize=LEGEND_FONT_SIZE)
    title = f"Learned vs. True Graph: {dataset_name_str}" if dataset_name_str else "Learned vs. True Graph Comparison"
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.subplots_adjust(bottom=0.22 if n_nodes > 6 else 0.20)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()
