# project/visualization/graph_visualization.py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import os

# Assuming plotting_utils.py is in the same directory
from .plotting_utils import (
    NODE_SIZE, LABEL_FONT_SIZE, EDGE_WIDTH, ARROW_SIZE,
    FIG_SIZE, TITLE_FONT_SIZE, LEGEND_FONT_SIZE
)


def ensure_visualization_folder(viz_folder="visualizations_output"):
    """Ensures the visualization folder exists, creating it if necessary."""
    os.makedirs(viz_folder, exist_ok=True)
    return viz_folder


def visualize_true_graph(true_adj, variable_names, filename, pos, dataset_name_str="",
                         viz_folder="visualizations_output"):
    """
    Visualizes the ground truth DAG.
    true_adj: Adjacency matrix where true_adj[i,j]=1 means i -> j.
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
            if true_adj[r_idx, c_idx] == 1:  # i -> j
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
    print(f"Saved ground truth graph to {filepath}")


def visualize_learned_graph_nx(learned_adj_raw_cpdag, variable_names, filename, pos, dataset_name_str="",
                               viz_folder="visualizations_output"):
    """
    Visualizes the learned CPDAG from its raw adjacency matrix representation.
    G[i,j] =  1, G[j,i] = -1 => i -> j
    G[i,j] = -1, G[j,i] = -1 => i -- j
    G[i,j] =  1, G[j,i] =  1 => i <-> j (bi-directed)
    """
    target_folder = ensure_visualization_folder(viz_folder)
    filepath = os.path.join(target_folder, filename)

    n_nodes = len(variable_names)
    idx_to_node = {i: name for i, name in enumerate(variable_names)}

    g_learned = nx.DiGraph()  # Use DiGraph to handle arrows
    g_learned.add_nodes_from(variable_names)

    directed_edges = []  # i -> j
    undirected_edges = []  # i -- j
    bidirected_edges = []  # i <-> j

    for r_idx in range(n_nodes):
        for c_idx in range(n_nodes):
            if r_idx == c_idx:
                continue

            u_node = idx_to_node.get(r_idx)
            v_node = idx_to_node.get(c_idx)
            if u_node is None or v_node is None: continue

            l_rc = learned_adj_raw_cpdag[r_idx, c_idx]
            l_cr = learned_adj_raw_cpdag[c_idx, r_idx]

            if l_rc == 1 and l_cr == -1:  # i -> j
                directed_edges.append((u_node, v_node))
            elif r_idx < c_idx:  # Process undirected and bidirected once for each pair
                if l_rc == -1 and l_cr == -1:  # i -- j
                    undirected_edges.append((u_node, v_node))
                elif l_rc == 1 and l_cr == 1:  # i <-> j
                    bidirected_edges.append((u_node, v_node))

    plt.figure(figsize=FIG_SIZE)
    nx.draw_networkx_nodes(g_learned, pos, node_size=NODE_SIZE, node_color='skyblue', edgecolors='black',
                           linewidths=1.0)
    nx.draw_networkx_labels(g_learned, pos, font_size=LABEL_FONT_SIZE, font_weight='bold')

    # Draw directed edges
    if directed_edges:
        nx.draw_networkx_edges(g_learned, pos, edgelist=directed_edges, edge_color='darkblue', style='solid',
                               arrows=True, arrowstyle='-|>', arrowsize=ARROW_SIZE, width=EDGE_WIDTH,
                               node_size=NODE_SIZE)
    # Draw undirected edges
    if undirected_edges:
        nx.draw_networkx_edges(g_learned, pos, edgelist=undirected_edges, edge_color='darkblue', style='solid',
                               arrows=False, width=EDGE_WIDTH, node_size=NODE_SIZE)
    # Draw bidirected edges (e.g., with a different style or color if needed)
    if bidirected_edges:
        nx.draw_networkx_edges(g_learned, pos, edgelist=bidirected_edges, edge_color='purple', style='dashed',
                               # Example style
                               arrows=True, arrowstyle='<->', arrowsize=ARROW_SIZE, width=EDGE_WIDTH,
                               node_size=NODE_SIZE)

    algo_name_part = dataset_name_str  # Use the full string passed
    title = f"Learned Graph ({algo_name_part})" if algo_name_part else "Learned Graph (NetworkX)"
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved learned graph visualization (NetworkX) to {filepath}")


def visualize_comparison_graphs(learned_adj_raw_cpdag, true_adj_dag, variable_names, filename, pos, dataset_name_str="",
                                viz_folder="visualizations_output"):
    """
    Visualizes a comparison between the learned CPDAG and the true DAG.
    - learned_adj_raw_cpdag: CPDAG format (1 for -> head, -1 for tail/undirected part, 0 for no mark)
                             A pair (i,j) can be (1,-1) for i->j, (-1,-1) for i--j, (1,1) for i<->j.
    - true_adj_dag: True DAG format (1 for i->j, 0 otherwise)
    """
    target_folder = ensure_visualization_folder(viz_folder)
    filepath = os.path.join(target_folder, filename)

    n_nodes = len(variable_names)
    idx_to_node = {i: name for i, name in enumerate(variable_names)}

    plot_g_nodes = nx.DiGraph()  # Base graph for nodes only
    plot_g_nodes.add_nodes_from(variable_names)

    plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1] + 1.8))  # Adjusted extra height for legend
    nx.draw_networkx_nodes(plot_g_nodes, pos, node_size=NODE_SIZE, node_color='lightcoral', edgecolors='black',
                           linewidths=1.0)
    nx.draw_networkx_labels(plot_g_nodes, pos, font_size=LABEL_FONT_SIZE, font_weight='bold')

    # Edge classification lists (for unique pairs i < j, or directed edges)
    edges_tp_directed = []  # Correctly learned i->j
    edges_fp_directed = []  # Learned i->j, but true is none or j->i (counted as reversed separately)
    edges_fn_missing_true_directed = []  # True i->j, but learned as none or i--j (counted as correct_adj_undir separately)
    edges_reversed = []  # Learned i->j, true was j->i

    edges_correct_adj_learned_undir = []  # Learned i--j, true was i->j or j->i
    edges_fp_undirected = []  # Learned i--j, but true was no edge

    edges_fp_bidirected = []  # Learned i<->j (true DAG cannot have this)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j: continue

            u_node = idx_to_node.get(i)
            v_node = idx_to_node.get(j)
            if u_node is None or v_node is None: continue

            l_ij, l_ji = learned_adj_raw_cpdag[i, j], learned_adj_raw_cpdag[j, i]
            t_ij, t_ji = true_adj_dag[i, j], true_adj_dag[j, i]  # t_ij=1 means i->j

            # Learned states
            learned_i_to_j = (l_ij == 1 and l_ji == -1)
            learned_j_to_i = (l_ij == -1 and l_ji == 1)  # Will be handled when loop gets to (j,i) for directed
            # For undirected/bidirected, process once for pair i < j
            is_pair_processing = (i < j)
            learned_undirected_ij = is_pair_processing and (l_ij == -1 and l_ji == -1)
            learned_bidirected_ij = is_pair_processing and (l_ij == 1 and l_ji == 1)

            # True states
            true_i_to_j = (t_ij == 1 and t_ji == 0)
            true_j_to_i = (t_ji == 1 and t_ij == 0)
            true_no_edge_ij_pair = (t_ij == 0 and t_ji == 0)

            # --- Classify DIRECTED learned edges (i->j) ---
            if learned_i_to_j:
                if true_i_to_j:
                    edges_tp_directed.append((u_node, v_node))
                elif true_j_to_i:
                    edges_reversed.append((u_node, v_node))
                elif true_no_edge_ij_pair:  # True has no edge between i and j
                    edges_fp_directed.append((u_node, v_node))
                # else: if true_no_edge_ij_pair is false, means true_j_to_i was missed by above, already caught by reversed.

            # --- Classify MISSING true DIRECTED edges (FNs) ---
            # A true i->j is a FN if it's not learned_i_to_j AND not learned_j_to_i (reversed) AND not learned_undirected_ij
            elif true_i_to_j:  # and learned_i_to_j is false (implicit from elif)
                # Check if the learned graph has j->i or i--j for this true i->j
                is_learned_as_j_to_i = (learned_adj_raw_cpdag[j, i] == 1 and learned_adj_raw_cpdag[i, j] == -1)

                # For checking learned undirected, ensure we look at the canonical pair
                min_idx, max_idx = min(i, j), max(i, j)
                is_learned_as_undirected_pair = (
                            learned_adj_raw_cpdag[min_idx, max_idx] == -1 and learned_adj_raw_cpdag[
                        max_idx, min_idx] == -1)

                if not is_learned_as_j_to_i and not is_learned_as_undirected_pair:
                    edges_fn_missing_true_directed.append((u_node, v_node))

            # --- Classify UNDIRECTED and BIDIRECTED learned edges (processed once per pair i < j) ---
            if is_pair_processing:
                if learned_undirected_ij:
                    # True edge for this pair could be i->j, j->i, or none
                    if true_i_to_j or true_j_to_i:
                        edges_correct_adj_learned_undir.append((u_node, v_node))
                    elif true_no_edge_ij_pair:
                        edges_fp_undirected.append((u_node, v_node))

                elif learned_bidirected_ij:
                    # Any learned bidirected edge is an FP because the true graph is a DAG.
                    edges_fp_bidirected.append((u_node, v_node))

    # Ensure uniqueness
    edges_tp_directed = list(set(edges_tp_directed))
    edges_fp_directed = list(set(edges_fp_directed))
    edges_fn_missing_true_directed = list(set(edges_fn_missing_true_directed))
    edges_reversed = list(set(edges_reversed))
    edges_correct_adj_learned_undir = list(set(edges_correct_adj_learned_undir))
    edges_fp_undirected = list(set(edges_fp_undirected))
    edges_fp_bidirected = list(set(edges_fp_bidirected))

    conn_style_arc = 'arc3,rad=0.1'  # For potentially overlapping directed edges

    # Draw order: FNs (most background) to TPs (most foreground)
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
    plt.subplots_adjust(bottom=0.22 if n_nodes > 6 else 0.20)  # Adjusted bottom margin for legend
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved comparison graph to {filepath}")