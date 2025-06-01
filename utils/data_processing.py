# project/utils/data_processing.py
import pandas as pd
import numpy as np
import os
import csv # Added for write_causal_edges if it was missing from your LLM_commonsense.py merge

def load_data(data_filepath):
    """Loads observational data from a CSV file."""
    if not os.path.exists(data_filepath):
        raise FileNotFoundError(f"Data file not found: {data_filepath}")
    data_df = pd.read_csv(data_filepath)
    data_np = data_df.to_numpy()
    variable_names = data_df.columns.tolist()
    return data_np, variable_names, data_df

def load_ground_truth_edges(gt_edges_filepath):
    """Loads ground truth edges from a text file (format: node1,node2 per line)."""
    if not os.path.exists(gt_edges_filepath):
        raise FileNotFoundError(f"Ground truth edges file not found: {gt_edges_filepath}")
    gt_edges = []
    with open(gt_edges_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): # Skip empty lines or comments
                continue
            parts = line.split(',')
            if len(parts) == 2:
                u, v = parts[0].strip(), parts[1].strip()
                if u and v: # Ensure nodes are not empty strings
                    gt_edges.append((u, v))
                else:
                    print(f"Warning: Malformed edge (empty node name) in '{line}' from {gt_edges_filepath}. Skipping.")
            else:
                print(f"Warning: Malformed line '{line}' in {gt_edges_filepath}. Expected 'node1,node2'. Skipping.")
    return gt_edges

def convert_edges_to_adj_matrix(edges, variable_names):
    """Converts a list of edges to an adjacency matrix (0 or 1 for directed edges)."""
    n_vars = len(variable_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
    node_to_idx = {name: i for i, name in enumerate(variable_names)}

    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            adj_matrix[idx_u, idx_v] = 1  # Represents u -> v
        else:
            if u not in node_to_idx:
                print(f"Warning: Node '{u}' from edge ('{u}', '{v}') not found in variable_names: {variable_names}")
            if v not in node_to_idx:
                print(f"Warning: Node '{v}' from edge ('{u}', '{v}') not found in variable_names: {variable_names}")
    return adj_matrix

def get_dataset_name(data_filepath):
    """Extracts a clean dataset name from the filepath (e.g., 'asia_N2000')."""
    return os.path.basename(data_filepath).replace(".csv", "")


def load_llm_prior_edges_as_adj_matrix(llm_edges_filepath, variable_names):
    """
    Loads LLM-generated causal edges from a CSV file (source,target per line)
    and converts them into an adjacency matrix (0 or 1 for directed edges).
    Returns None if file not found or errors occur.
    """
    if not os.path.exists(llm_edges_filepath):
        print(f"Warning: LLM prior edges file not found: {llm_edges_filepath}")
        return None

    n_vars = len(variable_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
    node_to_idx = {name: i for i, name in enumerate(variable_names)}

    edges_loaded_count = 0
    try:
        with open(llm_edges_filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    u, v = row[0].strip(), row[1].strip()
                    if u in node_to_idx and v in node_to_idx:
                        idx_u = node_to_idx[u]
                        idx_v = node_to_idx[v]
                        adj_matrix[idx_u, idx_v] = 1  # Represents u -> v
                        edges_loaded_count += 1
                    else:
                        if u not in node_to_idx:
                            print(
                                f"Warning (LLM Prior): Node '{u}' from edge ('{u}', '{v}') not in dataset variables. Skipping edge.")
                        if v not in node_to_idx:
                            print(
                                f"Warning (LLM Prior): Node '{v}' from edge ('{u}', '{v}') not in dataset variables. Skipping edge.")
                elif row:  # Non-empty row but not 2 elements
                    print(
                        f"Warning (LLM Prior): Malformed line '{','.join(row)}' in {llm_edges_filepath}. Expected 'source,target'. Skipping.")
        print(f"Loaded {edges_loaded_count} edges from LLM prior: {llm_edges_filepath}")
        return adj_matrix
    except Exception as e:
        print(f"Error loading LLM prior edges from {llm_edges_filepath}: {e}")
        return None
