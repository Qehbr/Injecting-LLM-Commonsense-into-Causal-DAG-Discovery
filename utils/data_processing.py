import pandas as pd
import numpy as np
import os
import csv


def load_data(data_filepath):
    """
    Loads data from a CSV file into pandas and NumPy formats.

    Parameters
    ----------
    data_filepath : str
        The path to the input CSV data file.

    Returns
    -------
    tuple
        A tuple containing:
        - np.ndarray: The data from the CSV as a NumPy array.
        - list of str: The column headers from the CSV file.
        - pd.DataFrame: The data loaded into a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified `data_filepath` does not exist.
    """
    if not os.path.exists(data_filepath):
        raise FileNotFoundError(f"Data file not found: {data_filepath}")
    data_df = pd.read_csv(data_filepath)
    data_np = data_df.to_numpy()
    variable_names = data_df.columns.tolist()
    return data_np, variable_names, data_df


def load_ground_truth_edges(gt_edges_filepath):
    """
    Loads ground truth causal edges from a comma-separated file.

    The file is expected to have one edge per line, formatted as 'cause,effect'.
    Lines starting with '#' and empty lines are ignored.

    Parameters
    ----------
    gt_edges_filepath : str
        The path to the ground truth edges file.

    Returns
    -------
    list of tuple
        A list of tuples, where each tuple `(u, v)` represents a directed
        edge from node `u` to node `v`.

    Raises
    ------
    FileNotFoundError
        If the specified `gt_edges_filepath` does not exist.
    """
    if not os.path.exists(gt_edges_filepath):
        raise FileNotFoundError(f"Ground truth edges file not found: {gt_edges_filepath}")
    gt_edges = []
    with open(gt_edges_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) == 2:
                u, v = parts[0].strip(), parts[1].strip()
                if u and v:
                    gt_edges.append((u, v))
                else:
                    pass
            else:
                pass
    return gt_edges


def convert_edges_to_adj_matrix(edges, variable_names):
    """
    Converts a list of edges to a NumPy adjacency matrix.

    Parameters
    ----------
    edges : list of tuple
        A list of tuples, where each tuple `(u, v)` represents a directed
        edge from node `u` to node `v`.
    variable_names : list of str
        A list of all variable names, defining the order and size of the
        adjacency matrix.

    Returns
    -------
    np.ndarray
        An adjacency matrix where `A[i, j] = 1` indicates an edge from
        variable `i` to variable `j`.
    """
    n_vars = len(variable_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
    node_to_idx = {name: i for i, name in enumerate(variable_names)}

    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            adj_matrix[idx_u, idx_v] = 1
        else:
            if u not in node_to_idx:
                pass
            if v not in node_to_idx:
                pass
    return adj_matrix


def get_dataset_name(data_filepath):
    """
    Extracts a clean dataset name from a file path.

    This is done by taking the base name of the file and removing the
    '.csv' extension.

    Parameters
    ----------
    data_filepath : str
        The path to the data file.

    Returns
    -------
    str
        The extracted name of the dataset.
    """
    return os.path.basename(data_filepath).replace(".csv", "")


def load_llm_prior_edges_as_adj_matrix(llm_edges_filepath, variable_names):
    """
    Loads prior edges from a CSV file and converts them to an adjacency matrix.

    If the file does not exist, it returns None. The CSV is expected to have
    two columns: cause and effect.

    Parameters
    ----------
    llm_edges_filepath : str
        The path to the CSV file containing the prior edges.
    variable_names : list of str
        A list of all variable names to define the matrix dimensions and map
        node names to indices.

    Returns
    -------
    np.ndarray or None
        An adjacency matrix representing the prior graph, or None if the
        specified file does not exist.
    """
    if not os.path.exists(llm_edges_filepath):
        return None

    n_vars = len(variable_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
    node_to_idx = {name: i for i, name in enumerate(variable_names)}
    edges_loaded_count = 0
    with open(llm_edges_filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                u, v = row[0].strip(), row[1].strip()
                if u in node_to_idx and v in node_to_idx:
                    idx_u = node_to_idx[u]
                    idx_v = node_to_idx[v]
                    adj_matrix[idx_u, idx_v] = 1
                    edges_loaded_count += 1
                else:
                    if u not in node_to_idx:
                        pass
                    if v not in node_to_idx:
                        pass
            elif row:
                pass
    return adj_matrix
