# project/algorithms/base_algorithm.py
from abc import ABC, abstractmethod
import numpy as np


class BaseAlgorithm(ABC):
    """
    Abstract base class for causal discovery algorithms.
    """

    def __init__(self, data_type="continuous", score_name=None, parameters=None):
        self.data_type = data_type
        self.score_name = score_name
        self.parameters = parameters if parameters is not None else {}

        self.learned_graph_cpdag_raw = None  # Raw adjacency matrix from the algorithm (e.g., CPDAG format)
        self.learned_graph_strict_dag = None  # Strictly directed DAG derived from CPDAG for evaluation
        self.execution_time_seconds = None
        self.variable_names = None  # Store variable names used during learning

    @abstractmethod
    def learn_structure(self, data_np, variable_names):
        """
        Learns the causal structure from data.
        This method should populate self.learned_graph_cpdag_raw and self.execution_time_seconds.

        Args:
            data_np (np.ndarray): The input data (samples x variables).
            variable_names (list): List of variable names corresponding to columns in data_np.

        Returns:
            A result object or dictionary specific to the algorithm, often containing the
            learned graph representation and other metadata.
        """
        self.variable_names = variable_names
        pass

    def get_learned_cpdag_adj_matrix(self):
        """Returns the raw learned CPDAG-like adjacency matrix from the algorithm."""
        if self.learned_graph_cpdag_raw is None:
            raise ValueError("Graph has not been learned yet. Call learn_structure() first.")
        return self.learned_graph_cpdag_raw

    def get_learned_strict_dag_adj_matrix(self):
        """
        Derives and returns a strictly directed DAG matrix from the learned CPDAG-like matrix.
        This is primarily for evaluation purposes (e.g., calculating strict F1).

        Assumes self.learned_graph_cpdag_raw representation:
         - G[i,j] =  1, G[j,i] = -1  => i -> j (This is a strictly directed edge)
         - G[i,j] = -1, G[j,i] =  1  => j -> i (This is also a strictly directed edge)
         - G[i,j] = -1, G[j,i] = -1  => i -- j (Undirected)
         - G[i,j] =  1, G[j,i] =  1  => i <-> j (Bi-directed, not part of a strict DAG)
         - G[i,j] =  0, G[j,i] =  0  => No edge

        Returns:
            np.ndarray: Adjacency matrix representing only the strictly directed edges (1 for i->j, 0 otherwise).
        """
        if self.learned_graph_cpdag_raw is None:
            raise ValueError("CPDAG-like graph has not been learned yet. Call learn_structure() first.")

        cpdag_matrix = self.learned_graph_cpdag_raw
        n_nodes = cpdag_matrix.shape[0]
        strict_dag_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

        num_strict_edges_derived = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                # A strictly directed edge i -> j in CPDAG is G[i,j]=1 and G[j,i]=-1
                if cpdag_matrix[i, j] == 1 and cpdag_matrix[j, i] == -1:
                    strict_dag_matrix[i, j] = 1
                    num_strict_edges_derived += 1

        # print(f"  Derived Strict DAG: {num_strict_edges_derived} strictly directed edges found.")
        self.learned_graph_strict_dag = strict_dag_matrix
        return self.learned_graph_strict_dag

    def get_execution_time(self):
        """Returns the execution time of the learn_structure method in seconds."""
        return self.execution_time_seconds

    def get_variable_names(self):
        """Returns the list of variable names used during learning."""
        return self.variable_names