from abc import ABC, abstractmethod
import numpy as np


class BaseAlgorithm(ABC):
    """
    An abstract base class for causal discovery algorithms.

    This class provides a common interface for various structure learning
    algorithms, ensuring they can be used interchangeably. It handles storing
    the learned graph, execution time, and other metadata.

    Parameters
    ----------
    data_type : str, optional
        The type of data the algorithm is designed for, e.g., "continuous" or
        "discrete". Default is "continuous".
    score_name : str, optional
        The name of the scoring function used by the algorithm, if applicable.
        Default is None.
    parameters : dict, optional
        A dictionary of hyperparameters for the algorithm. Default is an
        empty dictionary.

    Attributes
    ----------
    data_type : str
        The type of data.
    score_name : str
        The name of the score used.
    parameters : dict
        Hyperparameters for the algorithm.
    learned_graph_cpdag_raw : np.ndarray
        The adjacency matrix of the learned CPDAG (Completed Partially
        Directed Acyclic Graph).
    learned_graph_strict_dag : np.ndarray
        The adjacency matrix of the learned graph with only directed edges.
    execution_time_seconds : float
        The time taken to run the `learn_structure` method.
    variable_names : list of str
        The names of the variables (nodes) in the graph.

    """

    def __init__(self, data_type="continuous", score_name=None, parameters=None):
        self.data_type = data_type
        self.score_name = score_name
        self.parameters = parameters if parameters is not None else {}
        self.learned_graph_cpdag_raw = None
        self.learned_graph_strict_dag = None
        self.execution_time_seconds = None
        self.variable_names = None

    @abstractmethod
    def learn_structure(self, data_np, variable_names):
        """
        Learns the causal graph structure from data.

        This is an abstract method that must be implemented by any subclass.

        Parameters
        ----------
        data_np : np.ndarray
            The dataset as a NumPy array, where rows are samples and columns
            are variables.
        variable_names : list of str
            The names corresponding to the columns of `data_np`.

        """
        self.variable_names = variable_names
        pass

    def get_learned_cpdag_adj_matrix(self):
        """
        Returns the adjacency matrix of the learned CPDAG.

        The matrix follows a specific encoding:
        - `A[i, j] == 1` and `A[j, i] == -1`: a directed edge i -> j
        - `A[i, j] == 1` and `A[j, i] == 1`: an undirected edge i - j
        - `A[i, j] == 0` and `A[j, i] == 0`: no edge between i and j

        Returns
        -------
        np.ndarray
            The adjacency matrix representing the learned CPDAG.

        Raises
        ------
        ValueError
            If the graph structure has not been learned by calling
            `learn_structure()` first.

        """
        if self.learned_graph_cpdag_raw is None:
            raise ValueError("Graph has not been learned yet. Call learn_structure() first.")
        return self.learned_graph_cpdag_raw

    def get_learned_strict_dag_adj_matrix(self):
        """
        Converts the learned CPDAG to a strict DAG by removing undirected edges.

        This method extracts only the purely directed edges from the learned
        CPDAG representation. An edge i -> j exists in the result if and only
        if `cpdag_matrix[i, j] == 1` and `cpdag_matrix[j, i] == -1`.

        Returns
        -------
        np.ndarray
            A standard adjacency matrix (containing only 0s and 1s) for the
            strict DAG.

        Raises
        ------
        ValueError
            If the CPDAG has not been learned by calling `learn_structure()`
            first.

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
                if cpdag_matrix[i, j] == 1 and cpdag_matrix[j, i] == -1:
                    strict_dag_matrix[i, j] = 1
                    num_strict_edges_derived += 1
        self.learned_graph_strict_dag = strict_dag_matrix
        return self.learned_graph_strict_dag

    def get_execution_time(self):
        """
        Returns the execution time of the learning process.

        Returns
        -------
        float or None
            The execution time in seconds, or None if not recorded.

        """
        return self.execution_time_seconds

    def get_variable_names(self):
        """
        Returns the names of the variables used.

        Returns
        -------
        list of str or None
            A list of variable names, or None if not set.

        """
        return self.variable_names
