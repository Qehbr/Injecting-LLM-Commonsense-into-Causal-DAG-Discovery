# project/algorithms/ges_algorithm.py
import time
import numpy as np
from causallearn.search.ScoreBased.GES import ges as causallearn_ges_algorithm
from .base_algorithm import BaseAlgorithm


class GESAlgorithm(BaseAlgorithm):
    """
    Implementation of the GES (Greedy Equivalence Search) algorithm
    using the causallearn library.
    """

    def __init__(self, data_type="continuous", score_name_override=None, parameters_override=None):
        super().__init__(data_type)  # Call parent's __init__

        # Set default score and parameters based on data_type
        if score_name_override:
            self.score_name = score_name_override
            self.parameters = parameters_override if parameters_override is not None else {}
        elif self.data_type == "discrete":
            self.score_name = "local_score_BDeu"
            self.parameters = {"bdeu_sample_prior": 10.0, "bdeu_structure_prior": 1.0}
        elif self.data_type == "continuous":
            # Default to BIC for continuous data as it's common
            self.score_name = "local_score_BIC"
            self.parameters = {"penalty_discount": 2.0}  # Default for BIC
            # Example for local_score_CV_general if you want to use it:
            # self.score_name = "local_score_CV_general"
            # self.parameters = {'kfold': 10, 'lambda_coe': 0.01} # Check causallearn docs for these
        else:
            raise ValueError(f"Unsupported data_type '{self.data_type}' for GES default scores. "
                             "Must be 'discrete' or 'continuous', or provide score_name_override.")

        # Allow parameters_override to update/add to defaults even if score is not overridden
        if parameters_override:
            self.parameters.update(parameters_override)

    def learn_structure(self, data_np, variable_names=None):
        """
        Runs the GES algorithm on the provided data.

        Args:
            data_np (np.ndarray): The input data (samples x variables).
            variable_names (list, optional): List of variable names.
                                            If None, causallearn will use default names (X0, X1...).

        Populates:
            self.learned_graph_cpdag_raw: The raw adjacency matrix from causallearn's GES.
                                         (CPDAG representation)
            self.execution_time_seconds: Time taken for the GES algorithm.
            self.variable_names: Stores the provided variable names.

        Returns:
            dict: The full 'Record' object returned by causallearn's ges function,
                  which includes the graph and other details.
        """
        super().learn_structure(data_np, variable_names)  # Sets self.variable_names

        print(f"Starting GES algorithm with score: '{self.score_name}' and parameters: {self.parameters}")
        start_time = time.time()

        # causallearn's ges function can take node_names.
        # If variable_names is None, causallearn handles it internally.
        ges_results_record = causallearn_ges_algorithm(
            data_np,
            score_func=self.score_name,
            parameters=self.parameters,
            node_names=self.variable_names  # Pass the stored variable names
        )
        end_time = time.time()
        self.execution_time_seconds = end_time - start_time
        print(f"GES algorithm finished in {self.execution_time_seconds:.3f} seconds.")

        # The learned graph (CPDAG) is in ges_results_record['G'].graph
        # This is an adjacency matrix where (from causallearn documentation/common practice):
        # G[i,j] =  1 and G[j,i] = -1 iff i -> j (directed edge from i to j)
        # G[i,j] = -1 and G[j,i] =  1 iff j -> i (directed edge from j to i) - should be consistent
        # G[i,j] = -1 and G[j,i] = -1 iff i -- j (undirected edge between i and j)
        # G[i,j] =  0 and G[j,i] =  0 iff no edge between i and j
        # G[i,j] =  1 and G[j,i] =  1 iff i <-> j (bi-directed edge, if the algorithm/score supports this)
        self.learned_graph_cpdag_raw = ges_results_record['G'].graph

        if self.variable_names and self.learned_graph_cpdag_raw.shape[0] != len(self.variable_names):
            print("Warning: Shape of learned graph doesn't match number of variable names.")

        return ges_results_record  # Return the full record for potential further inspection