import time
import numpy as np
from causallearn.search.ScoreBased.GES import ges as causallearn_ges_algorithm
from .base_algorithm import BaseAlgorithm


class GESAlgorithm(BaseAlgorithm):
    """
    A wrapper for the Greedy Equivalence Search (GES) algorithm from the
    `causallearn` library.

    This class provides a standardized interface for the GES algorithm,
    setting appropriate default scoring functions and parameters based on the
    data type ('continuous' or 'discrete').

    Parameters
    ----------
    data_type : str, optional
        The type of data, either "continuous" or "discrete".
        Default is "continuous".
    score_name_override : str, optional
        A string to override the default score function name. If not provided,
        'local_score_BIC' is used for continuous data and 'local_score_BDeu'
        for discrete data.
    parameters_override : dict, optional
        A dictionary of parameters to update or add to the default
        parameters for the chosen score function.

    Attributes
    ----------
    score_name : str
        The name of the scoring function being used (e.g., 'local_score_BIC').
    parameters : dict
        The parameters for the scoring function.

    """

    def __init__(self, data_type="continuous", score_name_override=None, parameters_override=None):
        super().__init__(data_type)

        if self.data_type == "discrete":
            default_score_name_for_type = "local_score_BDeu"
            default_params_for_type = {"bdeu_sample_prior": 10.0, "bdeu_structure_prior": 1.0}
        elif self.data_type == "continuous":
            default_score_name_for_type = "local_score_BIC"
            default_params_for_type = {"penalty_discount": 2.0}
        else:
            raise ValueError(
                f"Unsupported data_type '{self.data_type}' for GES default scores. "
                "Must be 'discrete' or 'continuous'."
            )

        self.score_name = default_score_name_for_type
        self.parameters = default_params_for_type.copy()

        if score_name_override:
            self.score_name = score_name_override

        if parameters_override:
            self.parameters.update(parameters_override)

    def learn_structure(self, data_np, variable_names=None):
        """
        Executes the GES algorithm to learn a causal graph structure.

        This method runs the forward and backward search phases of GES to find
        the best-fitting causal graph for the given data, represented as a
        Completed Partially Directed Acyclic Graph (CPDAG).

        Parameters
        ----------
        data_np : np.ndarray
            The dataset as a NumPy array, where rows are samples and columns
            are variables.
        variable_names : list of str, optional
            The names corresponding to the columns of `data_np`.

        Returns
        -------
        dict
            A dictionary containing the results from the `causallearn` GES
            implementation. The learned graph object is under the key 'G'.
            The raw adjacency matrix can be accessed via `ges_results_record['G'].graph`.

        """
        super().learn_structure(data_np, variable_names)
        start_time = time.time()
        ges_results_record = causallearn_ges_algorithm(
            data_np,
            score_func=self.score_name,
            parameters=self.parameters,
            node_names=self.variable_names
        )
        end_time = time.time()
        self.execution_time_seconds = end_time - start_time
        self.learned_graph_cpdag_raw = ges_results_record['G'].graph
        if self.variable_names and self.learned_graph_cpdag_raw.shape[0] != len(self.variable_names):
            pass  # Silently proceed if shape mismatch
        return ges_results_record
