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

        # 1. Determine default score and parameters based *only* on data_type
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

        # 2. Initialize with these type-based defaults
        self.score_name = default_score_name_for_type
        self.parameters = default_params_for_type.copy()  # Use a copy

        # 3. If score_name_override is provided, use it.
        if score_name_override:
            self.score_name = score_name_override
            # If the new score requires entirely different parameters,
            # parameters_override should ideally provide them. Otherwise,
            # parameters_override will modify the existing default_params_for_type.

        # 4. Finally, apply any parameters_override from the config.
        if parameters_override:
            self.parameters.update(parameters_override)

    def learn_structure(self, data_np, variable_names=None):
        """
        Runs the GES algorithm on the provided data.
        """
        super().learn_structure(data_np, variable_names)

        print(f"Starting GES algorithm with score: '{self.score_name}' and parameters: {self.parameters}")
        start_time = time.time()

        ges_results_record = causallearn_ges_algorithm(
            data_np,
            score_func=self.score_name,
            parameters=self.parameters,  # Pass the fully resolved parameters
            node_names=self.variable_names
        )
        end_time = time.time()
        self.execution_time_seconds = end_time - start_time
        print(f"GES algorithm finished in {self.execution_time_seconds:.3f} seconds.")

        self.learned_graph_cpdag_raw = ges_results_record['G'].graph

        if self.variable_names and self.learned_graph_cpdag_raw.shape[0] != len(self.variable_names):
            print("Warning: Shape of learned graph doesn't match number of variable names.")

        return ges_results_record