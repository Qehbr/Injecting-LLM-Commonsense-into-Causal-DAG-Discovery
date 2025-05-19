# project/algorithms/algorithm_factory.py
from .ges_algorithm import GESAlgorithm
from .pc_algorithm import PCAlgorithm  # Import PCAlgorithm


class AlgorithmFactory:
    """
    Factory class to create and return instances of causal discovery algorithms.
    """

    @staticmethod
    def create_algorithm(algorithm_name: str, data_type: str = "continuous",
                         score_name_override: str = None,  # For score-based algos like GES
                         parameters_override: dict = None):  # Generic dict for other params
        """
        Creates an instance of the specified causal discovery algorithm.

        Args:
            algorithm_name (str): The name of the algorithm (e.g., "ges", "pc").
            data_type (str): The type of data ("continuous" or "discrete").
            score_name_override (str, optional): Specific score function (for GES).
            parameters_override (dict, optional): Specific parameters for the algorithm.
                                                  For PC, this could include 'alpha', 'indep_test', 'stable'.
                                                  For GES, this would be for its score function.

        Returns:
            An instance of a class derived from BaseAlgorithm.

        Raises:
            ValueError: If the algorithm_name is unknown.
        """
        algo_name_lower = algorithm_name.lower()

        # Initialize common parameters from parameters_override if they exist
        # These are general and might be used by multiple algorithms or set as defaults
        common_params = parameters_override if parameters_override is not None else {}

        if algo_name_lower == "ges":
            # GES takes score_name_override and its own parameters in parameters_override
            return GESAlgorithm(data_type=data_type,
                                score_name_override=score_name_override,
                                parameters_override=common_params.get("ges_params", common_params))  # Allow nesting

        elif algo_name_lower == "pc":
            # PC specific parameters can be passed directly or via the parameters_override dict
            alpha = common_params.get('alpha', 0.05)  # Default alpha if not provided
            indep_test = common_params.get('indep_test', None)  # None will let PCAlgorithm pick based on data_type
            stable = common_params.get('stable', True)
            # Pass the whole common_params too, as PCAlgorithm might pick other relevant keys
            return PCAlgorithm(data_type=data_type,
                               alpha=alpha,
                               indep_test_override=indep_test,
                               stable=stable,
                               parameters_override=common_params)  # Pass the full dict for other PC params

        # Add more algorithms here as elif blocks
        else:
            raise ValueError(f"Unknown algorithm name: '{algorithm_name}'. "
                             "Available algorithms: ['ges', 'pc', ...]")


# For simpler usage, you can also provide a function if you prefer over a static method
def get_algorithm(algorithm_name: str, data_type: str = "continuous",
                  score_name_override: str = None, parameters_override: dict = None):
    return AlgorithmFactory.create_algorithm(algorithm_name, data_type,
                                             score_name_override, parameters_override)
