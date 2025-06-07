from .ges_algorithm import GESAlgorithm
from .pc_algorithm import PCAlgorithm
from .dag_gnn_algorithm import DAG_GNN_Algorithm


class AlgorithmFactory:
    """
    A factory class for creating instances of causal discovery algorithms.

    This class provides a centralized way to instantiate different algorithm
    objects based on a given name and configuration.
    """

    @staticmethod
    def create_algorithm(algorithm_name: str, data_type: str = "continuous",
                         score_name_override: str = None,
                         parameters_override: dict = None):
        """
        Creates and returns an instance of a specified causal discovery algorithm.

        Parameters
        ----------
        algorithm_name : str
            The name of the algorithm to create. Supported values are "ges",
            "pc", and "dag-gnn". Case-insensitive.
        data_type : str, optional
            The type of data, e.g., "continuous" or "discrete".
            Default is "continuous".
        score_name_override : str, optional
            A specific score to use, overriding the default for the chosen
            algorithm and data type. Default is None.
        parameters_override : dict, optional
            A dictionary containing hyperparameters for the algorithm. The structure
            can vary depending on the algorithm. For "pc", it can contain
            'alpha', 'indep_test', etc. For "dag-gnn", it can contain
            model-specific parameters. Default is None.

        Returns
        -------
        object
            An instance of the requested algorithm class (e.g., GESAlgorithm,
            PCAlgorithm).

        Raises
        ------
        ValueError
            If the `algorithm_name` is not recognized.
        """
        algo_name_lower = algorithm_name.lower()
        common_params = parameters_override if parameters_override is not None else {}

        if algo_name_lower == "ges":
            return GESAlgorithm(data_type=data_type,
                                score_name_override=score_name_override,
                                parameters_override=common_params.get("ges_params", common_params))

        elif algo_name_lower == "pc":
            alpha = common_params.get('alpha', 0.05)
            indep_test = common_params.get('indep_test', None)
            stable = common_params.get('stable', True)
            return PCAlgorithm(data_type=data_type,
                               alpha=alpha,
                               indep_test_override=indep_test,
                               stable=stable,
                               parameters_override=common_params)

        elif algo_name_lower.startswith("dag-gnn"):
            dagnn_specific_params_block = common_params.get(algo_name_lower, {})
            if not dagnn_specific_params_block and "dag-gnn" in common_params:
                dagnn_specific_params_block = common_params.get("dag-gnn", {})
            if not dagnn_specific_params_block:
                dagnn_specific_params_block = common_params

            dagnn_params_for_init = dagnn_specific_params_block.copy()
            llm_prior_path = dagnn_params_for_init.pop("llm_prior_edges_path", None)

            return DAG_GNN_Algorithm(data_type=data_type,
                                     parameters_override=dagnn_params_for_init,
                                     llm_prior_edges_path=llm_prior_path)
        else:
            raise ValueError(f"Unknown algorithm name: '{algorithm_name}'.")


def get_algorithm(algorithm_name: str, data_type: str = "continuous",
                  score_name_override: str = None, parameters_override: dict = None):
    """
    A convenience function to get an algorithm instance from the factory.

    This function is a simple wrapper around the `AlgorithmFactory.create_algorithm`
    static method.

    Parameters
    ----------
    algorithm_name : str
        The name of the algorithm to create. Supported values are "ges",
        "pc", and "dag-gnn". Case-insensitive.
    data_type : str, optional
        The type of data, e.g., "continuous" or "discrete".
        Default is "continuous".
    score_name_override : str, optional
        A specific score to use, overriding the default for the chosen
        algorithm and data type. Default is None.
    parameters_override : dict, optional
        A dictionary containing hyperparameters for the algorithm. Default is None.

    Returns
    -------
    object
        An instance of the requested algorithm class.
    """
    return AlgorithmFactory.create_algorithm(algorithm_name, data_type, score_name_override, parameters_override)
