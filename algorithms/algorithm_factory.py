# project/algorithms/algorithm_factory.py
from .ges_algorithm import GESAlgorithm
from .pc_algorithm import PCAlgorithm
from .dag_gnn_algorithm import DAG_GNN_Algorithm


class AlgorithmFactory:
    @staticmethod
    def create_algorithm(algorithm_name: str, data_type: str = "continuous",
                         score_name_override: str = None,
                         parameters_override: dict = None):
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

        elif algo_name_lower.startswith("dag-gnn"):  # Catches "dag-gnn", "dag-gnn-gemini", "dag-gnn-claude"
            # Extract DAG-GNN specific params, which might be nested under the full algo name
            # e.g., parameters_override might be {"dag-gnn-gemini": {"lr": 0.01, "llm_prior_edges_path": "..."}}
            # or just {"lr": 0.01, "llm_prior_edges_path": "..."} if not nested.

            # Try to get params specific to the full algo_name_lower first
            dagnn_specific_params_block = common_params.get(algo_name_lower, {})
            # If not found, try to get params for plain "dag-gnn"
            if not dagnn_specific_params_block and "dag-gnn" in common_params:
                dagnn_specific_params_block = common_params.get("dag-gnn", {})
            # If still nothing, use common_params directly (assuming they are flat)
            if not dagnn_specific_params_block:
                dagnn_specific_params_block = common_params

            # Ensure we make a copy to modify for pop
            dagnn_params_for_init = dagnn_specific_params_block.copy()

            llm_prior_path = dagnn_params_for_init.pop("llm_prior_edges_path", None)
            # Pop other params that are not for DAG_GNN_Algorithm's direct parameters_override if any

            return DAG_GNN_Algorithm(data_type=data_type,
                                     parameters_override=dagnn_params_for_init,  # Pass potentially modified params
                                     llm_prior_edges_path=llm_prior_path)
        else:
            raise ValueError(f"Unknown algorithm name: '{algorithm_name}'.")


# ... (get_algorithm function remains same) ...
def get_algorithm(algorithm_name: str, data_type: str = "continuous",
                  score_name_override: str = None, parameters_override: dict = None):
    return AlgorithmFactory.create_algorithm(algorithm_name, data_type,
                                             score_name_override, parameters_override)