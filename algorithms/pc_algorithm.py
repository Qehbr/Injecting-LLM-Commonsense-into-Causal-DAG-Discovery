# project/algorithms/pc_algorithm.py
import time
import numpy as np
from causallearn.search.ConstraintBased.PC import pc as causallearn_pc_algorithm
from causallearn.utils.cit import fisherz, chisq, \
    gsq  # For explicitly passing test functions if needed, or use string names
from .base_algorithm import BaseAlgorithm


class PCAlgorithm(BaseAlgorithm):
    """
    Implementation of the PC (Peter-Clark) algorithm using the causallearn library.
    """

    def __init__(self, data_type="continuous", alpha=0.05, indep_test_override=None, stable=True,
                 parameters_override=None):
        super().__init__(data_type)  # Call parent's __init__

        self.alpha = alpha
        self.stable = stable
        self.parameters = parameters_override if parameters_override is not None else {}  # For any other PC params

        if indep_test_override:
            self.indep_test = indep_test_override
        elif self.data_type == "discrete":
            self.indep_test = "chisq"  # Chi-square test for discrete data
            # self.indep_test = "gsq" # Alternative for discrete data
        elif self.data_type == "continuous":
            self.indep_test = "fisherz"  # Fisher's Z test (partial correlation) for continuous data
        else:
            raise ValueError(f"Unsupported data_type '{self.data_type}' for PC default independence tests. "
                             "Must be 'discrete' or 'continuous', or provide indep_test_override.")

        # Merge any additional parameters provided via parameters_override
        # For PC, alpha and indep_test are primary, but others might exist (e.g. uc_rule, uc_priority)
        self.parameters.setdefault('alpha', self.alpha)
        self.parameters.setdefault('stable', self.stable)
        # self.parameters.setdefault('verbose', False) # Example if you want to control verbosity

    def learn_structure(self, data_np, variable_names=None):
        """
        Runs the PC algorithm on the provided data.

        Args:
            data_np (np.ndarray): The input data (samples x variables).
            variable_names (list, optional): List of variable names.

        Populates:
            self.learned_graph_cpdag_raw: The raw adjacency matrix from causallearn's PC.
            self.execution_time_seconds: Time taken for the PC algorithm.
            self.variable_names: Stores the provided variable names.

        Returns:
            causallearn.graph.GraphClass.CausalGraph: The CausalGraph object returned by PC.
        """
        super().learn_structure(data_np, variable_names)  # Sets self.variable_names

        print(f"Starting PC algorithm with alpha: {self.alpha}, indep_test: '{self.indep_test}', stable: {self.stable}")
        print(f"  Additional PC parameters: {self.parameters}")  # Print any other params being used
        start_time = time.time()

        # Call the causallearn pc function
        # Pass only the core, known parameters directly, and unpack others from self.parameters
        # Note: causallearn's pc might not take a 'parameters' dict directly like GES.
        # It expects them as keyword arguments.

        # Parameters that pc() explicitly takes:
        pc_kwargs = {
            'alpha': self.alpha,
            'indep_test': self.indep_test,
            'stable': self.stable,
            'node_names': self.variable_names,
            # Add other specific PC parameters from self.parameters if they are known args of pc()
            'uc_rule': self.parameters.get('uc_rule', 0),  # Default from causallearn docs
            'uc_priority': self.parameters.get('uc_priority', 2),  # Default from causallearn docs
            'verbose': self.parameters.get('verbose', False),
            'show_progress': self.parameters.get('show_progress', True)
        }
        # Remove any None values from kwargs if pc function doesn't like them
        pc_kwargs = {k: v for k, v in pc_kwargs.items() if v is not None or k in ['node_names']}

        causal_graph_obj = causallearn_pc_algorithm(
            data_np,
            **pc_kwargs
            # alpha=self.alpha,
            # indep_test=self.indep_test,
            # stable=self.stable,
            # node_names=self.variable_names,
            # uc_rule=self.parameters.get('uc_rule', 0), # Example: 0: uc_sepset (Default)
            # uc_priority = self.parameters.get('uc_priority',2) # Example: 2: definiteMaxP
            # Add other relevant pc parameters here from self.parameters
        )
        end_time = time.time()
        self.execution_time_seconds = end_time - start_time
        print(f"PC algorithm finished in {self.execution_time_seconds:.3f} seconds.")

        # The learned graph (CPDAG or sometimes PAG depending on variant/settings)
        # is in causal_graph_obj.G.graph
        if hasattr(causal_graph_obj, 'G') and hasattr(causal_graph_obj.G, 'graph'):
            self.learned_graph_cpdag_raw = causal_graph_obj.G.graph
        else:
            print("Error: Could not extract graph matrix from PC algorithm output.")
            self.learned_graph_cpdag_raw = np.zeros((data_np.shape[1], data_np.shape[1]), dtype=int)

        if self.variable_names and self.learned_graph_cpdag_raw.shape[0] != len(self.variable_names):
            print("Warning: Shape of learned graph from PC doesn't match number of variable names.")
            # Fallback if graph is not properly formed or names mismatch
            if self.learned_graph_cpdag_raw.shape[0] != data_np.shape[1]:
                self.learned_graph_cpdag_raw = np.zeros((data_np.shape[1], data_np.shape[1]), dtype=int)

        return causal_graph_obj  # Return the full CausalGraph object