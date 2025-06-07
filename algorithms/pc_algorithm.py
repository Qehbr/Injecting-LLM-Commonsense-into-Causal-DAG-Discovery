import time
import numpy as np
from causallearn.search.ConstraintBased.PC import pc as causallearn_pc_algorithm
from causallearn.utils.cit import fisherz, chisq, gsq
from .base_algorithm import BaseAlgorithm


class PCAlgorithm(BaseAlgorithm):
    """
    A wrapper for the PC (Peter-Clark) algorithm from the `causallearn` library.

    This class provides a standardized interface for the PC algorithm, a
    constraint-based method for causal structure learning. It sets appropriate
    default conditional independence tests based on the data type and manages
    algorithm parameters.

    Parameters
    ----------
    data_type : str, optional
        The type of data, either "continuous" or "discrete".
        Default is "continuous".
    alpha : float, optional
        The significance level for the conditional independence tests.
        Default is 0.05.
    indep_test_override : str, optional
        A string to override the default independence test. If not provided,
        'fisherz' is used for continuous data and 'chisq' for discrete data.
    stable : bool, optional
        Whether to use the "stable" version of the PC algorithm, which makes
        the output order-independent. Default is True.
    parameters_override : dict, optional
        A dictionary of additional parameters to pass to the `causallearn` PC
        function, such as 'uc_rule' or 'uc_priority'.

    Attributes
    ----------
    alpha : float
        The significance level for independence tests.
    stable : bool
        Flag for using the stable PC algorithm.
    indep_test : str
        The name of the conditional independence test being used.
    parameters : dict
        A dictionary of all parameters for the algorithm.

    """

    def __init__(self, data_type="continuous", alpha=0.05, indep_test_override=None, stable=True,
                 parameters_override=None):
        super().__init__(data_type)

        self.alpha = alpha
        self.stable = stable
        self.parameters = parameters_override if parameters_override is not None else {}

        if indep_test_override:
            self.indep_test = indep_test_override
        elif self.data_type == "discrete":
            self.indep_test = "chisq"
        elif self.data_type == "continuous":
            self.indep_test = "fisherz"
        else:
            raise ValueError(f"Unsupported data_type '{self.data_type}' for PC default independence tests. "
                             "Must be 'discrete' or 'continuous', or provide indep_test_override.")

        self.parameters.setdefault('alpha', self.alpha)
        self.parameters.setdefault('stable', self.stable)

    def learn_structure(self, data_np, variable_names=None):
        """
        Executes the PC algorithm to learn a causal graph structure.

        This method first finds the skeleton of the graph by performing a
        series of conditional independence tests, and then orients the edges
        to the extent possible, resulting in a Completed Partially Directed
        Acyclic Graph (CPDAG).

        Parameters
        ----------
        data_np : np.ndarray
            The dataset as a NumPy array, where rows are samples and columns
            are variables.
        variable_names : list of str, optional
            The names corresponding to the columns of `data_np`.

        Returns
        -------
        causallearn.graph.GeneralGraph.GeneralGraph
            The CausalGraph object returned by the `causallearn` PC function,
            which contains the learned graph and other metadata. The raw adjacency
            matrix can be accessed via `result.G.graph`.

        """
        super().learn_structure(data_np, variable_names)
        start_time = time.time()

        pc_kwargs = {
            'alpha': self.alpha,
            'indep_test': self.indep_test,
            'stable': self.stable,
            'node_names': self.variable_names,
            'uc_rule': self.parameters.get('uc_rule', 0),
            'uc_priority': self.parameters.get('uc_priority', 2),
            'verbose': self.parameters.get('verbose', False),
            'show_progress': self.parameters.get('show_progress', True)
        }
        pc_kwargs = {k: v for k, v in pc_kwargs.items() if v is not None or k in ['node_names']}

        causal_graph_obj = causallearn_pc_algorithm(
            data_np,
            **pc_kwargs
        )
        end_time = time.time()
        self.execution_time_seconds = end_time - start_time

        if hasattr(causal_graph_obj, 'G') and hasattr(causal_graph_obj.G, 'graph'):
            self.learned_graph_cpdag_raw = causal_graph_obj.G.graph
        else:
            self.learned_graph_cpdag_raw = np.zeros((data_np.shape[1], data_np.shape[1]), dtype=int)

        if self.variable_names and self.learned_graph_cpdag_raw.shape[0] != len(self.variable_names):
            if self.learned_graph_cpdag_raw.shape[0] != data_np.shape[1]:
                self.learned_graph_cpdag_raw = np.zeros((data_np.shape[1], data_np.shape[1]), dtype=int)
        return causal_graph_obj
