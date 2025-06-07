import torch
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader

from .base_algorithm import BaseAlgorithm
from .dag_gnn.modules import MLPEncoder, MLPDecoder, MLPDEncoder
from .dag_gnn.utils import (
    nll_gaussian, kl_gaussian_sem, stau,
    nll_categorical, _h_A
)
from utils.data_processing import load_llm_prior_edges_as_adj_matrix


class DAG_GNN_Algorithm(BaseAlgorithm):
    """
    Implements the DAG-GNN algorithm for causal structure discovery.

    This method uses a variational autoencoder (VAE) framework with a
    structural constraint to learn a Directed Acyclic Graph (DAG) from data.
    The acyclicity constraint is enforced using an augmented Lagrangian method.
    The implementation supports both continuous and discrete data and can
    optionally incorporate prior knowledge from Large Language Models (LLMs).

    Parameters
    ----------
    data_type : str, optional
        The type of data, either "continuous" or "discrete".
        Default is "continuous".
    parameters_override : dict, optional
        A dictionary of hyperparameters to override the defaults.
    llm_prior_edges_path : str, optional
        Path to a file containing prior edge information from an LLM.

    Attributes
    ----------
    params : dict
        The hyperparameters used by the algorithm.
    device : torch.device
        The computing device (CPU or CUDA).
    encoder : torch.nn.Module
        The encoder model of the VAE.
    decoder : torch.nn.Module
        The decoder model of the VAE.
    final_adj_matrix : np.ndarray
        The learned and processed adjacency matrix.
    llm_prior_edges_path : str
        The path to the LLM prior file.

    """

    def __init__(self, data_type="continuous", parameters_override=None, llm_prior_edges_path=None):
        super().__init__(data_type)
        self.params = parameters_override if parameters_override is not None else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.decoder = None
        self.final_adj_matrix = None
        self.llm_prior_edges_path = llm_prior_edges_path
        self.variable_names_for_prior_init = None

    def _setup_hyperparameters(self, num_nodes, data_dim):
        """
        Initializes hyperparameters by merging defaults with overrides.

        This is an internal helper method.

        Parameters
        ----------
        num_nodes : int
            The number of variables/nodes in the data.
        data_dim : int
            The dimensionality of the feature for each node. For discrete data,
            this is the number of classes.

        Returns
        -------
        Args
            An object containing all the hyperparameters.
        """
        defaults = {
            'epochs': 300, 'batch_size': 100, 'lr': 3e-3, 'encoder_hidden': 64,
            'decoder_hidden': 64, 'z_dims': num_nodes, 'lambda_A': 0.0, 'c_A': 1.0,
            'h_tol': 1e-8, 'k_max_iter': 100, 'graph_threshold': 0.3, 'seed': 42,
            'tau_A': 0.0, 'encoder_dropout': 0.0, 'decoder_dropout': 0.0,
            'log_interval': 50
        }
        defaults.update(self.params)

        class Args:
            pass

        args = Args()
        for key, value in defaults.items():
            setattr(args, key, value)
        args.data_variable_size = num_nodes
        args.x_dims = data_dim
        args.device = self.device
        self.args = args
        return args

    def _get_initial_adj_A_numpy(self, num_nodes, variable_names):
        """
        Creates the initial adjacency matrix (A) for the model.

        This matrix is initialized with small random values. If an LLM prior
        is provided, specified edges are given a stronger initial weight.
        This is an internal helper method.

        Parameters
        ----------
        num_nodes : int
            The number of nodes in the graph.
        variable_names : list of str
            The names of the variables, required for mapping LLM priors.

        Returns
        -------
        np.ndarray
            The initial adjacency matrix.
        """
        initial_adj_A_np = np.random.uniform(low=-0.01, high=0.01, size=(num_nodes, num_nodes))
        np.fill_diagonal(initial_adj_A_np, 0)

        if self.llm_prior_edges_path and variable_names:
            llm_adj_matrix = load_llm_prior_edges_as_adj_matrix(self.llm_prior_edges_path, variable_names)
            if llm_adj_matrix is not None and llm_adj_matrix.shape == (num_nodes, num_nodes):
                llm_edge_init_value = self.params.get("llm_prior_init_value", 0.3)
                num_llm_edges_applied = 0
                for r_idx in range(num_nodes):
                    for c_idx in range(num_nodes):
                        if llm_adj_matrix[r_idx, c_idx] == 1:
                            initial_adj_A_np[c_idx, r_idx] = llm_edge_init_value
                            num_llm_edges_applied += 1
                print(
                    f"Initialized DAG-GNN adj_A with {num_llm_edges_applied} edges from LLM prior (raw value: {llm_edge_init_value}).")
            else:
                print("Warning: LLM prior not loaded or shape mismatch. Using default random initialization for adj_A.")
        else:
            print("No LLM prior path or variable names for prior. Using default random initialization for adj_A.")
        return initial_adj_A_np

    def _initialize_models(self, variable_names_for_prior):
        """
        Initializes the encoder and decoder networks.

        This is an internal helper method.

        Parameters
        ----------
        variable_names_for_prior : list of str
            Variable names passed to initialize the adjacency matrix,
            potentially with priors.
        """
        args = self.args
        num_nodes = args.data_variable_size
        data_dim = args.x_dims
        initial_adj_A_np = self._get_initial_adj_A_numpy(num_nodes, variable_names_for_prior)

        if self.data_type == 'discrete':
            self.encoder = MLPDEncoder(
                n_hid=args.encoder_hidden,
                n_out=args.z_dims,
                num_classes=data_dim,
                adj_A=initial_adj_A_np,
            ).double().to(self.device)
            self.decoder = MLPDecoder(
                n_in_z=args.z_dims,
                n_out=data_dim,
                n_hid=args.decoder_hidden,
            ).double().to(self.device)
        else:  # Continuous
            self.encoder = MLPEncoder(
                n_xdims=data_dim,
                n_hid=args.encoder_hidden,
                n_out=args.z_dims,
                adj_A=initial_adj_A_np,
            ).double().to(self.device)
            self.decoder = MLPDecoder(
                n_in_z=args.z_dims,
                n_out=data_dim,
                n_hid=args.decoder_hidden,
            ).double().to(self.device)

    def _preprocess_discrete_data(self, data_np):
        """
        Maps discrete data values to zero-indexed integers.

        This is an internal helper method.

        Parameters
        ----------
        data_np : np.ndarray
            The input discrete data.

        Returns
        -------
        tuple[np.ndarray, int]
            A tuple containing the mapped data array and the number of
            unique classes.
        """
        if data_np.min() == 0:
            unique_vals = np.unique(data_np)
            if np.all(unique_vals == np.arange(len(unique_vals))):
                return data_np.astype(np.int64), len(unique_vals)
        unique_vals = np.unique(data_np)
        self.val_to_idx_map = {val: i for i, val in enumerate(unique_vals)}
        num_classes = len(unique_vals)
        mapped_data = np.vectorize(self.val_to_idx_map.get)(data_np)
        return mapped_data.astype(np.int64), num_classes

    def learn_structure(self, data_np, variable_names=None):
        """
        Learns the causal graph structure from data using the DAG-GNN model.

        This method executes the full training pipeline, including data setup,
        model initialization, and the augmented Lagrangian optimization loop to
        find a DAG.

        Parameters
        ----------
        data_np : np.ndarray
            The dataset, with shape (samples, nodes, features_per_node).
        variable_names : list of str, optional
            Names of the variables, used for initializing priors.

        Returns
        -------
        dict
            A dictionary containing the learned graph and model state, with keys:
            'G_cpdag', 'encoder_state', 'final_A_raw', 'final_A_processed'.
        """
        super().learn_structure(data_np, variable_names)
        self.variable_names_for_prior_init = variable_names
        start_time = time.time()

        if data_np.ndim == 2:
            data_np = np.expand_dims(data_np, axis=2)

        n_samples, n_nodes, n_dims_data_per_node = data_np.shape

        if self.data_type == 'discrete':
            data_np_processed_2d, num_classes_for_model = self._preprocess_discrete_data(data_np.copy())
            args = self._setup_hyperparameters(n_nodes, num_classes_for_model)
            feat_tensor = torch.from_numpy(data_np_processed_2d)
        else:
            args = self._setup_hyperparameters(n_nodes, n_dims_data_per_node)
            feat_tensor = torch.from_numpy(data_np)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_dataset = TensorDataset(feat_tensor, feat_tensor)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self._initialize_models(variable_names_for_prior=self.variable_names_for_prior_init)

        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=args.lr
        )
        c_A = args.c_A
        lambda_A = args.lambda_A
        h_A_old = np.inf
        print(f"Starting DAG-GNN training. Device: {self.device}, Log interval: {args.log_interval} epochs")

        for step_k in range(int(args.k_max_iter)):
            c_A_inner_loop = c_A

            print(
                f"\n[Augmented Lagrangian Step {step_k + 1}/{int(args.k_max_iter)}] Initial c_A: {c_A_inner_loop:.2e}, lambda_A: {lambda_A:.2e}")

            while c_A_inner_loop < 1e+20:
                for epoch in range(args.epochs):
                    results = self._train_epoch(
                        train_loader, lambda_A, c_A_inner_loop, optimizer
                    )
                    avg_total_loss, avg_nll, avg_kl, avg_hA_loss, avg_sparse, current_adj_A_param_matrix = results

                    if (epoch + 1) % args.log_interval == 0 or epoch == args.epochs - 1:
                        print(f"  Epoch [{epoch + 1}/{args.epochs}] | Total Loss: {avg_total_loss:.4f} "
                              f"| NLL: {avg_nll:.4f} | KL: {avg_kl:.4f} "
                              f"| h(A)Penalty: {avg_hA_loss:.4e} | L1Sparse: {avg_sparse:.4f}")

                final_A_this_stage = self.encoder.adj_A.data.clone()  # A_raw
                h_A_new = _h_A(final_A_this_stage, n_nodes)  # h(A_raw)

                if h_A_new.item() > 0.25 * h_A_old and c_A_inner_loop < 1e+19:
                    c_A_inner_loop *= 10
                    print(
                        f"    [AL Inner Update] h(A)={h_A_new.item():.2e} > 0.25*h_A_old({h_A_old:.2e}). Boosting c_A_inner to {c_A_inner_loop:.2e}")
                else:
                    print(
                        f"    [AL Inner Update] h(A)={h_A_new.item():.2e} sufficiently small or c_A maxed. Breaking inner c_A loop.")
                    break

            c_A = c_A_inner_loop  # Update main c_A with the one from the inner loop

            # Update lambda_A using the h_A from the end of the inner c_A optimization
            h_A_for_lambda_update = _h_A(final_A_this_stage, n_nodes)  # Use h_A_new based on final_A_this_stage

            lambda_A += c_A * h_A_for_lambda_update.item()
            h_A_old = h_A_for_lambda_update.item()  # Update h_A_old for the next AL step

            if h_A_old <= args.h_tol:
                print(f"\n[Convergence] h(A) = {h_A_old:.4e} <= h_tol ({args.h_tol}). Training stopped.")
                break
            if step_k == int(args.k_max_iter) - 1:
                print(
                    f"\n[Max AL Iterations Reached] k_max_iter = {args.k_max_iter}. Training stopped. Final h(A) = {h_A_old:.4e}")

        self.execution_time_seconds = time.time() - start_time
        print(f"DAG-GNN training finished in {self.execution_time_seconds:.2f} seconds.")

        final_A_raw = self.encoder.adj_A.data.clone().cpu().numpy()

        final_A_processed = np.sinh(3. * final_A_raw)

        final_A_processed[np.abs(final_A_processed) < args.graph_threshold] = 0

        self.learned_graph_cpdag_raw = np.zeros_like(final_A_processed, dtype=int)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if final_A_processed[
                    i, j] != 0:  # If W_adj[i,j] is non-zero (meaning edge j -> i in paper's formulation)
                    self.learned_graph_cpdag_raw[j, i] = 1  # Edge j -> i
                    self.learned_graph_cpdag_raw[i, j] = -1  # Mark other direction

        self.final_adj_matrix = final_A_processed
        return {"G_cpdag": self.learned_graph_cpdag_raw, "encoder_state": self.encoder.state_dict(),
                "final_A_raw": final_A_raw, "final_A_processed": self.final_adj_matrix}

    def _train_epoch(self, train_loader, lambda_A, c_A, optimizer):
        """
        Performs one epoch of training for the VAE model.

        This is an internal helper method.

        Parameters
        ----------
        train_loader : DataLoader
            The data loader for training batches.
        lambda_A : float
            The Lagrange multiplier for the acyclicity constraint.
        c_A : float
            The penalty parameter for the acyclicity constraint.
        optimizer : torch.optim.Optimizer
            The optimizer for updating model weights.

        Returns
        -------
        tuple
            A tuple containing average losses (total, NLL, KL, h(A) penalty,
            sparsity) and the final adjacency matrix parameter from the epoch.
        """
        epoch_total_loss = 0.0
        epoch_nll_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_hA_penalty = 0.0
        epoch_sparse_loss = 0.0

        self.encoder.train()
        self.decoder.train()
        current_adj_A_param_for_al_update = None  # This will be updated with the adj_A from the last batch

        for batch_idx, (data_batch, _) in enumerate(train_loader):
            data_batch = data_batch.to(self.args.device)
            optimizer.zero_grad()

            # Encoder forward pass
            z_encoded, adj_A_raw_learnable_param, _, Wa_encoder = self.encoder(data_batch)

            # Decoder forward pass
            x_reconstructed = self.decoder(z_encoded, adj_A_raw_learnable_param, Wa_encoder)

            if self.data_type == 'discrete':
                loss_nll = nll_categorical(x_reconstructed, data_batch)
                loss_kl = kl_gaussian_sem(z_encoded)
            else:  # Continuous
                loss_nll = nll_gaussian(x_reconstructed, data_batch, variance=self.args.variance)
                loss_kl = kl_gaussian_sem(z_encoded)

            sparse_loss = self.args.tau_A * torch.sum(torch.abs(adj_A_raw_learnable_param))
            h_A = _h_A(adj_A_raw_learnable_param, self.args.data_variable_size)
            h_A_loss = lambda_A * h_A + 0.5 * c_A * h_A * h_A  # Constraint penalty from Augmented Lagrangian

            loss = loss_nll + loss_kl + h_A_loss + sparse_loss
            loss.backward()
            optimizer.step()

            adj_A_raw_learnable_param.data = stau(adj_A_raw_learnable_param.data,
                                                  self.args.tau_A * optimizer.param_groups[0]['lr'])

            epoch_total_loss += loss.item()
            epoch_nll_loss += loss_nll.item()
            epoch_kl_loss += loss_kl.item()
            epoch_hA_penalty += h_A_loss.item()
            epoch_sparse_loss += sparse_loss.item()

            current_adj_A_param_for_al_update = adj_A_raw_learnable_param.data.clone()

        num_batches = len(train_loader)
        avg_total_loss = epoch_total_loss / num_batches
        avg_nll_loss = epoch_nll_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_hA_penalty = epoch_hA_penalty / num_batches
        avg_sparse_loss = epoch_sparse_loss / num_batches

        return (avg_total_loss, avg_nll_loss, avg_kl_loss, avg_hA_penalty, avg_sparse_loss,
                current_adj_A_param_for_al_update)

    def load_trained_model(self, model_path, num_nodes, data_feature_dim_or_num_classes, trained_params):
        """
        Loads a pre-trained encoder model and reconstructs its learned graph.

        Parameters
        ----------
        model_path : str
            Path to the saved PyTorch model state dictionary.
        num_nodes : int
            The number of nodes in the graph.
        data_feature_dim_or_num_classes : int
            The feature dimension (for continuous) or number of classes
            (for discrete) of the data the model was trained on.
        trained_params : dict
            A dictionary of hyperparameters used during the original training.
        """
        if trained_params:
            self.params.update(trained_params)
        args = self._setup_hyperparameters(num_nodes, data_feature_dim_or_num_classes)
        self._initialize_models(variable_names_for_prior=self.variable_names)

        state_dict_to_load = torch.load(model_path, map_location=self.device)

        self.encoder.load_state_dict(state_dict_to_load)
        self.encoder.eval()
        print(f"Successfully loaded model state from {model_path}")

        # After loading, extract the graph based on the loaded encoder's adj_A
        final_A_raw = self.encoder.adj_A.data.clone().cpu().numpy()
        final_A_processed = np.sinh(3. * final_A_raw)
        final_A_processed[np.abs(final_A_processed) < args.graph_threshold] = 0

        self.learned_graph_cpdag_raw = np.zeros_like(final_A_processed, dtype=int)
        n_nodes_loaded = final_A_processed.shape[0]
        for i in range(n_nodes_loaded):
            for j in range(n_nodes_loaded):
                if final_A_processed[i, j] != 0:
                    self.learned_graph_cpdag_raw[j, i] = 1
                    self.learned_graph_cpdag_raw[i, j] = -1
        self.final_adj_matrix = final_A_processed
