# project/algorithms/dag_gnn_algorithm.py

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
    Implementation of the DAG-GNN algorithm.
    This class wraps the core logic, handling both continuous and discrete data.
    """

    def __init__(self, data_type="continuous", parameters_override=None,
                 llm_prior_edges_path=None):  # Added llm_prior_edges_path
        super().__init__(data_type)
        self.params = parameters_override if parameters_override is not None else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DAG-GNN will run on device: {self.device} for data_type: {self.data_type}")

        self.encoder = None
        self.decoder = None
        self.final_adj_matrix = None
        self.llm_prior_edges_path = llm_prior_edges_path  # Store the path
        self.variable_names_for_prior_init = None  # Will be set in learn_structure

    def _setup_hyperparameters(self, num_nodes, data_dim):
        """Sets up hyperparameters from defaults and overrides."""
        defaults = {
            'epochs': 300, 'batch_size': 100, 'lr': 3e-3, 'encoder_hidden': 64,
            'decoder_hidden': 64, 'z_dims': num_nodes, 'lambda_A': 0.0, 'c_A': 1.0,
            'h_tol': 1e-8, 'k_max_iter': 100, 'graph_threshold': 0.3, 'seed': 42,
            'tau_A': 0.0, 'encoder_dropout': 0.0, 'decoder_dropout': 0.0,
        }
        defaults.update(self.params)

        class Args:
            pass

        args = Args()
        for key, value in defaults.items():
            setattr(args, key, value)

        args.data_variable_size = num_nodes
        args.x_dims = data_dim  # For continuous: feature dim. For discrete: num_classes
        args.device = self.device
        self.args = args
        return args

    def _get_initial_adj_A_numpy(self, num_nodes, variable_names):
        """Prepares the initial numpy adjacency matrix for the encoder."""
        initial_adj_A_np = np.random.uniform(low=-0.01, high=0.01, size=(num_nodes, num_nodes))
        np.fill_diagonal(initial_adj_A_np, 0)  # No self-loops initially

        if self.llm_prior_edges_path and variable_names:
            print(f"Attempting to load LLM prior for DAG-GNN from: {self.llm_prior_edges_path}")
            llm_adj_matrix = load_llm_prior_edges_as_adj_matrix(self.llm_prior_edges_path, variable_names)

            if llm_adj_matrix is not None and llm_adj_matrix.shape == (num_nodes, num_nodes):
                llm_edge_init_value = self.params.get("llm_prior_init_value", 0.3)  # Configurable strength
                # A value of 0.3 for A_raw makes sinh(3*A_raw) approx sinh(0.9) ~ 1.03
                # A value of 0.1 for A_raw makes sinh(3*A_raw) approx sinh(0.3) ~ 0.30

                num_llm_edges_applied = 0
                for r_idx in range(num_nodes):
                    for c_idx in range(num_nodes):
                        if llm_adj_matrix[r_idx, c_idx] == 1:  # LLM suggests edge r_idx -> c_idx
                            # In DAG-GNN, A[j,i] influences edge i->j in the final graph
                            initial_adj_A_np[c_idx, r_idx] = llm_edge_init_value
                            num_llm_edges_applied += 1
                print(
                    f"Initialized DAG-GNN adj_A with {num_llm_edges_applied} edges from LLM prior (raw value: {llm_edge_init_value}).")
            else:
                print(
                    f"Warning: LLM prior not loaded or shape mismatch. Using default random initialization for adj_A.")
        else:
            print("No LLM prior path or variable names for prior. Using default random initialization for adj_A.")
        return initial_adj_A_np

    def _initialize_models(self, variable_names_for_prior):  # Pass variable_names
        """Initializes the correct Encoder and Decoder based on data type."""
        args = self.args
        num_nodes = args.data_variable_size
        data_dim = args.x_dims  # This is num_classes for discrete, or feature_dim for continuous

        # Prepare initial adj_A using LLM prior if available
        # This numpy array will be passed to the encoder
        initial_adj_A_np = self._get_initial_adj_A_numpy(num_nodes, variable_names_for_prior)

        if self.data_type == 'discrete':
            self.encoder = MLPDEncoder(
                n_in=num_nodes,  # Not directly used this way in MLPDEncoder if num_classes is primary
                n_hid=args.encoder_hidden,
                n_out=args.z_dims,
                num_classes=data_dim,  # Pass number of classes for the embedding layer
                adj_A=initial_adj_A_np,  # Pass the pre-initialized numpy array
                batch_size=args.batch_size,  # Keep for consistency if modules use it
                do_prob=args.encoder_dropout
            ).double().to(self.device)
            self.decoder = MLPDecoder(
                n_in_node=num_nodes,
                n_in_z=args.z_dims,
                n_out=data_dim,  # Output layer size must be number of classes
                data_variable_size=num_nodes,
                batch_size=args.batch_size,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout
            ).double().to(self.device)
        else:  # Continuous
            self.encoder = MLPEncoder(
                n_in=num_nodes * data_dim,  # Total input features if flattened
                n_xdims=data_dim,  # Dimension of features per node
                n_hid=args.encoder_hidden,
                n_out=args.z_dims,
                adj_A=initial_adj_A_np,  # Pass the pre-initialized numpy array
                batch_size=args.batch_size,
                do_prob=args.encoder_dropout
            ).double().to(self.device)
            self.decoder = MLPDecoder(
                n_in_node=num_nodes * data_dim,
                n_in_z=args.z_dims,
                n_out=data_dim,  # Output is reconstruction of node features
                data_variable_size=num_nodes,
                batch_size=args.batch_size,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout
            ).double().to(self.device)

    def _preprocess_discrete_data(self, data_np):
        """
        Maps arbitrary integer values in discrete data to a zero-indexed range
        [0, 1, ..., num_classes-1]. This is CRITICAL for embedding layers.
        """
        if data_np.min() == 0:
            unique_vals = np.unique(data_np)
            if np.all(unique_vals == np.arange(len(unique_vals))):
                print("Data is already zero-indexed and sequential. No preprocessing needed.")
                return data_np.astype(np.int64), len(unique_vals)

        print(f"Preprocessing discrete data. Original range: [{data_np.min()}, {data_np.max()}]")
        unique_vals = np.unique(data_np)
        self.val_to_idx_map = {val: i for i, val in enumerate(unique_vals)}
        num_classes = len(unique_vals)

        # Vectorized mapping
        mapped_data = np.vectorize(self.val_to_idx_map.get)(data_np)

        print(f"Data mapped to range: [{mapped_data.min()}, {mapped_data.max()}]. Num classes: {num_classes}")
        return mapped_data.astype(np.int64), num_classes

    def learn_structure(self, data_np, variable_names=None):
        super().learn_structure(data_np, variable_names)
        self.variable_names_for_prior_init = variable_names  # Store for _initialize_models
        start_time = time.time()

        # data_np is expected to be 3D: (samples, nodes, features_per_node)
        # For discrete, features_per_node is usually 1 (the class index)
        # For continuous, features_per_node can be >1 if multivariate nodes
        if data_np.ndim == 2:  # If (samples, nodes)
            data_np = np.expand_dims(data_np, axis=2)  # Assume univariate nodes [samples, nodes, 1]

        n_samples, n_nodes, n_dims_data_per_node = data_np.shape

        if self.data_type == 'discrete':
            # _preprocess_discrete_data expects 2D or 3D [N, nodes, 1] and returns 2D mapped data
            data_np_processed_2d, num_classes_for_model = self._preprocess_discrete_data(data_np.copy())  # Use copy
            args = self._setup_hyperparameters(n_nodes, num_classes_for_model)  # x_dims = num_classes
            # MLPDEncoder expects LongTensor for input if it's class indices.
            feat_tensor = torch.from_numpy(data_np_processed_2d)  # Shape [N, nodes]
        else:  # continuous
            args = self._setup_hyperparameters(n_nodes, n_dims_data_per_node)  # x_dims = features_per_node
            feat_tensor = torch.from_numpy(data_np)  # Shape [N, nodes, features_per_node]

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # DataLoader still expects first dim to be batchable
        train_dataset = TensorDataset(feat_tensor, feat_tensor)  # Using feat_tensor as both input and target
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        self._initialize_models(variable_names_for_prior=self.variable_names_for_prior_init)  # Pass names

        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=args.lr
        )
        # Scheduler remains, but its effectiveness might change with priors
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

        c_A = args.c_A
        lambda_A = args.lambda_A
        h_A_old = np.inf

        # Training loop (largely the same)
        # ... (rest of the learn_structure method, including the AL loop) ...
        for step_k in range(int(args.k_max_iter)):
            print(
                f"\n[Augmented Lagrangian Step {step_k + 1}/{int(args.k_max_iter)}] | c_A: {c_A:.2e}, lambda_A: {lambda_A:.2e}")
            # The inner loop for epochs and c_A updates
            c_A_inner_loop = c_A  # Use a temporary c_A for the inner while loop
            h_A_current_step = h_A_old  # To track h_A within this AL step

            while c_A_inner_loop < 1e+20:  # Original condition for c_A
                for epoch in range(args.epochs):
                    # Pass self.variable_names_for_prior_init if _train_epoch needs it, though it doesn't currently
                    elbo_loss, current_adj_A_param_matrix = self._train_epoch(
                        train_loader, lambda_A, c_A_inner_loop, optimizer
                    )
                    # Print ELBO, NLL, KL etc. if desired for monitoring
                    if epoch % 50 == 0 or epoch == args.epochs - 1:
                        print(f"  Epoch {epoch + 1}/{args.epochs}, ELBO: {elbo_loss:.4f}")

                # scheduler.step() # Step scheduler after all epochs for this c_A value

                # Check h(A) condition for this c_A_inner_loop
                h_A_new_for_inner_loop = _h_A(current_adj_A_param_matrix, n_nodes)  # Use current A from encoder

                # Paper's condition: if h(A) > 0.25 * h_A_previous_AL_step
                # Here, we check against h_A from the *start* of this AL step (h_A_current_step)
                # or a more dynamic check if h_A is not decreasing enough for current c_A.
                # Let's stick to the paper's outer loop logic for c_A updates.
                # The inner while loop is more about ensuring h(A) gets small enough for current lambda_A.
                # The original code's while c_A < 1e+20 combined with break seemed to achieve this.
                # For simplicity, let's use the structure where c_A is updated based on h_A_new outside this inner epoch loop.
                # The original had this c_A update logic inside the while c_A loop.

                # For now, let the inner epochs run, then check h_A outside.
                # The provided DAG-GNN code structure in the prompt has the c_A update logic
                # inside the "while c_A < 1e+20" loop, and breaks from it. Let's try to follow that.

                # Let's use the A from the encoder after all epochs for this c_A
                final_A_this_stage = self.encoder.adj_A.data.clone()  # This is A_raw
                h_A_new = _h_A(final_A_this_stage, n_nodes)  # h(A_raw)

                if h_A_new.item() > 0.25 * h_A_old and c_A_inner_loop < 1e+19:  # Check against h_A from PREVIOUS AL step
                    c_A_inner_loop *= 10
                    print(
                        f"  [AL Inner Update] h(A)={h_A_new.item():.2e} > 0.25*h_A_old({h_A_old:.2e}). Boosting c_A_inner to {c_A_inner_loop:.2e}")
                else:
                    print(
                        f"  [AL Inner Update] h(A)={h_A_new.item():.2e} sufficiently small or c_A maxed. Breaking inner c_A loop.")
                    break  # Break from the while c_A_inner_loop < 1e+20

            c_A = c_A_inner_loop  # Update main c_A with the one from the inner loop

            # Update lambda_A using the h_A from the end of the inner c_A optimization
            final_A_after_c_A_loop = self.encoder.adj_A.data.clone()
            h_A_for_lambda_update = _h_A(final_A_after_c_A_loop, n_nodes)

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

        final_A_raw = self.encoder.adj_A.data.clone().cpu().numpy()  # This is A_ij from paper (eta_ij)
        print(f"\nLearned raw adjacency matrix (A_raw) statistics before thresholding:")
        # ... (stats printing) ...
        final_A_processed = np.sinh(3. * final_A_raw)  # This is W_adj from paper
        final_A_processed[np.abs(final_A_processed) < args.graph_threshold] = 0

        self.learned_graph_cpdag_raw = np.zeros_like(final_A_processed, dtype=int)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if final_A_processed[
                    i, j] != 0:  # If W_adj[i,j] is non-zero (meaning edge j -> i in paper's formulation)
                    self.learned_graph_cpdag_raw[j, i] = 1  # Edge j -> i
                    self.learned_graph_cpdag_raw[i, j] = -1  # Mark other direction

        self.final_adj_matrix = final_A_processed  # Store the thresholded W_adj
        # Ensure this structure is what get_learned_strict_dag_adj_matrix expects for CPDAG
        return {"G_cpdag": self.learned_graph_cpdag_raw, "encoder_state": self.encoder.state_dict(),
                "final_A_raw": final_A_raw, "final_A_processed": self.final_adj_matrix}

    def _train_epoch(self, train_loader, lambda_A, c_A, optimizer):
        epoch_loss = 0.0
        self.encoder.train()
        self.decoder.train()

        current_adj_A_param_for_al_update = None

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.args.device)
            optimizer.zero_grad()

            # This is the nn.Parameter for A_raw, which is learnable
            # Its .data attribute is used for updates like stau and AL.
            adj_A_raw_learnable_param = self.encoder.adj_A

            if self.data_type == 'discrete':
                data = data.long()
                # Corrected unpacking: encoder now returns (z_nodes, A_raw_param, W_adj)
                z_encoded, _, _ = self.encoder(data, None, None)
                # The A_raw and W_adj are directly accessible via self.encoder.adj_A
                # and torch.sinh(3.*self.encoder.adj_A) if needed by other parts,
                # but for loss calculation, we use adj_A_raw_learnable_param.

                # Decoder uses A_raw_learnable_param internally to calculate W_adj and its inverse
                x_reconstructed = self.decoder(data, z_encoded, adj_A_raw_learnable_param)
                loss_nll = nll_categorical(x_reconstructed, data)
                loss_kl = kl_gaussian_sem(z_encoded)
            else:  # Continuous
                data = data.double()
                # Corrected unpacking
                z_encoded, _, _ = self.encoder(data, None, None)

                x_reconstructed = self.decoder(data, z_encoded, adj_A_raw_learnable_param)
                loss_nll = nll_gaussian(x_reconstructed, data, variance=self.args.variance)  # Use self.args.variance
                loss_kl = kl_gaussian_sem(z_encoded)

            # Loss terms use the learnable A_raw parameter
            sparse_loss = self.args.tau_A * torch.sum(torch.abs(adj_A_raw_learnable_param))
            h_A = _h_A(adj_A_raw_learnable_param, self.args.data_variable_size)
            h_A_loss = lambda_A * h_A + 0.5 * c_A * h_A * h_A

            loss = loss_nll + loss_kl + h_A_loss + sparse_loss
            loss.backward()

            # Grad clipping can be useful
            # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)

            optimizer.step()

            # Apply soft thresholding to A_raw (adj_A_raw_learnable_param.data)
            adj_A_raw_learnable_param.data = stau(adj_A_raw_learnable_param.data,
                                                  self.args.tau_A * optimizer.param_groups[0]['lr'])

            epoch_loss += loss.item()
            # This is the raw learnable parameter's data, after stau
            current_adj_A_param_for_al_update = adj_A_raw_learnable_param.data.clone()

        return epoch_loss / len(train_loader), current_adj_A_param_for_al_update

    def load_trained_model(self, model_path, num_nodes, data_feature_dim_or_num_classes, trained_params):
        # ... (load_trained_model as before, ensure it uses _get_initial_adj_A_numpy if needed for consistency,
        # though for loading, it just re-initializes structures and loads state_dict)
        print(f"Loading trained DAG-GNN model from: {model_path}")

        # Update self.params with trained_params FIRST, as _setup_hyperparameters uses self.params
        if trained_params:
            self.params.update(trained_params)

        args = self._setup_hyperparameters(num_nodes, data_feature_dim_or_num_classes)

        # Need variable names if the model used an LLM prior during its original training for _get_initial_adj_A_numpy
        # However, for loading a saved state, the adj_A will be overwritten by the loaded state_dict.
        # So, the LLM prior path for *this* instance is what matters if we were to re-initialize adj_A,
        # but we are loading it.
        # For consistency, let's assume _initialize_models creates the structures, and then load_state_dict fills them.
        # The `variable_names_for_prior` passed here should ideally match those used during the training of the loaded model,
        # but it's mostly for setting up the encoder structure dimensions.
        self._initialize_models(variable_names_for_prior=self.variable_names)  # Use stored variable_names

        try:
            state_dict_to_load = torch.load(model_path, map_location=self.device)
            self.encoder.load_state_dict(state_dict_to_load)
            self.encoder.eval()  # Set to evaluation mode
            print(f"Successfully loaded model state from {model_path}")
        except Exception as e:
            print(f"Error loading model state from {model_path}: {e}")
            raise

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
