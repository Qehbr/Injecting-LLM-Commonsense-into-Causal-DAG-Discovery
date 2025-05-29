# project/algorithms/dag_gnn_algorithm.py

import torch
import torch.optim as optim
import numpy as np
import time
import math
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

from .base_algorithm import BaseAlgorithm
from .dag_gnn.modules import MLPEncoder, MLPDecoder, MLPDEncoder
from .dag_gnn.utils import (
    nll_gaussian, kl_gaussian_sem, matrix_poly, stau,
    nll_categorical, update_optimizer, _h_A
)


class DAG_GNN_Algorithm(BaseAlgorithm):
    """
    Implementation of the DAG-GNN algorithm.
    This class wraps the core logic, handling both continuous and discrete data.
    """

    def __init__(self, data_type="continuous", parameters_override=None):
        super().__init__(data_type)
        self.params = parameters_override if parameters_override is not None else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DAG-GNN will run on device: {self.device} for data_type: {self.data_type}")

        self.encoder = None
        self.decoder = None
        self.final_adj_matrix = None

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

    def _initialize_models(self):
        """Initializes the correct Encoder and Decoder based on data type."""
        args = self.args
        num_nodes = args.data_variable_size
        data_dim = args.x_dims

        if self.data_type == 'discrete':
            # Use MLPDEncoder for discrete data, which has an embedding layer.
            self.encoder = MLPDEncoder(
                n_in=num_nodes,
                n_hid=args.encoder_hidden,
                n_out=args.z_dims,
                num_classes=data_dim,  # Pass number of classes for the embedding layer
                adj_A=np.zeros((num_nodes, num_nodes)),
                batch_size=args.batch_size,
                do_prob=args.encoder_dropout
            ).double().to(self.device)
            # Decoder outputs logits for each class.
            self.decoder = MLPDecoder(
                n_in_node=num_nodes,  # Not used this way, legacy param
                n_in_z=args.z_dims,
                n_out=data_dim,  # Output layer size must be number of classes
                data_variable_size=num_nodes,
                batch_size=args.batch_size,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout
            ).double().to(self.device)
        else:  # Continuous
            self.encoder = MLPEncoder(
                n_in=num_nodes * data_dim,
                n_xdims=data_dim,
                n_hid=args.encoder_hidden,
                n_out=args.z_dims,
                adj_A=np.zeros((num_nodes, num_nodes)),
                batch_size=args.batch_size,
                do_prob=args.encoder_dropout
            ).double().to(self.device)
            # Decoder outputs reconstructed continuous values.
            self.decoder = MLPDecoder(
                n_in_node=num_nodes * data_dim,
                n_in_z=args.z_dims,
                n_out=data_dim,
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
        start_time = time.time()

        n_samples, n_nodes, n_dims_data = data_np.shape

        if self.data_type == 'discrete':
            data_np, num_classes = self._preprocess_discrete_data(data_np)
            args = self._setup_hyperparameters(n_nodes, num_classes)
            feat_tensor = torch.from_numpy(data_np)  # Will be converted to long in training loop
        else:
            args = self._setup_hyperparameters(n_nodes, n_dims_data)
            feat_tensor = torch.from_numpy(data_np)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_dataset = TensorDataset(feat_tensor, feat_tensor)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        self._initialize_models()

        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=args.lr
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

        # Augmented Lagrangian method
        c_A = args.c_A
        lambda_A = args.lambda_A
        h_A_old = np.inf

        best_elbo_loss = np.inf

        for step_k in range(int(args.k_max_iter)):
            print(
                f"\n[Augmented Lagrangian Step {step_k + 1}/{int(args.k_max_iter)}] | c_A: {c_A:.2e}, lambda_A: {lambda_A:.2e}")
            while c_A < 1e+20:
                for epoch in range(args.epochs):
                    elbo_loss, origin_A = self._train_epoch(
                        train_loader, lambda_A, c_A, optimizer
                    )

                scheduler.step()

                h_A_new = _h_A(origin_A, n_nodes)
                if h_A_new.item() > 0.25 * h_A_old:
                    c_A *= 10
                    print(f"  [AL Update] h(A) increased. Boosting c_A to {c_A:.2e}")
                else:
                    print("  [AL Update] h(A) stable or decreasing. Moving to next AL step.")
                    break

            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_old <= args.h_tol:
                print(f"\n[Convergence] h(A) = {h_A_old:.4e} <= {args.h_tol}. Training stopped.")
                break

        self.execution_time_seconds = time.time() - start_time
        print(f"DAG-GNN training finished in {self.execution_time_seconds:.2f} seconds.")

        # Final graph extraction
        final_A = self.encoder.adj_A.data.clone().cpu().numpy()
        print(f"\nLearned adjacency matrix statistics before thresholding:")
        print(f"  Max absolute value: {np.max(np.abs(final_A)):.6f}")
        print(f"  Mean absolute value: {np.mean(np.abs(final_A)):.6f}")
        print(f"  Std of values: {np.std(final_A):.6f}")
        print(f"  Number of values > 0.01: {np.sum(np.abs(final_A) > 0.01)}")
        print(f"  Number of values > 0.05: {np.sum(np.abs(final_A) > 0.05)}")
        print(f"  Number of values > 0.1: {np.sum(np.abs(final_A) > 0.1)}")
        print(
            f"  Number of values > threshold ({args.graph_threshold}): {np.sum(np.abs(final_A) > args.graph_threshold)}")

        final_A[np.abs(final_A) < args.graph_threshold] = 0

        self.learned_graph_cpdag_raw = np.zeros_like(final_A, dtype=int)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if final_A[i, j] != 0:
                    self.learned_graph_cpdag_raw[j, i] = 1
                    self.learned_graph_cpdag_raw[i, j] = -1

        self.final_adj_matrix = final_A
        return {"G_cpdag": self.learned_graph_cpdag_raw, "encoder_state": self.encoder.state_dict()}

    def _train_epoch(self, train_loader, lambda_A, c_A, optimizer):
        """A single training epoch with corrected architecture and loss."""
        epoch_loss = 0.0

        self.encoder.train()
        self.decoder.train()

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()

            # --- Correct VAE Forward Pass ---
            if self.data_type == 'discrete':
                data = data.long()
                # Encoder gets latent variable Z (called `edges` in original)
                z, _, origin_A, _, _, _, myA, Wa, _ = self.encoder(data, None, None)
                # Decoder reconstructs X_hat from Z
                _, x_hat, _ = self.decoder(data, z, self.args.data_variable_size, None, None, origin_A, None, Wa)
                loss_nll = nll_categorical(x_hat, data)
                loss_kl = kl_gaussian_sem(z)
            else:  # Continuous
                data = data.double()
                # Encoder gets latent variable Z
                z, _, origin_A, _, _, _, myA, Wa = self.encoder(data, None, None)
                # Decoder reconstructs X_hat from Z
                _, x_hat, _ = self.decoder(data, z, self.args.data_variable_size, None, None, origin_A, None, Wa)
                loss_nll = nll_gaussian(x_hat, data, variance=0.0)
                loss_kl = kl_gaussian_sem(z)

            # --- Loss Calculation ---
            sparse_loss = self.args.tau_A * torch.sum(torch.abs(origin_A))
            h_A = _h_A(origin_A, self.args.data_variable_size)
            h_A_loss = lambda_A * h_A + 0.5 * c_A * h_A * h_A

            edge_penalty = -0.1 * torch.sum(torch.abs(origin_A))  # Negative to encourage edges
            if torch.sum(torch.abs(origin_A)) < 1.0:  # If too few edges
                edge_penalty *= 10  # Stronger penalty

            loss = loss_nll + loss_kl + h_A_loss + sparse_loss + edge_penalty

            loss.backward()
            optimizer.step()

            # Apply soft thresholding
            myA.data = stau(myA.data, self.args.tau_A * optimizer.param_groups[0]['lr'])

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader), self.encoder.adj_A

    def load_trained_model(self, model_path, num_nodes, data_feature_dim_or_num_classes, trained_params):
        """Loads a pre-trained encoder model state."""
        print(f"Loading trained DAG-GNN model from: {model_path}")
        self.params.update(trained_params)
        args = self._setup_hyperparameters(num_nodes, data_feature_dim_or_num_classes)
        self._initialize_models()

        self.encoder.load_state_dict(torch.load(model_path, map_location=self.device))
        self.encoder.eval()

        final_A = self.encoder.adj_A.data.clone().cpu().numpy()
        final_A[np.abs(final_A) < args.graph_threshold] = 0

        self.learned_graph_cpdag_raw = np.zeros_like(final_A, dtype=int)
        n_nodes = final_A.shape[0]
        for i in range(n_nodes):
            for j in range(n_nodes):
                if final_A[i, j] != 0:
                    self.learned_graph_cpdag_raw[j, i] = 1
                    self.learned_graph_cpdag_raw[i, j] = -1
