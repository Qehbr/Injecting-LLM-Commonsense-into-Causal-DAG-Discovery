# project/algorithms/dag_gnn_algorithm.py
import torch
import torch.optim as optim
import numpy as np
import time
import argparse
import math

from .base_algorithm import BaseAlgorithm
# Import from the encapsulated dag_gnn code
from .dag_gnn.modules import MLPEncoder, MLPDecoder, MLPDEncoder  # Add MLPDEncoder
from .dag_gnn.utils import nll_gaussian, kl_gaussian_sem, matrix_poly, my_softmax


# Helper to mimic the original script's stau function
def stau(w, tau):
    prox_plus = torch.nn.Threshold(0., 0.)
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


# Helper for discrete loss, adapted from original utils
def nll_categorical(preds, target):
    """Compute the log-likelihood of discrete variables."""
    # preds shape: (batch, nodes, classes)
    # target shape: (batch, nodes, 1) or (batch, nodes)
    # Ensure target is long and has the right shape
    target = target.squeeze(-1).long()

    # Use cross_entropy which combines log_softmax and nll_loss
    # It expects preds of shape (N, C) and target of shape (N)
    # We need to compute it for each node and average
    total_loss = 0.0
    for node_idx in range(preds.size(1)):
        total_loss += torch.nn.functional.cross_entropy(
            preds[:, node_idx, :], target[:, node_idx]
        )
    return total_loss / preds.size(1)


class DAG_GNN_Algorithm(BaseAlgorithm):
    """
    Wrapper for the DAG-GNN algorithm.
    Adapts the original training script to fit into the framework,
    now with separate handling for continuous and discrete data.
    """

    def __init__(self, data_type="continuous", parameters_override=None):
        super().__init__(data_type)
        self.params = parameters_override if parameters_override is not None else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DAG-GNN will run on device: {self.device} for data_type: {self.data_type}")

    def _setup_hyperparameters(self, num_nodes, data_dim):
        # ... (This method remains the same as before)
        args = argparse.Namespace()
        defaults = {
            'epochs': 300, 'batch_size': 100, 'lr': 3e-3, 'lr_decay': 200,
            'gamma': 1.0, 'encoder_hidden': 64, 'decoder_hidden': 64,
            'encoder': 'mlp', 'decoder': 'mlp', 'encoder_dropout': 0.0,
            'decoder_dropout': 0.0, 'z_dims': 1,
            'c_A': 1.0,  # Initial penalty factor
            'lambda_A': 0.01,  # Initial Lagrange multiplier (MUST be > 0 to start)
            'h_tol': 1e-8,
            'k_max_iter': 150,  # Increased outer loop iterations
            'tau_A': 0.0,  # L1 sparsity on A (keep 0 unless you want to force more sparsity)
            'graph_threshold': 0.3,
            'use_A_connect_loss': 0, 'use_A_positiver_loss': 0
        }
        defaults.update(self.params)
        for key, value in defaults.items():
            setattr(args, key, value)
        args.data_variable_size = num_nodes
        args.x_dims = data_dim
        args.device = self.device
        self.args = args
        return args

    def learn_structure(self, data_np, variable_names=None):
        """
        Runs the DAG-GNN training loop to learn the causal graph.
        """
        super().learn_structure(data_np, variable_names)
        n_samples, n_nodes, n_dims = data_np.shape[0], data_np.shape[1], data_np.shape[2] if data_np.ndim == 3 else 1

        if data_np.ndim == 2:
            data_np = np.expand_dims(data_np, axis=2)
            n_dims = 1

        args = self._setup_hyperparameters(n_nodes, n_dims)

        feat_tensor = torch.from_numpy(data_np).double()
        from torch.utils.data import TensorDataset
        train_data = TensorDataset(feat_tensor, feat_tensor)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_data, batch_size=args.batch_size)

        # --- Model Initialization based on data type ---
        if self.data_type == 'discrete':
            num_classes = int(np.max(data_np)) + 1
            args.x_dims = num_classes

            # Pass the new 'num_classes' argument here
            encoder = MLPDEncoder(
                n_in=n_nodes, n_hid=args.encoder_hidden, n_out=args.z_dims,
                num_classes=num_classes,  # <-- ADDED THIS ARGUMENT
                adj_A=np.zeros((n_nodes, n_nodes)), batch_size=args.batch_size,
                do_prob=args.encoder_dropout
            ).double().to(self.device)

            decoder = MLPDecoder(
                n_in_node=n_nodes, n_in_z=args.z_dims, n_out=num_classes, encoder=None,
                data_variable_size=n_nodes, batch_size=args.batch_size,
                n_hid=args.decoder_hidden, do_prob=args.decoder_dropout
            ).double().to(self.device)
            reconstruction_loss_fn = nll_categorical

        else:  # Continuous data
            # ... (continuous data initialization remains the same)
            encoder = MLPEncoder(
                n_nodes * args.x_dims, args.x_dims, args.encoder_hidden,
                int(args.z_dims), np.zeros((n_nodes, n_nodes)), args.batch_size,
                do_prob=args.encoder_dropout
            ).double().to(self.device)

            decoder = MLPDecoder(
                n_nodes * args.x_dims, args.z_dims, args.x_dims, encoder,
                data_variable_size=n_nodes, batch_size=args.batch_size,
                n_hid=args.decoder_hidden, do_prob=args.decoder_dropout
            ).double().to(self.device)
            reconstruction_loss_fn = nll_gaussian

        # ... (The rest of the function remains the same)
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr
        )
        c_A = args.c_A
        lambda_A = args.lambda_A
        h_A_old = np.inf
        best_graph = np.zeros((n_nodes, n_nodes))
        start_time = time.time()

        try:
            for step_k in range(int(args.k_max_iter)):
                for epoch in range(args.epochs):
                    epoch_nll = []
                    for batch_data, _ in train_loader:
                        batch_data = batch_data.to(self.device)
                        optimizer.zero_grad()
                        if self.data_type == 'discrete':
                            _, logits, origin_A, _, _, _, myA, Wa, _ = encoder(batch_data, None, None)
                            _, output, _ = decoder(batch_data, logits, n_nodes, None, None, origin_A, None, Wa)
                            output = my_softmax(output, axis=-1)
                        else:
                            enc_x, logits, origin_A, _, _, _, myA, Wa = encoder(batch_data, None, None)
                            _, output, _ = decoder(batch_data, logits, n_nodes * args.x_dims, None, None, origin_A, None, Wa)
                        if self.data_type == 'discrete':
                            loss_nll = reconstruction_loss_fn(output, batch_data)
                        else:
                            loss_nll = reconstruction_loss_fn(output, batch_data, 0.0)
                        loss_kl = kl_gaussian_sem(logits)
                        h_A = torch.trace(matrix_poly(origin_A * origin_A, n_nodes)) - n_nodes
                        sparse_loss = args.tau_A * torch.sum(torch.abs(origin_A))
                        loss = loss_kl + loss_nll + \
                               lambda_A * h_A + 0.5 * c_A * h_A * h_A + \
                               100. * torch.trace(origin_A * origin_A) + sparse_loss
                        loss.backward()
                        optimizer.step()
                        myA.data = stau(myA.data, args.tau_A * args.lr)
                    epoch_nll.append(loss_nll.item())
                    if (epoch + 1) % 100 == 0:
                        print(f"Epoch {epoch+1}/{args.epochs}, NLL: {np.mean(epoch_nll):.4f}, h(A): {h_A.item():.4f}")
                h_A_new = h_A.item()
                if abs(h_A_new) > 0.25 * abs(h_A_old):
                    c_A *= 10
                else:
                    break
                lambda_A += c_A * h_A_new
                h_A_old = h_A_new
                if abs(h_A_new) <= args.h_tol:
                    break
            best_graph = encoder.adj_A.data.clone().cpu().numpy()
        except KeyboardInterrupt:
            print("Training interrupted.")
            if 'encoder' in locals():
                best_graph = encoder.adj_A.data.clone().cpu().numpy()
        self.execution_time_seconds = time.time() - start_time
        final_adj_matrix = best_graph
        final_adj_matrix[np.abs(final_adj_matrix) < args.graph_threshold] = 0
        cpdag_matrix = np.zeros_like(final_adj_matrix, dtype=int)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if final_adj_matrix[i, j] != 0:
                    cpdag_matrix[j, i] = 1
                    cpdag_matrix[i, j] = -1
        self.learned_graph_cpdag_raw = cpdag_matrix
        return {"G_cpdag": cpdag_matrix, "final_A": final_adj_matrix}