# project/algorithms/dag_gnn/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import preprocess_adj_new, preprocess_adj_new1


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True,
                 tol=0.1):  # adj_A is numpy array
        super(MLPEncoder, self).__init__()

        # Directly use the passed numpy array for the nn.Parameter
        self.adj_A = nn.Parameter(torch.from_numpy(adj_A).double())  # Changed this line

        self.Wa = nn.Parameter(torch.zeros(n_out, dtype=torch.double),
                               requires_grad=True)  # Check n_out or n_in for Wa size based on usage
        self.fc1 = nn.Linear(n_xdims, n_hid, bias=True)  # n_xdims is features per node
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)  # n_out is z_dims

        self.z = nn.Parameter(torch.tensor(tol, dtype=torch.double))  # Not used in provided code
        self.z_positive = nn.Parameter(torch.ones_like(self.adj_A.data),
                                       dtype=torch.double)  # Not used in provided code

        self._init_weights()  # Initialize fc layers

    # ... (_init_weights and forward remain the same) ...
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:  # Check if bias exists
                    m.bias.data.fill_(0.0)

    def forward(self, inputs, rel_rec, rel_send):  # inputs is [batch, nodes, x_dims] for continuous
        # adj_A is A_raw. adj_A_processed is W_adj = sinh(3*A_raw)
        adj_A_processed = torch.sinh(3. * self.adj_A)  # This is W_adj from paper

        # For continuous, inputs might be [batch_size, num_nodes, x_dims]
        # The original code seems to expect inputs to be [batch_size * num_nodes, x_dims]
        # or handle reshaping internally if fc1 takes n_xdims.
        # If inputs is [batch_size, num_nodes, x_dims], and fc1 input is n_xdims
        # Assuming inputs is correctly shaped [batch_size, num_nodes, x_dims]
        # and we process each node's features independently first.

        # x is the output of feature extraction per node, before graph convolution
        # inputs typically: [batch, num_nodes, data_dim_per_node]
        # fc1 input: data_dim_per_node. Output: n_hid
        # fc2 input: n_hid. Output: z_dims (if z_dims is per node latent representation)

        # Original paper: X_i = f_i(X_pa(i)) + Z_i -> Z_i = X_i - f_i(X_pa(i))
        # Here, encoder seems to produce Z from X directly, then decoder reconstructs X from Z and graph A.
        # Let x be the node features transformed by MLP (encoder part for features)
        # x should be [batch, num_nodes, z_dims] if z_dims is latent dim per node

        # If n_in was num_nodes * data_dim (flattened), inputs must be flattened.
        # Given n_xdims, it's likely fc1 processes features of each node.
        # Input shape: (batch_size, num_nodes, n_xdims)

        h1 = F.relu(self.fc1(inputs))  # Output: (batch_size, num_nodes, n_hid)
        z_nodes = self.fc2(h1)  # Output: (batch_size, num_nodes, n_out=z_dims) -> These are the Z_i

        # The 'logits' (reconstructed means before noise for VAE part)
        # or latent interaction part, this seems to be graph convolution like step
        # The paper's encoder is g_i(Z_i, N_i) for edge (i,j) prediction
        # Here, adj_A_for_z (derived from adj_A_processed/W_adj) is used.
        # adj_A_for_z is I - W_adj^T
        # logits = (I - W_adj^T) @ (Z + Wa) - Wa  (if Wa is broadcasted or per-node)
        # This step computes the "message passing" or influence.
        # If Wa is scalar or global, broadcasting needed. If Wa is per-node (size z_dims), it's fine.
        # Let's assume Wa is size z_dims and applies to z_nodes.
        # The original code has Wa as size n_out (z_dims).

        adj_A_for_influence = preprocess_adj_new(adj_A_processed)  # I - W_adj^T

        # z_nodes: [batch, num_nodes, z_dims]
        # Wa: [z_dims]
        # (z_nodes + self.Wa) -> broadcast Wa
        # torch.matmul(adj_A_for_influence, z_nodes + self.Wa) needs care with batching.
        # adj_A_for_influence is [num_nodes, num_nodes]
        # We need to operate on each sample in batch:
        # For each sample s: adj_A_for_influence @ (z_nodes[s] + self.Wa)

        # Corrected matrix multiplication for batch:
        # (z_nodes + self.Wa) has shape [batch, num_nodes, z_dims]
        # We want to left-multiply each [num_nodes, z_dims] slice by adj_A_for_influence [num_nodes, num_nodes]
        # Result should be [batch, num_nodes, z_dims]
        # This is equivalent to ( (z_nodes[s] + self.Wa)^T @ adj_A_for_influence^T )^T
        # Or, more simply, if z_nodes_plus_wa = z_nodes + self.Wa:
        # influenced_z = torch.einsum('jk,bkl->bjl', adj_A_for_influence, z_nodes_plus_wa)
        # Or, by reshaping z_nodes_plus_wa to [batch*num_nodes, z_dims] then matmul with something, then reshape back.
        # Or, loop (not efficient):
        # batch_influenced_z = []
        # for i in range(inputs.size(0)): # Batch
        #     batch_influenced_z.append(torch.matmul(adj_A_for_influence, z_nodes[i] + self.Wa))
        # influenced_z = torch.stack(batch_influenced_z)

        # The original modules.py had simpler matmul, implying x was different.
        # Let's re-check old modules.py 'x' for MLPEncoder
        # Old 'inputs' was n_in = num_nodes * data_dim, then reshaped.
        # Here 'inputs' is [batch, nodes, x_dims].
        # 'x' in original was [batch, num_nodes*z_dims_per_node_if_any_else_n_out_directly]
        # The 'logits' in original modules.py for MLPEncoder was:
        # logits = torch.matmul(adj_A_for_z, x_reshaped_for_graph_conv + self.Wa) - self.Wa
        # where x_reshaped_for_graph_conv might have been [batch, num_nodes, n_out_of_fc2]
        # if fc2 output n_out was global.
        # If fc2 output n_out (z_dims) is *per node latent representation*, then z_nodes is correct.

        # Assuming z_nodes are the latent variables Z_i for each node i.
        # The "logits" here might be a term used loosely, maybe for the prior p(Z) or something.
        # For DAG-GNN, the encoder usually produces Z, and decoder reconstructs X from Z and A.
        # The `kl_gaussian_sem(z_nodes)` suggests z_nodes are indeed the latent variables.
        # The term `logits` here is not directly used in loss in _train_epoch apart from `z` being derived.
        # The important return values for loss are `z_nodes` (as `z` in `_train_epoch`) and `self.adj_A`.

        # The example code from paper for continuous encoder does:
        # H1 = relu(XW1+b1), Z = H1W2+b2. This is what we have for z_nodes.
        # This Z is then used in decoder.
        # The term 'logits' from original module might be misnomer or related to an edge prediction variant not used.
        # We will return z_nodes as the primary latent variable output.
        # The original code structure was a bit convoluted. Let's simplify the return signature for clarity.

        # Returns:
        # z_nodes: latent variables [batch, num_nodes, z_dims]
        # adj_A_raw: learnable parameter A [num_nodes, num_nodes] (this is A_raw)
        # adj_A_processed: sinh(3*adj_A_raw) [num_nodes, num_nodes] (this is W_adj)

        # The other returned items (adj_A_for_z, self.z, self.z_positive, self.Wa) from original encoder
        # were not directly used in the provided _train_epoch loss for VAE part.
        # adj_A_for_z (I - W_adj^T) is used in decoder.

        return z_nodes, self.adj_A, adj_A_processed


class MLPDEncoder(nn.Module):  # For discrete data
    def __init__(self, n_in, n_hid, n_out, num_classes, adj_A, batch_size, do_prob=0., factor=True,
                 tol=0.1):  # adj_A is numpy array
        super(MLPDEncoder, self).__init__()

        # Directly use the passed numpy array for the nn.Parameter
        self.adj_A = nn.Parameter(torch.from_numpy(adj_A).double())  # Changed this line

        self.embed = nn.Embedding(num_classes, n_hid)  # Embedding dim = n_hid
        self.fc1 = nn.Linear(n_hid, n_hid, bias=True)  # From embedded dim to hidden
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)  # From hidden to latent dim (n_out = z_dims)

        # Wa and alpha seem specific to original NoTears/GAE-like formulation not directly used in VAE part of DAG-GNN loss.
        # For DAG-GNN VAE, we primarily need Z and A.
        # self.Wa = nn.Parameter(torch.tensor(0.0, dtype=torch.double), requires_grad=True) # Original
        # self.alpha = nn.Parameter(Variable(torch.ones(adj_A.shape[0], n_out, dtype=torch.double) / n_out, requires_grad=True)) # Original

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        nn.init.xavier_normal_(self.embed.weight.data)

    def forward(self, inputs, rel_rec, rel_send):  # inputs is LongTensor [batch, nodes]
        if inputs.dim() == 3 and inputs.size(2) == 1:  # Ensure inputs are [batch, nodes]
            inputs = inputs.squeeze(-1)
        if not inputs.dtype == torch.long:
            inputs = inputs.long()

        embedded_input = self.embed(inputs)  # Output: [batch, nodes, n_hid_embed]

        h1 = F.relu(self.fc1(embedded_input))  # Output: [batch, nodes, n_hid_fc1]
        z_nodes = self.fc2(h1)  # Output: [batch, nodes, n_out=z_dims] -> These are Z_i

        adj_A_processed = torch.sinh(3. * self.adj_A)  # W_adj

        # Original MLPDEncoder also returned adj_A_for_z, self.z, self.z_positive, self.Wa, alpha_prob
        # which are not directly used in the VAE loss based on _train_epoch.
        # For simplicity, returning similar to MLPEncoder.
        return z_nodes, self.adj_A, adj_A_processed


# MLPDecoder needs to be consistent with what encoder returns and what loss needs
# The original decoder took (inputs, input_z, n_in_node, ..., origin_A, ...)
# origin_A was the learnable parameter matrix (A_raw)
# input_z was z_encoded

class MLPDecoder(nn.Module):
    def __init__(self, n_in_node, n_in_z, n_out, data_variable_size, batch_size, n_hid, do_prob=0., **kwargs):
        super(MLPDecoder, self).__init__()
        # n_in_z is z_dims (latent dimension per node)
        # n_out is x_dims (feature_dim_per_node for continuous, num_classes for discrete)
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias=True)

        # Wa was used in original decoder: mat_z = torch.matmul(adj_A_inv, input_z + Wa) - Wa
        # This implies Wa should be of size n_in_z (z_dims).
        # It needs to be a learnable parameter if used.
        # Let's make it part of the decoder if this structure is kept.
        self.Wa_decoder = nn.Parameter(torch.zeros(n_in_z, dtype=torch.double), requires_grad=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: m.bias.data.fill_(0.0)

    def forward(self, target_data_for_shape, z_encoded, adj_A_raw):
        # target_data_for_shape: used to get batch_size, num_nodes if needed. Or pass num_nodes.
        # z_encoded: [batch, num_nodes, z_dims] from encoder
        # adj_A_raw: learnable parameter A [num_nodes, num_nodes]

        adj_A_processed = torch.sinh(3. * adj_A_raw)  # This is W_adj
        adj_A_inv = preprocess_adj_new1(adj_A_processed)  # (I - W_adj^T)^-1

        # Matmul: (I - W_adj^T)^-1 @ (Z + Wa_decoder) - Wa_decoder
        # z_encoded_plus_Wa = z_encoded + self.Wa_decoder # Broadcasting Wa
        # For batched matmul:
        # Each sample: mat_z_s = adj_A_inv @ (z_encoded[s] + self.Wa_decoder) - self.Wa_decoder

        batch_size = z_encoded.size(0)
        num_nodes = z_encoded.size(1)
        z_dims = z_encoded.size(2)

        # Unsqueeze Wa_decoder to be [1, 1, z_dims] for broadcasting over batch and nodes
        wa_broadcastable = self.Wa_decoder.unsqueeze(0).unsqueeze(0)

        z_plus_wa = z_encoded + wa_broadcastable  # [batch, num_nodes, z_dims]

        # Perform batched matrix multiplication
        # adj_A_inv is [num_nodes, num_nodes]
        # z_plus_wa is [batch, num_nodes, z_dims]
        # We want result [batch, num_nodes, z_dims]
        # einsum approach: 'jk,bkl->bjl' where j=num_nodes, k=num_nodes, l=z_dims
        mat_z_intermediate = torch.einsum('ji,bjd->bid', adj_A_inv, z_plus_wa)
        # Note: einsum 'ji,bjd->bid' means (adj_A_inv^T @ z_plus_wa^T)^T if we map indices correctly.
        # If adj_A_inv is (I-A^T)^-1, then we multiply by this.
        # Paper uses (I-A)^-1 * Z. If A_ij is edge j->i, then A is the matrix.
        # Here adj_A_inv is (I - W_adj^T)^-1 where W_adj = sinh(3*A_raw)
        # Let's assume A_raw[j,i] means edge i->j. Then W_adj[j,i] means edge i->j.
        # W_adj^T [i,j] means edge i->j.
        # So adj_A_inv is (I - "matrix where entry (r,c) means edge c->r")^-1

        # Let's follow the structure from original modules.py more closely for mat_z:
        # x = torch.matmul(adj_A_inv, input_z+self.Wa) - self.Wa (from their code for linear GAE)
        # This implies input_z was perhaps [nodes, features] not batched.
        # For batched version:
        mat_z_terms = []
        for i in range(batch_size):
            term = torch.matmul(adj_A_inv, z_encoded[i] + self.Wa_decoder) - self.Wa_decoder
            mat_z_terms.append(term)
        mat_z = torch.stack(mat_z_terms)  # Shape: [batch, num_nodes, z_dims]

        h_reconstruct = F.relu(
            self.out_fc1(mat_z))  # Input: [batch, num_nodes, z_dims] -> Output: [batch, num_nodes, n_hid]
        x_reconstructed = self.out_fc2(
            h_reconstruct)  # Output: [batch, num_nodes, n_out] (n_out is x_dims / num_classes)

        # For discrete, n_out is num_classes (logits). For continuous, n_out is data_feature_dim.
        # The _train_epoch will apply nll_categorical (needs logits) or nll_gaussian.
        return x_reconstructed  # mat_z also returned in original but not used in loss.
