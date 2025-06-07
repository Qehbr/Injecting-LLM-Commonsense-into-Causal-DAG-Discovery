import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import preprocess_adj_new1


class MLPEncoder(nn.Module):
    """
    MLP-based encoder for continuous data.

    This encoder maps input features to a latent space using a Multi-Layer
    Perceptron (MLP) and a structural equation-based transformation involving
    a learnable adjacency matrix.

    Attributes
    ----------
    adj_A : torch.nn.Parameter
        The learnable adjacency matrix of the graph.
    fc1 : torch.nn.Linear
        The first fully connected layer.
    fc2 : torch.nn.Linear
        The second fully connected layer.
    Wa : torch.nn.Parameter
        A learnable weight parameter used in the structural transformation.

    """

    def __init__(self, n_xdims, n_hid, n_out, adj_A):
        """
            Initializes the MLPEncoder module.

        Parameters
        ----------
        n_xdims : int
            The number of input dimensions (features).
        n_hid : int
            The number of hidden units in the MLP.
        n_out : int
            The number of output dimensions (latent space).
        adj_A : numpy.ndarray
            The initial adjacency matrix

        """
        super(MLPEncoder, self).__init__()
        self.adj_A = nn.Parameter(torch.from_numpy(adj_A).double())
        self.fc1 = nn.Linear(n_xdims, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.Wa = nn.Parameter(torch.zeros(n_out).double(), requires_grad=True)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights of the linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, inputs):
        """
        Defines the forward pass of the encoder.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor of shape `(batch_size, num_nodes, num_features)`.

        Returns
        -------
        z_nodes_final : torch.Tensor
            The encoded latent representations of shape `(batch_size, num_nodes, n_out)`.
        adj_A : torch.Tensor
            The raw learnable adjacency matrix.
        adj_A_processed : torch.Tensor
            The processed adjacency matrix after applying `sinh`.
        Wa : torch.Tensor
            The learnable Wa parameter.

        """
        # MLP(X) part
        processed_inputs = inputs.double()
        h1 = F.relu(self.fc1(processed_inputs))
        mlp_x_output = self.fc2(h1)

        adj_A_processed = torch.sinh(3. * self.adj_A)

        # Structural factor (I - A^T) from paper's encoder formula
        identity = torch.eye(adj_A_processed.shape[0],
                             device=adj_A_processed.device,
                             dtype=adj_A_processed.dtype)
        structural_factor = identity - adj_A_processed.T

        # Apply Wa: Z = (I - A^T)(MLP(X) + Wa) - Wa
        mlp_x_plus_wa = mlp_x_output + self.Wa
        transformed_mlp_x = torch.einsum('ij,bjk->bik', structural_factor, mlp_x_plus_wa)
        z_nodes_final = transformed_mlp_x - self.Wa

        return z_nodes_final, self.adj_A, adj_A_processed, self.Wa


class MLPDEncoder(nn.Module):
    """
    MLP-based encoder for discrete data.

    This encoder handles discrete (categorical) inputs by first embedding them
    into a continuous space before applying an MLP and a structural
    equation-based transformation.

    Attributes
    ----------
    adj_A : torch.nn.Parameter
        The learnable adjacency matrix of the graph.
    embed : torch.nn.Embedding
        The embedding layer for discrete input features.
    fc1 : torch.nn.Linear
        The first fully connected layer.
    fc2 : torch.nn.Linear
        The second fully connected layer.
    Wa : torch.nn.Parameter
        A learnable weight parameter used in the structural transformation.

    """

    def __init__(self, n_hid, n_out, num_classes, adj_A):
        """
        Initializes the MLPDEncoder module.

        Parameters
        ----------
        n_hid : int
            The number of hidden units and embedding dimension.
        n_out : int
            The number of output dimensions (latent space).
        num_classes : int
            The number of unique classes for the embedding layer.
        adj_A : numpy.ndarray
            The initial adjacency matrix.

        """
        super(MLPDEncoder, self).__init__()
        self.adj_A = nn.Parameter(torch.from_numpy(adj_A).double())
        self.embed = nn.Embedding(num_classes, n_hid)
        self.fc1 = nn.Linear(n_hid, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.Wa = nn.Parameter(torch.zeros(n_out).double(), requires_grad=True)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights of the linear and embedding layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        nn.init.xavier_normal_(self.embed.weight.data)

    def forward(self, inputs):
        """
        Defines the forward pass for the discrete data encoder.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor of class indices, shape `(batch_size, num_nodes)`.

        Returns
        -------
        z_nodes_final : torch.Tensor
            The encoded latent representations of shape `(batch_size, num_nodes, n_out)`.
        adj_A : torch.Tensor
            The raw learnable adjacency matrix.
        adj_A_processed : torch.Tensor
            The processed adjacency matrix after applying `sinh`.
        Wa : torch.Tensor
            The learnable Wa parameter.

        """
        if inputs.dim() == 3 and inputs.size(2) == 1:
            inputs = inputs.squeeze(-1)
        if not inputs.dtype == torch.long:
            inputs = inputs.long()

        # MLP(X) part
        embedded_input = self.embed(inputs)
        h1 = F.relu(self.fc1(embedded_input))
        mlp_x_output = self.fc2(h1)

        adj_A_processed = torch.sinh(3. * self.adj_A)

        # Structural factor (I - A^T)
        identity = torch.eye(adj_A_processed.shape[0],
                             device=adj_A_processed.device,
                             dtype=adj_A_processed.dtype)
        structural_factor = identity - adj_A_processed.T

        # Apply Wa: Z = (I - A^T)(MLP(X) + Wa) - Wa
        mlp_x_plus_wa = mlp_x_output + self.Wa
        transformed_mlp_x = torch.einsum('ij,bjk->bik', structural_factor, mlp_x_plus_wa)
        z_nodes_final = transformed_mlp_x - self.Wa

        return z_nodes_final, self.adj_A, adj_A_processed, self.Wa


class MLPDecoder(nn.Module):
    """
    MLP-based decoder.

    This decoder reconstructs the original data from the latent space by
    first applying an inverse structural transformation and then passing the
    result through an MLP.

    Attributes
    ----------
    out_fc1 : torch.nn.Linear
        The first fully connected layer for reconstruction.
    out_fc2 : torch.nn.Linear
        The second fully connected layer for reconstruction.

    """

    def __init__(self, n_in_z, n_out, n_hid):
        """
        Initializes the MLPDecoder module.

        Parameters
        ----------
        n_in_z : int
            The number of input dimensions (latent space).
        n_out : int
            The number of output dimensions (original feature space).
        n_hid : int
            The number of hidden units in the MLP.

        """
        super(MLPDecoder, self).__init__()
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias=True)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights of the linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: m.bias.data.fill_(0.0)

    def forward(self, z_encoded, adj_A_raw, Wa_from_encoder):
        """
        Defines the forward pass of the decoder.

        Parameters
        ----------
        z_encoded : torch.Tensor
            The latent tensor from the encoder of shape `(batch_size, num_nodes, n_in_z)`.
        adj_A_raw : torch.Tensor
            The raw adjacency matrix from the encoder.
        Wa_from_encoder : torch.Tensor
            The Wa parameter from the encoder.

        Returns
        -------
        x_reconstructed : torch.Tensor
            The reconstructed data tensor of shape `(batch_size, num_nodes, n_out)`.

        """
        adj_A_processed = torch.sinh(3. * adj_A_raw)
        adj_A_inv = preprocess_adj_new1(adj_A_processed)

        # Paper form: (I - A^T)^-1 * Z
        z_plus_wa = z_encoded + Wa_from_encoder

        transformed_z = torch.einsum('ij,bjk->bik', adj_A_inv, z_plus_wa)

        mat_z = transformed_z - Wa_from_encoder

        # MLP part
        h_reconstruct = F.relu(self.out_fc1(mat_z))
        x_reconstructed = self.out_fc2(h_reconstruct)
        return x_reconstructed
