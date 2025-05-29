# project/algorithms/dag_gnn/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import my_softmax, preprocess_adj_new, preprocess_adj_new1


class MLPEncoder(nn.Module):
    """Encoder for continuous data."""

    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol=0.1):
        super(MLPEncoder, self).__init__()

        init_adj_A = torch.from_numpy(adj_A).double()
        nn.init.uniform_(init_adj_A, a=-1.0, b=1.0)
        self.adj_A = nn.Parameter(Variable(init_adj_A, requires_grad=True))

        self.Wa = nn.Parameter(torch.zeros(n_out, dtype=torch.double), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)

        self.z = nn.Parameter(torch.tensor(tol, dtype=torch.double))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A), dtype=torch.double))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)

    def forward(self, inputs, rel_rec, rel_send):
        adj_A_processed = torch.sinh(3. * self.adj_A)
        adj_A_for_z = preprocess_adj_new(adj_A_processed)

        h1 = F.relu(self.fc1(inputs))
        x = self.fc2(h1)

        logits = torch.matmul(adj_A_for_z, x + self.Wa) - self.Wa

        return x, logits, adj_A_processed, adj_A_for_z, self.z, self.z_positive, self.adj_A, self.Wa


class MLPDEncoder(nn.Module):
    """Encoder for discrete data with an embedding layer."""

    def __init__(self, n_in, n_hid, n_out, num_classes, adj_A, batch_size, do_prob=0., factor=True, tol=0.1):
        super(MLPDEncoder, self).__init__()

        init_adj_A = torch.from_numpy(adj_A).double()
        nn.init.uniform_(init_adj_A, a=-1.0, b=1.0)
        self.adj_A = nn.Parameter(Variable(init_adj_A, requires_grad=True))

        self.embed = nn.Embedding(num_classes, n_hid)
        self.fc1 = nn.Linear(n_hid, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.Wa = nn.Parameter(torch.tensor(0.0, dtype=torch.double), requires_grad=True)
        self.alpha = nn.Parameter(
            Variable(torch.ones(adj_A.shape[0], n_out, dtype=torch.double) / n_out, requires_grad=True))

        self.z = nn.Parameter(torch.tensor(tol, dtype=torch.double))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A), dtype=torch.double))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, inputs, rel_rec, rel_send):
        if inputs.dim() == 3 and inputs.size(2) == 1:
            inputs = inputs.squeeze(-1)

        embedded_input = self.embed(inputs)  # Input should be LongTensor

        h1 = F.relu(self.fc1(embedded_input))
        x = self.fc2(h1)

        adj_A_processed = torch.sinh(3. * self.adj_A)
        adj_A_for_z = preprocess_adj_new(adj_A_processed)

        logits = torch.matmul(adj_A_for_z, x + self.Wa) - self.Wa

        alpha_prob = my_softmax(self.alpha, axis=-1)

        return x, logits, adj_A_processed, adj_A_for_z, self.z, self.z_positive, self.adj_A, self.Wa, alpha_prob


class MLPDecoder(nn.Module):
    """General MLP decoder for both continuous and discrete data."""

    def __init__(self, n_in_node, n_in_z, n_out, data_variable_size, batch_size, n_hid, do_prob=0., **kwargs):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.softmax_out = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):
        adj_A_inv = preprocess_adj_new1(origin_A)

        mat_z = torch.matmul(adj_A_inv, input_z + Wa) - Wa

        h3 = F.relu(self.out_fc1(mat_z))
        output = self.out_fc2(h3)

        # NOTE: For discrete data, the loss function (cross-entropy) will apply softmax internally.
        # For continuous data, this output is the direct reconstruction.
        return mat_z, output, adj_A_tilt