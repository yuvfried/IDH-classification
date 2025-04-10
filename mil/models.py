import torch
import torch.nn as nn


# https://github.com/mahmoodlab/PANTHER/blob/main/src/mil_models
class AttentionBlock(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        attention_heads: number of attention branches
    """

    def __init__(self, L, D, K=1, dropout=0.0, gated=False):
        super().__init__()
        # self.in_features_size = L
        # self.hidden_features_size = D
        # self.attention_branches = K

        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.gate = None
        if gated:
            self.gate = nn.Sequential(
                nn.Linear(L, D),
                nn.Sigmoid(),
                nn.Dropout(dropout)
            )

        self.head = nn.Linear(D, K)

    def forward(self, H):
        # B: batch size (typically equals to 1)
        # N: bag size (num of samples per bag)
        # D: hidden_layer_size
        # K: num of attention branches
        # H of shape B x N x D
        A = self.attention(H)  # B x N x D
        if self.gate is not None:
            A_U = self.gate(H)  # B x N x D
            A = A.mul(A_U)  # B x N x D
        A = self.head(A)  # B x N x K
        return A

class ABMIL(nn.Module):
    def __init__(
            self,
            L: int,
            D: int,
            mlp_dim: int,
            num_classes: int = 1,
            temperature: float = 1.0,
            K: int = 1,
            dropout: float = 0.0,
            l2: float = None,
            gated: bool = True,
    ):
        super().__init__()

        # pre attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(L, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Attention block
        self.gated_attention = AttentionBlock(mlp_dim, D, K=K, dropout=dropout, gated=gated)

        # temperature of softmax of tile's attention scores
        self.t = temperature

        # Classifier takes K branches attention encodings and outputs a prediction
        self.classification_head = nn.Linear(mlp_dim * K, num_classes)

        # Add sparsity regularization to attention weights
        self.l2 = l2

    def classify(self, H, A):
        # H: N x mlp_dim
        # A: N x K
        A = torch.transpose(A, 1, 0)  # KxN
        A = nn.functional.softmax(A / self.t, dim=1)  # softmax over N
        M = torch.mm(A, H)  # K x mlp_dim
        logits = self.classification_head(M)
        return logits

    def forward(self, H):
        # H: 1xNxL
        H = H.squeeze(0)    # NxL
        H = self.mlp(H) # N x mlp_dim

        A = self.gated_attention(H) # N x K

        logits = self.classify(H, A)

        reg_score = None if self.l2 is None else self.l2 * torch.sum(torch.square(A))

        return {
            'logits': logits,
            'A': A,
            'l2_reg': reg_score,
        }