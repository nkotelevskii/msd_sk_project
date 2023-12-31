import torch
import torch.nn as nn
import torch
import torch.nn as nn
from typing import List, Tuple


from torch.nn.modules.batchnorm import BatchNorm1d

from .cell_attn import LSTMWithInputCellAttention


def attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = query.shape[-1]
    coefficient_matrix = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(
        torch.tensor(d).float()
    )  # TODO
    softmax = nn.Softmax(dim=-1)
    attention_matrix = softmax(coefficient_matrix)  # TODO
    out = torch.matmul(attention_matrix, value)  # TODO
    return attention_matrix, out


class Attention(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_q: int,
        dim_k: int,
    ):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        self.A, out = attention(query, key, value)  # TODO

        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop(x)
        h0 = torch.ones(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.ones(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x, (h0, c0))
        #out = self.drop(out)
        out = out[:, -1, :]
        out = self.linear(out)
        return self.activation(out)


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.drop = nn.Dropout(0.1)
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop(x)
        x = x.transpose(-1, -2)
        return self.act(self.classifier(self.features(x)))


class LSTMPlus(nn.Module):
    def __init__(self, input_size, hidden_size, r, d_a, num_classes):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.r = r
        self.d_a = d_a
        self.lstm = LSTMWithInputCellAttention(input_size, hidden_size, r=20, d_a=50)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop(x)
        h0 = torch.ones(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.ones(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        #print(f"{out.shape=}")
        out = self.fc(out)
        return self.activation(out)


class Transformer(nn.Module):
    def __init__(self, num_inputs, embed_dim, num_heads, num_classes):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.embed = nn.Linear(num_inputs, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                                    #kdim=embed_dim, vdim=embed_dim, 
                                                    batch_first=True,)
                                                    #average_attn_weights=False)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * 24, num_classes)
        )
        self.activation = nn.Softmax(dim=1)
        self.A = None
    
    def forward(self, x):
        x = self.drop(x)
        x = self.embed(x)
        out, self.A = self.multihead_attn(x, x, x)
        #print(f"{out.shape=}, {out2.shape=}")
        return self.activation(self.fc(out))
