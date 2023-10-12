import torch
import torch.nn as nn
import torch
import torch.nn as nn
from typing import List, Tuple


from torch.nn.modules.batchnorm import BatchNorm1d


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
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.7),
            nn.Linear(hidden_size, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.ones(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.ones(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.linear(out)
        return self.activation(out)


class LSTMAtt(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMAtt, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size, hidden_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.7),
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.ones(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.ones(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x, (h0, c0))
        out = self.attention(out, out, out)
        out = out[:, -1, :]
        out = self.classifier(out)
        return self.activation(out)


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
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
            nn.Dropout(0.7),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.7),
            nn.Linear(128, num_classes),
        )
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        return self.act(self.classifier(self.features(x)))
