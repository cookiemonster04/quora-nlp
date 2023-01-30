import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    """ A simple Character-level LSTM """
    vocab_size = 256
    embed_dim = 128
    hidden_dim = 256
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.vocab_size-1, max_norm=True)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2*self.hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.loss_fn = nn.BCELoss()
        
    def forward(self, x):
        x = self.embed(x)
        x = self.lstm(x)
        x = torch.select(x[0], 1, -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def load(self, state_dict):
        self.load_state_dict(state_dict)