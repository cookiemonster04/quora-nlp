import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    """ A simple Character-level LSTM """
    vocab_size = 256
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.vocab_size-1, max_norm=True)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=0.1, batch_first=False, bidirectional=True)
        self.fc1 = nn.Linear(2*self.hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.loss_fn = nn.BCELoss()
        
    def forward(self, x, x_len): # x: 32 x max_len
        batch_size = x.size(dim=0)
        x = self.embed(x) # 32 x max_len x embed_dim
        x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(1, 0), x_len, enforce_sorted=False) # max_len x 32 x embed_dim
        _, (x, _) = self.lstm(x) # batch_first = False -> seq_len, batch_size, input_size (hidden_dim)
        x = (x.view(self.num_layers, 2, batch_size, self.hidden_dim)[-1]
             .transpose(1, 0).contiguous().view(batch_size, -1))
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def load(self, state_dict):
        self.load_state_dict(state_dict)
        
class WordLSTM(nn.Module):
    """ A word level LSTM, using pretrained word embeddings of some sort """
    embed_dim = 300
    hidden_dim = 256
    def __init__(self, embeddings, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.embeddings = embeddings
        self.load_embedding()
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
        
    def load_embedding(self):
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(self.embeddings.weight_matrix))
        # embedding_matrix = 
        # self.embed.load_state_dict({
        #     'weight': 
        # })