import pandas as pd

class GloVeEmbedding():
    def __init__(self):
        self.data = pd.read_csv('./embeddings/glove.840B.300d/glove.840B.300d.txt', header=None)
        self.num_embeddings, self.embedding_dim = self.data.size()
        self.vocab = self.data[0]
        self.weight_matrix = self.data[1:].values
        self.stoi = {val: index for index, val in enumerate(self.vocab)}
        