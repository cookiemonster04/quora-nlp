import pandas as pd
import numpy as np
import torch

class GloVeEmbedding():
    def __init__(self, config):
        self.num_embeddings, self.embedding_dim = 2196017, 300
        self.weight_matrix = np.ndarray((self.num_embeddings+1, self.embedding_dim), dtype=np.float32)
        self.vocab = []
        with open(config.embed_path, 'r', encoding='utf-8', buffering=8196) as f:
            for idx, line in enumerate(f):
                values = line.split(' ')
                self.vocab.append(values[0])
                self.weight_matrix[idx] = np.asarray(values[1:], "float32")
        self.weight_matrix[self.num_embeddings] = np.mean(self.weight_matrix[:self.num_embeddings], axis=0)
        # print("reading csv")
        # if config.embed_chunk == 0:
        #     self.data = pd.read_csv(config.embed_path, header=None, sep=' ')
        #     print("finished reading")
        #     self.num_embeddings, self.embedding_dim = self.data.shape
        #     self.vocab = self.data[0]
        #     print(self.vocab)
        #     self.weight_matrix = self.data[1:].values
        # else:
        #     self.vocab = []
        #     with pd.read_csv(config.embed_path, header=None, sep=' ', chunksize=config.embed_chunk, quoting=3) as reader:
        #         self.num_embeddings, self.embedding_dim = 2036775, 300
        #         self.weight_matrix = np.ndarray((self.num_embeddings, self.embedding_dim), dtype=np.float32)
        #         for idx, chunk in enumerate(reader):
        #             print(chunk.iloc[:15])
        #             self.vocab.extend(chunk[0])
        #             self.weight_matrix[idx*config.embed_chunk:idx*config.embed_chunk+chunk.shape[0]] = chunk[1:].values
        self.stoi_dict = {val: index for index, val in enumerate(self.vocab)}
    def stoi(self, val):
        return self.stoi_dict[val] if val in self.stoi_dict else len(self.vocab)