import torch
import pandas as pd
from collections import Counter
from nltk.tokenize import wordpunct_tokenize

class CharLabeledDataset(torch.utils.data.Dataset):
    """
    Training data loader
    """
    MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
    PAD_CHAR = u"\u25A1" # the empty square character, for pad
    gen_vocab = True
    use_pad = True
    pad_len = 256

    def __init__(self, split, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        assert split == 'train' or split == 'dev'
        self.df = pd.read_csv(f'./data/{split}_split.csv')
        if self.gen_vocab:
            vocab_count = Counter()
            for entry in self.df['question_text']:
                vocab_count.update(Counter(entry))
            vocab_list = list(vocab_count.most_common(254))
            assert self.MASK_CHAR not in vocab_list
            assert self.PAD_CHAR not in vocab_list
            vocab_list.append((self.MASK_CHAR, 0))
            vocab_list.append((self.PAD_CHAR, 0))
            self.itos = [v for (v, _) in vocab_list]
            self.stoi = {v: i for i,(v,_) in enumerate(vocab_list)}
        self.qlens = [len(q) for q in self.df['question_text']]
        self.max_qlen = max(self.qlens)

    def __getitem__(self, idx):
        x_raw = self.df.loc[idx, 'question_text']
        x_len = self.qlens[idx]
        y_raw = self.df.loc[idx, 'target']
        x = torch.concat((torch.tensor([self.stoi.get(x_char, self.stoi[self.MASK_CHAR]) for x_char in x_raw], dtype=torch.long), 
                         torch.zeros(self.max_qlen-x_len, dtype=torch.long)))
        y = torch.tensor(y_raw, dtype=torch.float32)
        return x, x_len, y

    def __len__(self):
        return len(self.df)
    
    def get_max_len(self):
        if hasattr(self, 'max_len'):
            return self.max_len
        self.max_len = max([len(item) for item in self.df['question_text']])
        return self.max_len

    def get_gtlens(self, bound):
        return sum([len(item) > bound for item in self.df['question_text']])

    def get_pos(self):
        if hasattr(self, 'num_pos'):
            return self.num_pos
        self.num_pos = sum(self.df['target'])
        return self.num_pos

class WordLabeledDataset(torch.utils.data.Dataset):
    """
    Training data loader
    """

    def __init__(self, split, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        assert split == 'train' or split == 'dev', 'invalid split'
        self.df = pd.read_csv(f'./data/{split}_split.csv')
        assert 'embed' in kwargs, 'missing embed'
        self.input_seq = [torch.tensor([kwargs['embed'].stoi(token) for token in wordpunct_tokenize(question)]) 
            for question in self.df['question_text'].values]
        self.input_lens = torch.tensor([input.size() for input in self.input_seq])
        self.inputs = torch.nn.utils.rnn.pad_sequence(self.input_seq, batch_first=True)
        
    def __getitem__(self, idx):
        y_raw = self.df.loc[idx, 'target']
        x = self.inputs[idx]
        x_len = self.input_lens[idx]
        y = torch.tensor(y_raw, dtype=torch.float32)
        return x, x_len, y

    def __len__(self):
        return len(self.df)
    
    def get_max_len(self):
        if hasattr(self, 'max_len'):
            return self.max_len
        self.max_len = max([len(item) for item in self.df['question_text']])
        return self.max_len

    def get_gtlens(self, bound):
        return sum([len(item) > bound for item in self.df['question_text']])

    def get_pos(self):
        if hasattr(self, 'num_pos'):
            return self.num_pos
        self.num_pos = sum(self.df['target'])
        return self.num_pos
    