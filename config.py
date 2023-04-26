class Config:
    # path arg
    embed_path = './embeddings/glove.840B.300d/glove.840B.300d.txt'
    # embedding load config
    embed_chunk = 1000
    # train args
    epochs = 10
    batch_size = 32
    lr = 0.005
    grad_norm_clip = 1
    load_path = None
    ckpt_path = "./checkpoints"
    out_path = None # default stdout
    n_save = 1
    n_eval_t = 100
    n_eval_d = 1
    avg_interval = 100
    model_type = 'word'
    def __init__(self, **kwargs):
        self.modify(**kwargs)
    def modify(self, arg_dict=None, **kwargs):
        if arg_dict is not None:
            for k,v in arg_dict.items():
                setattr(self, k, v)
        for k,v in kwargs.items():
            setattr(self, k, v)
