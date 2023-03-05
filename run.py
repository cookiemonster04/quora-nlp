import torch
import model_utils
import train_utils
import argparse
from datasets import CharLabeledDataset
from pretrained_embed import GloVeEmbedding

argp = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
argp.add_argument('-e', '--epochs', dest='epochs', type=int)
argp.add_argument('-r', '--reading_params_path', dest='load_path')
argp.add_argument('-w', '--writing_params_path', dest='write_path')
argp.add_argument('-o', '--outputs_path', dest='out_path')
argp.add_argument('-s', '--save_interval', dest='n_save')
argp.add_argument('-b', '--batch_size', dest='batch_size')
args = argp.parse_args()

import logging
logger = logging.getLogger(__name__)

def main():
    logger.info('Processing train dataset...')
    train_data = CharLabeledDataset('train')
    dev_data = CharLabeledDataset('dev')
    logger.info('Finished processing dataset')
    config = train_utils.TrainerConfig()
    config.modify(vars(args))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    if config.model_type == 'word':
        embeddings = GloVeEmbedding()
        model = model_utils.WordLSTM(embeddings)
    else:
        model = model_utils.BaselineLSTM()
    model = model.to(device)
    trainer = train_utils.Trainer(config)
    trainer.train(model, train_data, dev_data, device, config)

if __name__ == '__main__':
    main()