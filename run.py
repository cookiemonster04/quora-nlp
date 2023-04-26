import torch
import model_utils
import train_utils
from config import Config
import argparse
from datasets import CharLabeledDataset, WordLabeledDataset
from pretrained_embed import GloVeEmbedding

argp = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
argp.add_argument('-n', '--epochs', dest='epochs', type=int)
argp.add_argument('-r', '--reading_params_path', dest='load_path')
argp.add_argument('-w', '--writing_params_path', dest='write_path')
argp.add_argument('-o', '--outputs_path', dest='out_path')
argp.add_argument('-s', '--save_interval', dest='n_save')
argp.add_argument('-b', '--batch_size', dest='batch_size')
argp.add_argument('-e', '--embed_path', dest='embed_path')
args = argp.parse_args()

import logging
logger = logging.getLogger(__name__)
import constants
import os
from datetime import datetime

def main():
    config = Config()
    config.start_time = datetime.now().strftime("%Y_%b_%d-%H_%M_%S")
    if hasattr(constants, "CKPT_PATH") and constants.CKPT_PATH != "":
        config.ckpt_path = constants.CKPT_PATH
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    if config.model_type == 'word':
        embeddings = GloVeEmbedding(config)
        model = model_utils.WordLSTM(embeddings)
        if hasattr(constants, 'LOAD_FILE') and constants.LOAD_FILE != "":
            model.load_state_dict(torch.load(os.path.join(constants.CKPT_PATH, constants.LOAD_FILE)))
        logger.info('Processing train dataset...')
        train_data = WordLabeledDataset('train', embed=embeddings)
        dev_data = WordLabeledDataset('dev', embed=embeddings)
        logger.info('Finished processing dataset')
    else:
        model = model_utils.BaselineLSTM()
        logger.info('Processing train dataset...')
        train_data = CharLabeledDataset('train')
        dev_data = CharLabeledDataset('dev')
        logger.info('Finished processing dataset')
    model = model.to(device)
    trainer = train_utils.Trainer(config)
    trainer.train(model, train_data, dev_data, device, config)

if __name__ == '__main__':
    main()