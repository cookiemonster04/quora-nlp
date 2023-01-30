import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler

from tqdm import tqdm
import datetime
import os

import logging
logger = logging.getLogger(__name__)

class TrainerConfig:
    epochs = 2
    batch_size = 32
    lr = 0.005
    grad_norm_clip = 1
    load_path = None
    write_path = "checkpoints"
    out_path = None # default stdout
    n_save = 100
    n_eval = 100
    def __init__(self, **kwargs):
        self.modify(**kwargs)
    def modify(self, arg_dict=None, **kwargs):
        if arg_dict is not None:
            for k,v in arg_dict.items():
                setattr(self, k, v)
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, config):
        self.config = config
    def train(self, model, train_data, val_data, device, config):
        writer = SummaryWriter()
        train_loader = DataLoader(train_data, batch_size=config.batch_size, 
                                  sampler=WeightedRandomSampler([1/train_data.dataset.get_pos() if i == 1 else 1/len(train_data) 
                                                                 for i in train_data.dataset.df['target'][train_data.indices]], num_samples=len(train_data)//10, replacement=True))
        if val_data is not None:
            val_loader = DataLoader(train_data, batch_size=config.batch_size,
                                    sampler=WeightedRandomSampler([1/val_data.dataset.get_pos() if i == 1 else 1/len(val_data) 
                                                                 for i in val_data.dataset.df['target'][train_data.indices]], num_samples=len(val_data)//10, replacement=True))
        optimizer = optim.Adam(model.parameters())
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for epoch in range(config.epochs):
            for it, (x, y) in pbar:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = model.loss_fn(logits.squeeze(), y)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")
                if it % config.n_eval == 0:
                    writer.add_scalar("Train loss", loss)
            if (epoch+1) % config.n_save == 0:
                self.save_checkpoint(model)
        if val_data:
            logger.info("Evaluating Dev...")
            dev_loss, dev_acc, dev_f1 = self.evaluate(model, device, val_loader)
            writer.add_scalar("Dev loss", dev_loss)
            writer.add_scalar("Dev acc", dev_acc)
            writer.add_scalar("Dev F1", dev_f1)
                
    def f1(self, logits, labels):
        preds = torch.round(torch.cat(logits))
        ys = torch.cat(labels)
        true_pos = 0; false_pos = 0; false_neg = 0
        for i in range(len(ys)):
            if preds[i] == 1:
                if ys[i] == 1:
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                if ys[i] == 1:
                    false_neg += 1
        if true_pos == 0: # avoid divide by 0
            return 0
        precision = true_pos/(true_pos+false_pos); recall = true_pos/(true_pos+false_neg)
        return 2*precision*recall/(precision+recall)
    
    def accuracy(self, logits, labels):
        n_correct = sum(a == b for (a,b) in zip(torch.round(torch.cat(logits)), torch.cat(labels)))
        return n_correct.item() / len(labels)

    def evaluate(self, model, device, test_loader):
        with torch.no_grad():
            yps = []
            ys = []
            total_loss = 0
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for it, (x, y) in pbar:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                yps.append(logits)
                ys.append(y)
                loss = model.loss_fn(torch.squeeze(logits), y)
                total_loss += loss*len(x)
                pbar.set_description(f"iter {it} loss: {total_loss/len(ys)} acc: {self.accuracy(yps, ys)} f1: {self.f1(yps, ys)}")
            return total_loss/len(ys), self.accuracy(yps, ys), self.f1(yps, ys)

    def save_checkpoint(self, model):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            filename = datetime.now().strftime('%d-%m-%y-%H_%M_state.pt')
            torch.save(ckpt_model.state_dict(), os.path.join(self.config.ckpt_path, filename))