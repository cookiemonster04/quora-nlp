import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler

from tqdm import tqdm
import datetime
import os

import logging
logger = logging.getLogger(__name__)

class Score:
    def __init__(self, loss=0, true_pos=0, true_neg=0, false_pos=0, false_neg=0):
        self.total_loss = loss
        self.true_pos = true_pos
        self.true_neg = true_neg
        self.false_pos = false_pos
        self.false_neg = false_neg
    @staticmethod
    def build(loss, logits, labels):
        score = Score()
        score.total_loss += loss
        for (yp, y) in zip(logits, labels):
            if round(yp.item()) == 1:
                if y == 1:
                    score.true_pos += 1
                else:
                    score.false_pos += 1
            else:
                if y == 1:
                    score.false_neg += 1
                else:
                    score.true_neg += 1
        return score
    def __iadd__(self, arg):
        self.total_loss += arg.total_loss
        self.true_pos += arg.true_pos
        self.true_neg += arg.true_neg
        self.false_pos += arg.false_pos
        self.false_neg += arg.false_neg
        return self
    def __add__(self, arg):
        return Score(self.total_loss+arg.total_loss,
                     self.true_pos+arg.true_pos,
                     self.true_neg+arg.true_neg,
                     self.false_pos+arg.false_pos,
                     self.false_neg+arg.false_neg)
    def __isub__(self, arg):
        self.total_loss -= arg.total_loss
        self.true_pos -= arg.true_pos
        self.true_neg -= arg.true_neg
        self.false_pos -= arg.false_pos
        self.false_neg -= arg.false_neg
        return self
    def __sub__(self, arg):
        return Score(self.total_loss-arg.total_loss,
                     self.true_pos-arg.true_pos,
                     self.true_neg-arg.true_neg,
                     self.false_pos-arg.false_pos,
                     self.false_neg-arg.false_neg)
    def __len__(self):
        return self.true_pos+self.true_neg+self.false_pos+self.false_neg
    def f1(self):
        if self.true_pos == 0:
            return 0
        precision = self.true_pos/(self.true_pos+self.false_pos)
        recall = self.true_pos/(self.true_pos+self.false_neg)
        return 2*precision*recall/(precision+recall)
    def acc(self):
        return (self.true_pos+self.true_neg)/len(self)
    def loss(self):
        return self.total_loss/len(self)

class TrainerConfig:
    epochs = 50
    batch_size = 32
    lr = 0.005
    grad_norm_clip = 1
    load_path = None
    write_path = "checkpoints"
    out_path = None # default stdout
    n_save = 100
    n_eval_t = 100
    n_eval_d = 1
    avg_interval = 100
    model_type = 'char'
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
                                  sampler=WeightedRandomSampler([1/train_data.get_pos() if i == 1 else 1/len(train_data) 
                                                                 for i in train_data.df['target']], num_samples=len(train_data), replacement=True))
        # for i in range(500):
        #     print(train_loader.sampler.weights[i])
        if val_data is not None:
            val_loader = DataLoader(val_data, batch_size=config.batch_size)
        optimizer = optim.Adam(model.parameters())
        score = Score()
        rolling_f1 = [Score() for _ in range(config.avg_interval)]
        for epoch in range(config.epochs):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for it, (x, x_len, y) in pbar:
                x = x.to(device)
                y = y.to(device)
                logits = model(x, x_len)
                loss = model.loss_fn(logits.squeeze(), y)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cur_score = Score.build(loss, logits, y)
                score -= rolling_f1[epoch % config.avg_interval]
                score += cur_score
                rolling_f1[epoch % config.avg_interval] = cur_score
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f} acc {score.acc()} f1 {score.f1()}")
                if it % config.n_eval_t == 0:
                    writer.add_scalar("Train loss", score.loss())
                    writer.add_scalar("Train acc", score.acc())
                    writer.add_scalar("Train f1", score.f1())
            if (epoch+1) % config.n_save == 0:
                logger.info("Saving model...")
                self.save_checkpoint(model)
            if (epoch+1 == config.epochs or (epoch+1) % config.n_eval_d == 0) and val_data:
                logger.info("Evaluating Dev...")
                dev_score = self.evaluate(model, device, val_loader)
                writer.add_scalar("Dev loss", dev_score.loss())
                writer.add_scalar("Dev acc", dev_score.acc())
                writer.add_scalar("Dev F1", dev_score.f1())
    
    def evaluate(self, model, device, test_loader):
        score = Score()
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for it, (x, x_len, y) in pbar:
                x = x.to(device)
                y = y.to(device)
                logits = model(x, x_len)
                loss = model.loss_fn(torch.squeeze(logits), y)
                score += Score.build(loss, logits, y)
                pbar.set_description(f"iter {it} loss: {score.loss()} acc: {score.acc()} f1: {score.f1()}")
            return score

    def save_checkpoint(self, model):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            filename = datetime.now().strftime('%d-%m-%y-%H_%M_state.pt')
            torch.save(ckpt_model.state_dict(), os.path.join(self.config.ckpt_path, filename))