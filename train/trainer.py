import random
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from utils.parquet_reader import ParquetReader
from utils.InitialModel import InitialModel

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.seed = config.training.seed
        self.lr = config.training.lr
        self.epochs = config.training.epochs
        self.gradient_clip = config.training.gradient_clip
        self.optimizer = config.training.optimizer
        self.if_warmup = config.training.if_warmup
        self.warmup_epochs = 5
        self.warmup_lr = 1e-6
        self.warmup_lr_schedule = "linear"

        self.setup_seed()

    def setup_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def epoch_train(self, model, batch):

        criterion = nn.CrossEntropyLoss()

        images = batch['image'].to(self.device)
        instructions = batch['instruction'].to(self.device)
        action_values = batch['action_value'].to(self.device)

        self.optimizer.zero_grad()

        action_logits = model(images, instructions)
        loss = criterion(action_logits, action_values)
        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        self.optimizer.step()

        return loss.item()
        

    def epoch_evaluate(self):
        pass
    
    def train(self, data):
        logger.info(f"Training on {self.device}")

        # process data
        data_reader = ParquetReader(self.config)
        train_loader = data_reader.read_parquet(split='train')
        n_batches = len(train_loader)

        # create/load model
        model = InitialModel(self.config)
        try:
            if self.config.model.continue_train:
                logger.info(f"Continue training, loading model from {self.config.model.pretrained_path}")
                model.load_model(self.config.model.pretrained_path)
        except Exception as e:
            logger.warning(f"{e}: Failed to load model from {self.config.model.pretrained_path}, starting from scratch")

        for epoch in range(self.epochs):
            if self.if_warmup and epoch < self.warmup_epochs:
                self.warmup_lr = self.warmup_lr * (1 - epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.warmup_lr

            progress_bar = tqdm(
                total=n_batches,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                position=0,
                leave=True,
            )
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                loss = self.epoch_train(model, batch)
                total_loss += loss
                avg_loss = total_loss / (batch_idx + 1)

                progress_bar.set_postfix(
                    loss=f'{loss:.4f}',
                    avg_loss=f'{avg_loss:.4f}'
                    )
                progress_bar.update(1)

            progress_bar.close()
            logger.info(f"Epoch {epoch+1}/{self.epochs} loss: {total_loss/n_batches}")

    def evaluate(self):
        pass