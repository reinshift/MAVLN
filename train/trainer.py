import random
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from utils.parquet_reader import ParquetReader
from utils.InitialModel import InitialModel
import swanlab
import os
import time

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.seed = config.training.seed
        self.lr = float(config.training.lr)
        self.epochs = config.training.epochs
        self.gradient_clip = config.training.gradient_clip
        self.optimizer = config.training.optimizer
        self.if_warmup = config.training.if_warmup
        self.warmup_epochs = 3
        self.warmup_start_lr = self.lr / 1000
        self.batch_size = config.data.batch_size
        self.save_dir = config.training.save_dir
        self.save_interval = config.training.save_interval
        self.loss_threshold = config.training.loss_threshold

        self.setup_seed()

        # swanlab: to record training process
        self.swanlab_run = swanlab.init(
            project="train-VLN-model",
            experiment_name=self.config.model.name,
            config={
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
            }
        )

    def setup_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def epoch_train(self, model, batch):
        images = batch['image'].to(self.device)
        if self.config.model.instruction_encoder == "bert":
            instructions = model.instruction_encoder.encode_text(batch['instruction'])
            instructions = {k: v.to(self.device) for k, v in instructions.items()}
        else:
            instructions = torch.zeros((len(batch['instruction']), 50), dtype=torch.long).to(self.device)
        converted_action_map = {v: k for k, v in self.config.model.action_map.items()}
        
        valid_actions = list(converted_action_map.keys())
        default_action = "stop"
        default_id = converted_action_map[default_action]
        
        corrected_actions = []
        
        action_ids = []
        for action in batch['action_type']:
            if action in converted_action_map:
                action_ids.append(converted_action_map[action])
            else:
                closest_match = None
                min_distance = float('inf')
                for valid_action in valid_actions:
                    if abs(len(action) - len(valid_action)) < min_distance:
                        closest_match = valid_action
                        min_distance = abs(len(action) - len(valid_action))
                
                corrected_actions.append((action, closest_match or default_action))
                action_ids.append(converted_action_map[closest_match] if closest_match else default_id)
        
        if corrected_actions:
            logger.warning(f"Corrected invalid actions: {corrected_actions}")
        
        action_ids = torch.tensor(action_ids).to(self.device)

        if self.config.model.name == "cma": # TODO: this writing style is not good, need to be improved: every time we need to check the model name and set criterion
            criterion = nn.CrossEntropyLoss()

            self.optimizer.zero_grad()

            action_logits = model(images, instructions)
            loss = criterion(action_logits, action_ids)
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
            self.optimizer.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            swanlab.log({"loss": loss.item(), "lr": current_lr})

            return loss.item()
        elif self.config.model.name == "seq2seq":
            criterion = nn.CrossEntropyLoss()

            self.optimizer.zero_grad()

            action_logits = model(images, instructions)
            loss = criterion(action_logits, action_ids)
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
            self.optimizer.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            swanlab.log({"loss": loss.item(), "lr": current_lr})

            return loss.item()
        else:
            raise ValueError(f"Model {self.config.model.name} not supported")

    def epoch_evaluate(self):
        pass
    
    def train(self):
        logger.info(f"Training on {self.device}")

        # process data
        data_reader = ParquetReader(self.config)
        train_loader = data_reader.read_parquet(split='train')
        n_batches = len(train_loader)

        # create/load model
        model = InitialModel(self.config)
        if self.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        elif self.optimizer == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

        # lr scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.warmup_start_lr)

        # warmup paras calculation
        total_warmup_steps = self.warmup_epochs * n_batches if n_batches > 0 else 0
        global_step = 0

        try:
            if self.config.model.continue_train:
                logger.info(f"Continue training...")
                model.load_model(self.config.model.pretrained_path, self.device)
        except Exception as e:
            logger.warning(f"{e}: Failed to load model from {self.config.model.pretrained_path}, starting from scratch")

        for epoch in range(self.epochs):

            progress_bar = tqdm(
                total=n_batches,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                position=0,
                leave=True,
                colour="green"
            )
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                if self.if_warmup:
                    warmup_progress = global_step / total_warmup_steps
                    current_lr = self.warmup_start_lr + (self.lr - self.warmup_start_lr) * warmup_progress
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr
                    if current_lr >= self.lr:
                        self.if_warmup = False
                else:
                    scheduler.step()

                loss = self.epoch_train(model, batch)
                global_step += 1
                total_loss += loss
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix(
                    loss=f'{loss:.4f}',
                    avg_loss=f'{avg_loss:.4f}'
                    )
                progress_bar.update(1)

            progress_bar.close()
            logger.info(f"Epoch {epoch+1}/{self.epochs} loss: {total_loss/n_batches}")
            
            # Save model checkpoint
            if (epoch + 1) % self.save_interval == 0 or epoch + 1 == self.epochs:
                if hasattr(self, 'save_dir') and self.save_dir:
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    base_name = f"{self.config.model.name}_{timestamp}_epoch{epoch+1}"
                    if self.config.model.continue_train:
                        save_path = os.path.join(self.save_dir, f"{base_name}_continue.pth")
                    else:
                        save_path = os.path.join(self.save_dir, f"{base_name}.pth")
                    
                    model.save_model(
                        path=save_path,
                        epoch=epoch+1,
                        optimizer=self.optimizer
                    )
                
            if avg_loss < self.loss_threshold:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    def evaluate(self):
        pass