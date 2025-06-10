import random
import torch

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.seed = config.training.seed
        self.lr = config.training.lr
        self.epochs = config.training.epochs
        self.gradient_clip = config.training.gradient_clip
        self.optimizer = config.training.optimizer

    def setup_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        self.setup_seed()

    def evaluate(self):
        pass