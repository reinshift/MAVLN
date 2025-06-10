import argparse
import yaml
import logging
from types import SimpleNamespace
from train.trainer import Trainer

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict_to_namespace(config_dict)
    return config

def dict_to_namespace(d):
    if not isinstance(d, dict):
        return d
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        elif isinstance(value, list):
            setattr(namespace, key, [dict_to_namespace(item) if isinstance(item, dict) else item for item in value])
        else:
            setattr(namespace, key, value)
    return namespace

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/train.yaml")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    config = load_config(args.config_path)

    if args.mode == "train":
        logger.info("Training model...")
        trainer = Trainer(config)
        trainer.train()
    elif args.mode == "evaluate":
        logger.info("Evaluating model...")
        trainer = Trainer(config)
        trainer.evaluate()