import argparse
import logging
from types import SimpleNamespace
from train.trainer import Trainer
import scripts.test_mavln as test_mavln
from utils.ConfigParser import load_config

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/cma.yaml")
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
    elif args.mode == "test":
        logger.info("Testing model...")
        test_mavln.main()

if __name__ == "__main__":
    main()