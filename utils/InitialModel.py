"""
Initialize model from config.
model can be chosen from MAVLN, CMA, etc.
"""
import logging

logger = logging.getLogger(__name__)

def InitialModel(config):
    if config.model.name == "mavln":
        from model.mavln.mavln import MAVLN
        return MAVLN(config)
    elif config.model.name == "cma":
        from model.CMA.CMA import CMA
        return CMA(config)
    else:
        info = f"Invalid model name: {config.model.name}"
        logger.error(info)
        raise ValueError(info)