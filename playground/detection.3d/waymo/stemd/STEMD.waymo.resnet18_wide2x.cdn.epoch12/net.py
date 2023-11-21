from env import CustomTrainer, CustomWDDataset
from modules.optimizer import AdamWMulti
from stemd import STEMD

__all__ = ["build_model", "AdamWMulti", "CustomTrainer", "CustomWDDataset", "STEMD"]


def build_model(self, config):

    model = STEMD(config)

    return model
