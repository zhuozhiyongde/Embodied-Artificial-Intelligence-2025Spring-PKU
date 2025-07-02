import os
from dataclasses import asdict
from typing import Dict
import warnings
import yaml
import torch
import wandb

from .config import Config
from .path import (
    get_exp_dir,
    get_log_path,
    get_exp_config_path,
    get_checkpoint_dir,
    get_checkpoint_path,
)
from .git import save_code_and_git


class Logger:

    exp_name: str

    def __init__(self, config: Config):
        """
        automatically create experiment directory and save config

        config needs to have at least 'exp_name'

        to avoid logging a lot of things in debugging, if exp_name is 'debug',
        we don't log anything in wandb
        """
        self.exp_name = config.exp_name
        if self.exp_name != "debug":
            wandb.init(project="Intro2EAI", name=self.exp_name, config=asdict(config))
        else:
            warnings.warn("exp_name is debug, so we don't log anything in wandb")

        # create exp directory
        os.makedirs(get_exp_dir(self.exp_name), exist_ok=True)
        os.makedirs(get_log_path(self.exp_name), exist_ok=True)
        os.makedirs(get_checkpoint_dir(self.exp_name), exist_ok=True)

        # save config
        with open(get_exp_config_path(self.exp_name), "w") as f:
            yaml.dump(asdict(config), f)

        save_code_and_git(get_exp_dir(self.exp_name))

    def log(self, dic: Dict[str, float], mode: str, step: int):
        """
        log a dictionary, requires all values to be scalar

        mode is used to distinguish train, val, ...

        step is the iteration number
        """
        if self.exp_name != "debug":
            wandb.log({f"{mode}/{k}": v for k, v in dic.items()}, step=step)

    def save(self, dic: dict, step: int):
        """
        save a dictionary (usually with model's weight and optimizer state) to a file
        """
        torch.save(dic, get_checkpoint_path(self.exp_name, step))
