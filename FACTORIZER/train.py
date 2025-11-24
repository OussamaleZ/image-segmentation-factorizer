from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from registry import read_config
import wandb
import os

seed_everything(42, workers=True)
torch.set_default_dtype(torch.float32)
print("Device: ", "cuda" if torch.cuda.is_available() else "cpu")


## wandb

if "WANDB_API_KEY" not in os.environ:
    raise RuntimeError("Set WANDB_API_KEY in your environment before running.")

wandb.login(key=os.environ["WANDB_API_KEY"])

def main(args: Namespace):
    # get config
    config = read_config(args.config)

    # data module
    dm = config["data"]

    # init model
    task_cls, task_params = config["task"]
    if "checkpoint_path" in task_params:
        model = task_cls.load_from_checkpoint(strict=False, **task_params)
    else:
        model = task_cls(**task_params)

    # init trainer
    trainer = Trainer(**config["training"])

    # save raw config for wandb
    wandb_logger = trainer.logger if isinstance(trainer.logger, WandbLogger) else None
    if wandb_logger:
        wandb_logger.experiment.config.update(config, allow_val_change=True)
        wandb_logger.experiment.config.update({"config_path": args.config})
        wandb_logger.experiment.save(args.config)  # attach the file itself

    # fit model
    trainer.fit(model, dm)


def get_args() -> Namespace:
    parser = ArgumentParser(description="""Train the model.""", add_help=False)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)