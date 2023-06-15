import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW

def get_optimizer_fn(trainer_config):
    train_method = trainer_config["trainer_name"].split("_")[0]

    # If we are using sklearn, we don't need to further configure the optimizer.
    if train_method != "sklearn":
        wd = trainer_config["wd"] if "wd" in trainer_config else 0
        if trainer_config["optim_name"] == "Adam":
            if "betas" not in trainer_config:
                def optim_fn(params):
                    return Adam(
                        params, lr=trainer_config["lr"], weight_decay=wd)
            else:
                def optim_fn(params):
                    return Adam(params, lr=trainer_config["lr"], betas=tuple(trainer_config["betas"]),
                                weight_decay=wd)
        elif trainer_config["optim_name"] == "AdamW":
            if "betas" not in trainer_config:
                def optim_fn(params):
                    return AdamW(
                        params, lr=trainer_config["lr"], weight_decay=wd)
            else:
                def optim_fn(params):
                    return AdamW(params, lr=trainer_config["lr"], betas=tuple(trainer_config["betas"]),
                                 weight_decay=wd)
        elif trainer_config["optim_name"] == "SGD":
            nesterov = trainer_config["nesterov"] if "nesterov" in trainer_config else False
            momentum = trainer_config["momentum"] if "momentum" in trainer_config else 0

            def optim_fn(params):
                return SGD(params, lr=trainer_config["lr"], weight_decay=wd, nesterov=nesterov,
                           momentum=momentum)
        else:
            raise ValueError("%s optimizer is unknown" %
                             trainer_config["optim_name"])
        trainer_config["optim_fn"] = optim_fn
    return trainer_config


def get_scheduler_fn(trainer_config):
    if "scheduler_name" in trainer_config:
        if trainer_config["scheduler_name"] == "CosineLR":
            def scheduler_fn(optimizer, total_steps):
                return cosine_lr(optimizer, base_lrs=trainer_config["lr"], warmup_length=trainer_config["warmup_steps"],
                                 steps=total_steps)
        elif trainer_config["scheduler_name"] == "StepLR":
            def scheduler_fn(optimizer, total_steps):
                return step_lr(optimizer, base_lr=trainer_config["lr"], step_size=trainer_config["step_size"],
                               gamma=trainer_config["gamma"])
        else:
            raise ValueError("%s scheduler is unknown" %
                             trainer_config["scheduler_name"])
        trainer_config["scheduler_fn"] = scheduler_fn

    return trainer_config


# Modified cosine_lr functions, copy from https://github.com/mlfoundations/wise-ft/blob/master/src/models/utils.py.
def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def step_lr(optimizer, base_lr, step_size, gamma):
    def _lr_adjuster(step):
        for param_group in optimizer.param_groups:
            if (step + 1) % step_size == 0:
                assign_learning_rate(param_group, base_lr * (gamma ** ((step + 1) // step_size)))

    return _lr_adjuster
