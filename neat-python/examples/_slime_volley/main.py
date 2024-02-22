import argparse
import os
import shutil
import jax

from evojax.task.slimevolley import SlimeVolley
from evojax.policy.mlp import MLPPolicy
from evojax.algo import CMA
from evojax import Trainer
from evojax import util

from arguments import parse_neat_args


def main(config):
    log_dir = './log/slimevolley'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='SlimeVolley', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX SlimeVolley')
    logger.info('=' * 30)

    max_steps = 3000
    train_task = SlimeVolley(test=False, max_steps=max_steps)
    test_task = SlimeVolley(test=True, max_steps=max_steps)

    # convert jax MLP to neat-jax MLP
    policy