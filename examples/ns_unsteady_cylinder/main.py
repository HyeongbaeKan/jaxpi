"""Entry point for the unsteady cylinder example.

The original example configured several JAX/XLA specific flags for
deterministic execution.  Since the codebase has been migrated to PyTorch,
those settings are no longer required and have been removed.
"""

import os

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags

# This example now runs with PyTorch instead of JAX.  Any previous JAX specific
# configuration (e.g. XLA flags) has been removed.

import train
import eval

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
