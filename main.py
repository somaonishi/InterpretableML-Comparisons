import logging

import hydra
import tensorflow as tf

import core
from core import ExpBase

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print("{} memory growth: {}".format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="main")
def main(config):
    exp: ExpBase = getattr(core, config.exp.name)(config)
    exp.run()


if __name__ == "__main__":
    main()
