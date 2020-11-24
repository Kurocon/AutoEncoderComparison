import importlib

import config
import logging

from models.base_corruption import BaseCorruption
from models.base_dataset import BaseDataset
from models.base_encoder import BaseEncoder
from models.test_run import TestRun

logger = logging.getLogger("main.py")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def load_dotted_path(path):
    split_path = path.split(".")
    modulename, classname = ".".join(split_path[:-1]), split_path[-1]
    model = getattr(importlib.import_module(modulename), classname)
    return model


def run_tests():
    for test in config.TEST_RUNS:
        logger.info(f"Running test run '{test['name']}'...")

        # Load dataset model
        dataset_model = load_dotted_path(test['dataset_model'])
        assert issubclass(dataset_model, BaseDataset), f"Invalid dataset_model: '{dataset_model.__name__}', should be subclass of BaseDataset."
        logger.debug(f"Using dataset model '{dataset_model.__name__}'")

        # Load auto-encoder model
        encoder_model = load_dotted_path(test['encoder_model'])
        assert issubclass(encoder_model, BaseEncoder), f"Invalid encoder_model: '{encoder_model.__name__}', should be subclass of BaseEncoder."
        logger.debug(f"Using encoder model '{encoder_model.__name__}'")

        # Load corruption model
        corruption_model = load_dotted_path(test['corruption_model'])
        assert issubclass(corruption_model, BaseCorruption), f"Invalid corruption_model: '{corruption_model.__name__}', should be subclass of BaseCorruption."
        logger.debug(f"Using corruption model '{corruption_model.__name__}'")

        # Create TestRun instance
        test_run = TestRun(dataset=dataset_model(**test['dataset_kwargs']),
                           encoder=encoder_model(**test['encoder_kwargs']),
                           corruption=corruption_model(**test['corruption_kwargs']))

        # Run TestRun
        test_run.run()


if __name__ == '__main__':
    run_tests()
