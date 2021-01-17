import logging
import multiprocessing

from models.base_corruption import BaseCorruption
from models.base_dataset import BaseDataset
from models.base_encoder import BaseEncoder
from utils import save_train_loss_graph


class TestRun:
    dataset: BaseDataset = None
    encoder: BaseEncoder = None
    corruption: BaseCorruption = None

    def __init__(self, dataset: BaseDataset, encoder: BaseEncoder, corruption: BaseCorruption):
        self.dataset = dataset
        self.encoder = encoder
        self.corruption = corruption
        self.log = logging.getLogger(self.__class__.__name__)

    def run(self, retrain: bool = False, save_model: bool = True):
        """
        Run the test
        :param retrain: If the auto-encoder should be trained from scratch
        :type retrain: bool
        :param save_model: If the auto-encoder should be saved after re-training (only effective when retraining)
        :type save_model: bool
        """
        # Verify inputs
        if self.dataset is None:
            raise ValueError("Cannot run test! Dataset is not specified.")
        if self.encoder is None:
            raise ValueError("Cannot run test! AutoEncoder is not specified.")
        if self.corruption is None:
            raise ValueError("Cannot run test! Corruption method is not specified.")

        # Load dataset
        self.log.info("Loading dataset...")
        self.dataset.load()

        if retrain:
            # Train encoder
            self.log.info("Training auto-encoder...")
            train_loss = self.encoder.train_encoder(self.dataset, epochs=50, num_workers=multiprocessing.cpu_count() - 1)

            if save_model:
                self.log.info("Saving auto-encoder model...")
                self.encoder.save_model(f"{self.encoder.name}_{self.dataset.name}")

            # Save train loss graph
            self.log.info("Saving loss graph...")
            save_train_loss_graph(train_loss, f"{self.encoder.name}_{self.dataset.name}")
        else:
            self.log.info("Loading saved auto-encoder...")
            load_success = self.encoder.load_model(f"{self.encoder.name}_{self.dataset.name}")
            if not load_success:
                self.log.error("Loading failed. Stopping test run.")
                return

        # Test encoder
        self.log.info("Testing auto-encoder...")
        self.encoder.test_encoder(self.dataset, corruption=self.corruption, num_workers=multiprocessing.cpu_count() - 1)

        self.log.info("Done!")

    def get_metrics(self):
        raise NotImplementedError()
