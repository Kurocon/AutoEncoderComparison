MODEL_STORAGE_BASE_PATH = "/path/to/this/project/saved_models"
DATASET_STORAGE_BASE_PATH = "/path/to/this/project/datasets"
TRAIN_TEMP_DATA_BASE_PATH = "/path/to/this/project/train_temp"
TEST_TEMP_DATA_BASE_PATH = "/path/to/this/project/test_temp"

TEST_RUNS = [
    {
        'name': "Basic test run",
        'encoder_model': "models.base_encoder.BaseEncoder",
        'encoder_kwargs': {},
        'dataset_model': "models.base_dataset.BaseDataset",
        'dataset_kwargs': {},
        'corruption_model': "models.base_corruption.NoCorruption",
        'corruption_kwargs': {},
    },
]
