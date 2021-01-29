MODEL_STORAGE_BASE_PATH = "/path/to/this/project/saved_models"
DATASET_STORAGE_BASE_PATH = "/path/to/this/project/datasets"
TRAIN_TEMP_DATA_BASE_PATH = "/path/to/this/project/train_temp"
TEST_TEMP_DATA_BASE_PATH = "/path/to/this/project/test_temp"


TEST_RUNS = [
    # CIFAR-10 dataset
    # {
    #     'name': "CIFAR-10 on basic auto-encoder",
    #     'encoder_model': "models.basic_encoder.BasicAutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.cifar10_dataset.Cifar10Dataset",
    #     'dataset_kwargs': {"path": "cifar-10-batches-py"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "CIFAR-10 on sparse L1 auto-encoder",
    #     'encoder_model': "models.sparse_encoder.SparseL1AutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.cifar10_dataset.Cifar10Dataset",
    #     'dataset_kwargs': {"path": "cifar-10-batches-py"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "CIFAR-10 on denoising auto-encoder",
    #     'encoder_model': "models.denoising_encoder.DenoisingAutoEncoder",
    #     'encoder_kwargs': {'input_corruption_model': "models.gaussian_corruption.GaussianCorruption"},
    #     'dataset_model': "models.cifar10_dataset.Cifar10Dataset",
    #     'dataset_kwargs': {"path": "cifar-10-batches-py"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "CIFAR-10 on contractive auto-encoder",
    #     'encoder_model': "models.contractive_encoder.ContractiveAutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.cifar10_dataset.Cifar10Dataset",
    #     'dataset_kwargs': {"path": "cifar-10-batches-py"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "CIFAR-10 on variational auto-encoder",
    #     'encoder_model': "models.variational_encoder.VariationalAutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.cifar10_dataset.Cifar10Dataset",
    #     'dataset_kwargs': {"path": "cifar-10-batches-py"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },

    # MNIST dataset
    # {
    #     'name': "MNIST on basic auto-encoder",
    #     'encoder_model': "models.basic_encoder.BasicAutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.mnist_dataset.MNISTDataset",
    #     'dataset_kwargs': {"path": "mnist"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "MNIST on sparse L1 auto-encoder",
    #     'encoder_model': "models.sparse_encoder.SparseL1AutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.mnist_dataset.MNISTDataset",
    #     'dataset_kwargs': {"path": "mnist"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "MNIST on denoising auto-encoder",
    #     'encoder_model': "models.denoising_encoder.DenoisingAutoEncoder",
    #     'encoder_kwargs': {'input_corruption_model': "models.gaussian_corruption.GaussianCorruption"},
    #     'dataset_model': "models.mnist_dataset.MNISTDataset",
    #     'dataset_kwargs': {"path": "mnist"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "MNIST on contractive auto-encoder",
    #     'encoder_model': "models.contractive_encoder.ContractiveAutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.mnist_dataset.MNISTDataset",
    #     'dataset_kwargs': {"path": "mnist"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "MNIST on variational auto-encoder",
    #     'encoder_model': "models.variational_encoder.VariationalAutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.mnist_dataset.MNISTDataset",
    #     'dataset_kwargs': {"path": "mnist"},
    #     'corruption_model': "models.gaussian_corruption.GaussianCorruption",
    #     'corruption_kwargs': {},
    # },

    # US Weather Events dataset
    # {
    #     'name': "US Weather Events on basic auto-encoder",
    #     'encoder_model': "models.basic_encoder.BasicAutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.usweather_dataset.USWeatherEventsDataset",
    #     'dataset_kwargs': {"path": "weather-events"},
    #     'corruption_model': "models.random_corruption.RandomCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "US Weather Events on sparse L1 auto-encoder",
    #     'encoder_model': "models.sparse_encoder.SparseL1AutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.usweather_dataset.USWeatherEventsDataset",
    #     'dataset_kwargs': {"path": "weather-events"},
    #     'corruption_model': "models.random_corruption.RandomCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "US Weather Events on denoising auto-encoder",
    #     'encoder_model': "models.denoising_encoder.DenoisingAutoEncoder",
    #     'encoder_kwargs': {'input_corruption_model': "models.random_corruption.RandomCorruption"},
    #     'dataset_model': "models.usweather_dataset.USWeatherEventsDataset",
    #     'dataset_kwargs': {"path": "weather-events"},
    #     'corruption_model': "models.random_corruption.RandomCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "US Weather Events on contractive auto-encoder",
    #     'encoder_model': "models.contractive_encoder.ContractiveAutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.usweather_dataset.USWeatherEventsDataset",
    #     'dataset_kwargs': {"path": "weather-events"},
    #     'corruption_model': "models.random_corruption.RandomCorruption",
    #     'corruption_kwargs': {},
    # },
    # {
    #     'name': "US Weather Events on variational auto-encoder",
    #     'encoder_model': "models.variational_encoder.VariationalAutoEncoder",
    #     'encoder_kwargs': {},
    #     'dataset_model': "models.usweather_dataset.USWeatherEventsDataset",
    #     'dataset_kwargs': {"path": "weather-events"},
    #     'corruption_model': "models.random_corruption.RandomCorruption",
    #     'corruption_kwargs': {},
    # },
]

