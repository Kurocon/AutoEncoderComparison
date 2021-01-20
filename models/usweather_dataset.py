import csv
import os
from collections import defaultdict
from datetime import datetime

from typing import Optional

import numpy
import torch

from config import DATASET_STORAGE_BASE_PATH
from models.base_dataset import BaseDataset


class USWeatherEventsDataset(BaseDataset):
    # Source: https://smoosavi.org/datasets/lstw
    # https://www.kaggle.com/sobhanmoosavi/us-weather-events
    name = "US Weather Events"

    def transform(self, data):
        return torch.from_numpy(numpy.array(data, numpy.float32, copy=False))

    def unpickle(self, filename):
        import pickle
        with open(filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load(self, name: Optional[str] = None, path: Optional[str] = None):
        if name is not None:
            self.name = name
        if path is not None:
            self._source_path = path

        self._data = []
        self._labels = defaultdict(list)

        # Load from cache pickle file if it exists, else create cache file and load from csv
        if os.path.isfile(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "weather_py_data.pickle"))\
                and os.path.isfile(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "weather_py_labels.pickle")):
            self.log.info("Loading cached version of dataset...")
            self._data = self.unpickle(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "weather_py_data.pickle"))
            self._labels = self.unpickle(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "weather_py_labels.pickle"))
        else:
            self.log.info("Creating cached version of dataset...")
            size = 5023709
            with open(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "WeatherEvents_Aug16_June20_Publish.csv")) as f:
                data = csv.DictReader(f)
                # Build label map before processing for 1-hot encoding
                self.log.info("Preparing labels...")
                for i, row in enumerate(data):
                    if i % 500000 == 0:
                        self.log.debug(f"{i} / ~{size} ({((i / size) * 100):.4f}%)")

                    for label_type in ['Type', 'Severity', 'TimeZone', 'State']:
                        if row[label_type] not in self._labels[label_type]:
                            self._labels[label_type].append(row[label_type])

            with open(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "WeatherEvents_Aug16_June20_Publish.csv")) as f:
                data = csv.DictReader(f)
                self.log.info("Processing data...")
                for i, row in enumerate(data):
                    self._data.append(numpy.array([] +
                        # Event ID doesn't matter
                        # 1-hot encoded event type columns
                        [int(row['Type'] == self._labels['Type'][i]) for i in range(len(self._labels['Type']))] +

                        # 1-hot encoded event severity columns
                        [int(row['Severity'] == self._labels['Severity'][i]) for i in range(len(self._labels['Severity']))] +

                        [
                            # Start time as unix timestamp
                            datetime.strptime(row['StartTime(UTC)'], "%Y-%m-%d %H:%M:%S").timestamp(),
                            # End time as unix timestamp
                            datetime.strptime(row['EndTime(UTC)'], "%Y-%m-%d %H:%M:%S").timestamp()
                        ] +

                        # 1-hot encoded event timezone columns
                        [int(row['TimeZone'] == self._labels['TimeZone'][i]) for i in range(len(self._labels['TimeZone']))] +

                        [
                            # Location Latitude as float
                            float(row['LocationLat']),
                            # Location Longitude as float
                            float(row['LocationLng']),
                        ] +

                        # 1-hot encoded event state columns
                        [int(row['State'] == self._labels['State'][i]) for i in range(len(self._labels['State']))]

                        # Airport code, city, county and zip code are not considered,
                        # as they have too many unique values for 1-hot encoding.
                    ))

                    if i % 500000 == 0:
                        self.log.debug(f"{i} / ~{size} ({((i / size) * 100):.4f}%)")

            self.log.info("Shuffling data...")
            rng = numpy.random.default_rng()
            rng.shuffle(self._data)

            self.log.info("Saving cached version...")
            import pickle
            with open(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "weather_py_data.pickle"), 'wb') as f:
                pickle.dump(self._data, f)
            with open(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "weather_py_labels.pickle"), 'wb') as f:
                pickle.dump(dict(self._labels), f)
            self.log.info("Cached version created.")

        train_data, test_data = self._data[:2500000], self._data[2500000:]

        self._trainset = self.__class__.get_new(name=f"{self.name} Training", data=train_data, labels=self._labels,
                                                source_path=self._source_path)

        self._testset = self.__class__.get_new(name=f"{self.name} Testing", data=test_data, labels=self._labels,
                                               source_path=self._source_path)

        self.log.info(f"Loaded {self}, divided into {self._trainset} and {self._testset}")

    def get_input_shape(self):
        if os.path.isfile(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "weather_py_labels.pickle")):
            labels = self.unpickle(os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path, "weather_py_labels.pickle"))
            size = 0
            size += len(labels['Type'])
            size += len(labels['Severity'])
            size += 2
            size += len(labels['TimeZone'])
            size += 2
            size += len(labels['State'])
            return size
        else:
            return 69

    def __getitem__(self, item):
        data = self._data[item]

        # Run transforms
        if self.transform is not None:
            data = self.transform(data)

        return data

    def save_batch_to_sample(self, batch, filename):
        res = ["Type,Severity,StartTime(UTC),EndTime(UTC),TimeZone,LocationLat,LocationLng,State\n"]

        for row in batch:
            # Get 1-hot encoded values as list per value, and other values as value
            row = row.tolist()
            start = 0
            length = len(self._labels['Type'])
            event_types = row[start:start+length]
            start += length
            length = len(self._labels['Severity'])
            severities = row[start:start+length]
            start += length
            start_time = row[start]
            end_time = row[start+1]
            start += 2
            length = len(self._labels['TimeZone'])
            timezones = row[start:start+length]
            start += length
            location_lat = row[start]
            location_lng = row[start+1]
            start += 2
            length = len(self._labels['State'])
            states = row[start:start+length]

            # Convert 1-hot encodings to normal labels, assume highest value as the true value.
            event_type = self._labels['Type'][event_types.index(max(event_types))]
            severity = self._labels['Severity'][severities.index(max(severities))]
            timezone = self._labels['TimeZone'][timezones.index(max(timezones))]
            state = self._labels['State'][states.index(max(states))]

            # Convert timestamp float into string time
            start_time = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
            end_time = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")

            res.append(f"{event_type},{severity},{start_time},{end_time},{timezone},{location_lat},{location_lng},{state}\n")

        with open(f"{filename}.csv", "w") as f:
            f.writelines(res)

    def calculate_score(self, originals, reconstruction, device):
        originals = originals.to(device)
        reconstruction = reconstruction.to(device)

        total_score = 0
        for i in range(len(originals)):
            original, recon = originals[i], reconstruction[i]
            total_score += sum(int(original[j] == recon[j]) for j in range(len(original))) / len(original)

        return total_score / len(originals)
