import os
import types
import unittest
from Utilities import imageDataStore
import pandas as pd
import numpy as np


class ImageDataStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        DATASET_FOLDER = r"../data"
        TRAIN_SUBFOLDER = os.path.join(DATASET_FOLDER, "train")
        TRAINING_ANNOTATION_FILE = os.path.join(DATASET_FOLDER, r"train_ship_segmentations.csv")
        training_data = pd.read_csv(TRAINING_ANNOTATION_FILE)
        self.fileNames = self.__joinFolder(TRAIN_SUBFOLDER, training_data.ImageId.tolist())
        self.encodedPixels = training_data.EncodedPixels.tolist()

    def testType(self):
        ds = imageDataStore(self.fileNames, self.encodedPixels, 16)
        self.assertIsInstance(ds, types.GeneratorType)

    def testBatchSize(self):
        batch_sizes = [16, 9]
        for batch_size in batch_sizes:
            ds = imageDataStore(self.fileNames, self.encodedPixels, batch_size)
            images, labels = next(ds)
            self.assertIsInstance(images, np.ndarray)
            self.assertIsInstance(labels, list)
            self.assertEquals(images.shape[0], batch_size)
            self.assertEquals(len(labels), batch_size)

    @staticmethod
    def __joinFolder(folder, file_names_list):
        return [os.path.join(folder, file_name) for file_name in file_names_list]


if __name__ == '__main__':
    unittest.main()
