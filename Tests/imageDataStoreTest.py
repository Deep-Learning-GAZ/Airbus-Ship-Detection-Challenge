import os
import types
import unittest

import numpy as np
import pandas as pd

from Utilities import annotation2Mask, imageDataStore, joinFolder


class ImageDataStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        DATASET_FOLDER = r"../data"
        TRAIN_SUBFOLDER = os.path.join(DATASET_FOLDER, "test")
        TRAINING_ANNOTATION_FILE = os.path.join(DATASET_FOLDER, r"test_ship_segmentations.csv")
        training_data = pd.read_csv(TRAINING_ANNOTATION_FILE)
        self.fileNames = joinFolder(TRAIN_SUBFOLDER, training_data.ImageId.tolist())
        self.encodedPixels = training_data.EncodedPixels.tolist()

    def testType(self):
        ds = imageDataStore(self.fileNames, self.encodedPixels, 16)
        self.assertIsInstance(ds, types.GeneratorType)

    def testBatchSize(self):
        batch_sizes = [4, 16, 9]
        for batch_size in batch_sizes:
            ds = imageDataStore(self.fileNames, self.encodedPixels, batch_size)
            images, labels = next(ds)
            self.assertIsInstance(images, np.ndarray)
            self.assertIsInstance(labels, list)
            self.assertEquals(images.shape[0], batch_size)
            self.assertEquals(len(labels), batch_size)
            next(ds)
            next(ds)

    def testConverter(self):
        batch_size = 16
        converter = lambda x: annotation2Mask(x).flatten()
        ds = imageDataStore(self.fileNames, self.encodedPixels, batch_size, converter)
        images, labels = next(ds)
        self.assertIsInstance(images, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEquals(images.shape[0], batch_size)
        self.assertEquals(labels.shape[0], batch_size)
        next(ds)
        next(ds)

    @staticmethod
    def __joinFolder(folder, file_names_list):
        return [os.path.join(folder, file_name) for file_name in file_names_list]


if __name__ == '__main__':
    unittest.main()
