import unittest

import numpy as np
from matplotlib.pyplot import imread

from Utilities import annotation2Mask
from getABSDData import getABSDData, getABSDDataMask, getABSDDataFrames


class GetABSDDataTest(unittest.TestCase):
    def testGetABSDDataFrames(self):
        train, dev, test = getABSDDataFrames('../data')
        self.assertTrue(len(train) > len(dev))
        self.assertTrue(len(dev) > len(test))
        train_image_names = set(train.ImageId)
        dev_image_names = set(dev.ImageId)
        test_image_names = set(test.ImageId)
        self.assertTrue(len(train_image_names & dev_image_names) == 0)
        self.assertTrue(len(dev_image_names & test_image_names) == 0)
        self.assertTrue(len(train_image_names & test_image_names) == 0)
        imread(train.ImageId.iloc[0])

    def testGetABSDDataConverter(self):
        batch_size = 16
        converter = lambda x: annotation2Mask(x).flatten()
        train, dev, test = getABSDData(batch_size, converter, '../data')

        def chechBatch(ds):
            n_pixels = 768*768
            images, labels = next(ds)
            self.assertIsInstance(images, np.ndarray)
            self.assertIsInstance(labels, np.ndarray)
            self.assertEquals(images.shape[0], batch_size)
            self.assertEquals(labels.shape[0], batch_size)
            self.assertEquals(labels.shape[1], n_pixels)

        chechBatch(train)
        chechBatch(train)
        chechBatch(dev)
        chechBatch(dev)
        chechBatch(test)
        chechBatch(test)

    def testGetABSDDataMask(self):
        batch_size = 16
        train, dev, test = getABSDDataMask(batch_size, folder='../data')

        def chechBatch(ds):
            mask_shape = 768* 768
            images, labels = next(ds)
            self.assertIsInstance(images, np.ndarray)
            self.assertIsInstance(labels, np.ndarray)
            self.assertEquals(images.shape[0], batch_size)
            self.assertEquals(labels.shape[0], batch_size)
            self.assertEquals(labels.shape[1], mask_shape)

        chechBatch(train)
        chechBatch(train)
        chechBatch(dev)
        chechBatch(dev)
        chechBatch(test)
        chechBatch(test)


if __name__ == '__main__':
    unittest.main()
