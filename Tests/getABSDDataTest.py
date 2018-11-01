import unittest

import numpy as np
from matplotlib.pyplot import imread

from Utilities import annotation2Mask, mask2BoxParameters
from getABSDData import getABSDData, getABSDDataFrames


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

    def testGetABSDData(self):
        batch_size = 16
        train, dev, test = getABSDData(batch_size, folder='../data')

        def chechBatch(ds):
            images, labels = next(ds)
            self.assertIsInstance(images, np.ndarray)
            self.assertIsInstance(labels, list)
            self.assertEquals(images.shape[0], batch_size)
            self.assertEquals(len(labels), batch_size)

        chechBatch(train)
        chechBatch(train)
        chechBatch(dev)
        chechBatch(dev)
        chechBatch(test)
        chechBatch(test)

    def testGetABSDDataConverter(self):
        batch_size = 16
        converter = lambda x: np.array(mask2BoxParameters(annotation2Mask(x)))
        train, dev, test = getABSDData(batch_size, converter, '../data')

        def chechBatch(ds):
            n_oriented_box_parameter = 5
            images, labels = next(ds)
            self.assertIsInstance(images, np.ndarray)
            self.assertIsInstance(labels, np.ndarray)
            self.assertEquals(images.shape[0], batch_size)
            self.assertEquals(labels.shape[0], batch_size)
            self.assertEquals(labels.shape[1], n_oriented_box_parameter)

        chechBatch(train)
        chechBatch(train)
        chechBatch(dev)
        chechBatch(dev)
        chechBatch(test)
        chechBatch(test)


if __name__ == '__main__':
    unittest.main()
