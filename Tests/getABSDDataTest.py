import unittest
from getABSDData import getABSDData


class GetABSDDataTest(unittest.TestCase):
    def test_something(self):
        train, dev, test = getABSDData('../data')
        self.assertTrue(len(train) > len(dev))
        self.assertTrue(len(dev) > len(test))
        train_image_names = set(train.ImageId)
        dev_image_names = set(dev.ImageId)
        test_image_names = set(test.ImageId)
        self.assertTrue(len(train_image_names & dev_image_names) == 0)
        self.assertTrue(len(dev_image_names & test_image_names) == 0)
        self.assertTrue(len(train_image_names & test_image_names) == 0)


if __name__ == '__main__':
    unittest.main()
