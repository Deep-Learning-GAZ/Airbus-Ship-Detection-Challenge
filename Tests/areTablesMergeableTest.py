import unittest
import pandas as pd
from Utilities import *


class AreTablesMergeableTest(unittest.TestCase):
    TB_A = '../data/train_ship_segmentations.csv'
    TB_B = '../data/test_ship_segmentations.csv'

    def test_mergeable(self):
        a = pd.read_csv(self.TB_A)
        b = pd.read_csv(self.TB_B)
        imageID = 'ImageId'
        self.assertTrue(a[imageID].apply(lambda x: isinstance(x, str)).all())
        self.assertTrue(b[imageID].apply(lambda x: isinstance(x, str)).all())
        self.assertTrue(areTablesMergeable(a, b, imageID))

    def test_not_mergeable(self):
        a = pd.read_csv(self.TB_A)
        b = pd.read_csv(self.TB_B)
        imageID = 'ImageId'
        imageIdIdx = a.columns.get_loc(imageID)
        a.iloc[0, imageIdIdx] = b.iloc[0, imageIdIdx]
        self.assertTrue(a[imageID].apply(lambda x: isinstance(x, str)).all())
        self.assertTrue(b[imageID].apply(lambda x: isinstance(x, str)).all())
        self.assertFalse(areTablesMergeable(a, b, imageID))



if __name__ == '__main__':
    unittest.main()
