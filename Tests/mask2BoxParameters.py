import unittest
import os
import pandas as pd
from Utilities import annotation2Mask
from Utilities import mask2BoxParameters


# Tip for testing: Use the following code to show the mask and gain a better intuition about  the input
#         plt.imshow(mask)
#         plt.show()
class Mask2BoxParametersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DATASET_FOLDER = "../data"
        TRAINING_ANNOTATION_FILE = os.path.join(DATASET_FOLDER, "train_ship_segmentations.csv")
        cls.training_data = pd.read_csv(TRAINING_ANNOTATION_FILE)

    def testHorizontal(self):
        annotation = self.training_data.loc[1, 'EncodedPixels']
        mask = annotation2Mask(annotation)
        [centerX, centerY, alpha, long_side, short_side] = mask2BoxParameters(mask)
        self.assertAlmostEqual(centerX, 447.51483845784924)
        self.assertAlmostEqual(centerY, 531.9995614142575)
        self.assertAlmostEqual(alpha, 1.541102087187987)
        self.assertAlmostEqual(long_side, 102.04410811016969)
        self.assertAlmostEqual(short_side, 32.0624390837628)

    def testSloping(self):
        annotation = self.training_data.loc[2, 'EncodedPixels']
        mask = annotation2Mask(annotation)
        [centerX, centerY, alpha, long_side, short_side] = mask2BoxParameters(mask)
        self.assertAlmostEqual(centerX, 174.0)
        self.assertAlmostEqual(centerY, 751.5)
        self.assertAlmostEqual(alpha, 0.9827937232473292)
        self.assertAlmostEqual(long_side, 36.05551275463989)
        self.assertAlmostEqual(short_side, 10.816653826391969)

    def testUpthick(self):
        annotation = self.training_data.loc[93, 'EncodedPixels']
        mask = annotation2Mask(annotation)
        [centerX, centerY, alpha, long_side, short_side] = mask2BoxParameters(mask)
        self.assertAlmostEqual(centerX, 282.6266530267141)
        self.assertAlmostEqual(centerY, 604.9955919287625)
        self.assertAlmostEqual(alpha, 1.9295669970654687)
        self.assertAlmostEqual(long_side, 215.72667892497674)
        self.assertAlmostEqual(short_side, 34.539832078341085)


if __name__ == '__main__':
    unittest.main()
