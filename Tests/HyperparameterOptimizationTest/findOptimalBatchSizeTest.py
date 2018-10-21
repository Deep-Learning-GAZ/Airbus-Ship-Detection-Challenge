import unittest
from Model import TrainableModel
import time
from HyperparameterOptimization import findOptimalBatchSize


class MyTestCase(unittest.TestCase):
    def test_something(self):
        optimal_batch_size_arr = [512, 128]
        for optimal_batch_size in optimal_batch_size_arr:
            model = _MockModel(optimal_batch_size)
            self.assertEqual(optimal_batch_size, findOptimalBatchSize(model, save_excel=False))


class _MockModel(TrainableModel):
    def __init__(self, optimal_batch_size):
        super().__init__('Mock')
        self.optimal_batch_size = optimal_batch_size

    def train(self, batch_size: int, l2_regularization: float = 0, dropout_keep_porb: float = 0, n_epoch: int = 3):
        wait_time_ms_fnc = lambda x: abs(x - self.optimal_batch_size) / 32
        ms2s = 1000
        time.sleep(wait_time_ms_fnc(batch_size) / ms2s)


if __name__ == '__main__':
    unittest.main()
