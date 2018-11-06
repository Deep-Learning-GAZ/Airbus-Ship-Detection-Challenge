from abc import ABC, abstractmethod


class TrainableModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def train(self, batch_size: int, l2_regularization: float = 0, dropout_keep_porb: float = 0, n_epoch: int = 3):
        """
        Trains a neural networks with the supplied parameters.
        :param n_epoch: Number of epochs to train.
        :param batch_size: Mini-batch size. Should be the power of to for maximal performance.
        :param l2_regularization: L2 regularization value.
        :param dropout_keep_porb: Number between 0 and 1. Layers with dropout will set outputs to 0 with
               (1 - dropout_keep_porb) probability.
        """
