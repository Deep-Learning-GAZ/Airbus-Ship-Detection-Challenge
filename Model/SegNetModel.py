from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Dropout
from keras.regularizers import l2

from Model import TrainableModel
from getABSDData import getABSDDataMask
from model import segnet


class SegNetModel(TrainableModel):
    def __init__(self, name):
        super().__init__(name)

    def reset(self):
        self.n_classes = 1
        self.model = segnet(input_shape=(768, 768, 3), n_labels=self.n_classes,
                            kernel=3, pool_size=(2, 2), output_mode="softmax")

    def train(self, batch_size: int, l2_regularization: float = 0, dropout_drop_porb: float = 0, n_epoch: int = 3):
        training, dev, _ = getABSDDataMask(batch_size=batch_size)
        self.model.compile(loss="binary_crossentropy", optimizer="adam")

        callbacks = [EarlyStopping(patience=10), TensorBoard(),
                     ModelCheckpoint('segnet.{epoch:02d}-{val_loss:.2f}.hdf5')]
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = l2(l2_regularization)
            if isinstance(layer, Dropout):
                layer.rate = dropout_drop_porb

        hst = self.model.fit_generator(training, validation_data=dev, callbacks=callbacks, epochs=n_epoch)

        self.model.save("segnet.hd5")

        return hst
