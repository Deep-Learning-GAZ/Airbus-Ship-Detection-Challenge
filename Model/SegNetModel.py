from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.regularizers import l2

from Model import TrainableModel
from getABSDData import getABSDDataMask
from model import segnet
from Utilities.Metrics import precision, recall, f1, f2, iou, MetricsCallback, pred_area, true_area


class SegNetModel(TrainableModel):
    def __init__(self, name, use_residual=False, use_argmax=True):
        super().__init__(name)
        self.reset(use_residual=False, use_argmax=True)

    def reset(self, use_residual=False, use_argmax=True):
        self.n_classes = 2
        self.model = segnet(input_shape=(768, 768, 3), n_labels=self.n_classes,
                            kernel=3, pool_size=(2, 2), output_mode="softmax", use_residual=False, use_argmax=True)

    def train(self, batch_size: int, l2_regularization: float = 0, dropout_drop_porb: float = 0, n_epoch: int = 3,
              reduced_size=None, remove_nan=True):
        image_converter = lambda x: x/255
        training, dev, _ = getABSDDataMask(batch_size=batch_size, image_converter=image_converter, reduced_size=reduced_size, remove_nan=remove_nan)
        optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
        self.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[precision, recall, f1, f2, iou, pred_area, true_area, "accuracy"])

        callbacks = [EarlyStopping(patience=10), TensorBoard(write_images=True),
                     ModelCheckpoint('segnet.{epoch:02d}-{val_loss:.2f}.hdf5')]
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = l2(l2_regularization)
            if isinstance(layer, Dropout):
                layer.rate = dropout_drop_porb

        hst = self.model.fit_generator(training, validation_data=dev, callbacks=callbacks, epochs=n_epoch)

        self.model.save(self.name + ".hd5")

        return hst
    
    def eval(self, batch_size: int, reduced_size=None, remove_nan=True):
        training, dev, _ = getABSDDataMask(batch_size=batch_size, reduced_size=reduced_size, remove_nan=remove_nan)
        return self.model.evaluate_generator(dev)
    