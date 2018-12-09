import cv2
import numpy as np

import keras
from keras import Model
from keras.layers import Flatten, Dropout, Reshape, Conv2D
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l2

from Model import TrainableModel
from getABSDData import getABSDDataMask
from Utilities.Metrics import precision, recall, f1


class RetrainedClassificationModel(TrainableModel):
    def __init__(self, name, img_width=224, img_height=224):
        super().__init__(name)
        self.img_width = img_width
        self.img_height = img_height
        self.model = None
        self.reset()

    def reset(self):
        model = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(
            self.img_width, self.img_height, 3))
        model.layers.pop()

        # Adding custom Layers
        x = model.layers[-1].output
        x = Reshape((self.img_width, self.img_height, 2))(x)
        x = Conv2D(1, 1, activation='sigmoid')(x)
        predictions = Flatten()(x)

        # Creating the final model
        self.model = Model(input=model.input, output=predictions)

    def train(self, batch_size: int, l2_regularization: float = 0, dropout_drop_porb: float = 0, n_epoch: int = 3,
              reduced_size=None, remove_nan=True):
        label_converter = lambda x: cv2.resize(x, (self.img_width, self.img_height))
        image_converter = self._input_converter
        image_converter_eval = lambda x: keras.applications.vgg16.preprocess_input(label_converter(x))

        training, _, _ = getABSDDataMask(batch_size, label_converter=label_converter, image_converter=image_converter,
                                           reduced_size=reduced_size, remove_nan=remove_nan)
        _, dev, _ = getABSDDataMask(batch_size, label_converter=label_converter, image_converter=image_converter_eval,
                                           reduced_size=reduced_size, remove_nan=remove_nan)

        callbacks = [TensorBoard(write_images=True),
                     ModelCheckpoint('tcm.{epoch:02d}.hdf5', monitor='val_f1', save_best_only=True, mode='max'),
                     EarlyStopping(monitor='val_loss', patience=10)]
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = l2(l2_regularization)
            if isinstance(layer, Dropout):
                layer.rate = dropout_drop_porb
        self.model.compile(loss="binary_crossentropy", optimizer='adagrad', metrics=[precision, recall, f1])

        hst = self.model.fit_generator(training, validation_data=dev, callbacks=callbacks, epochs=n_epoch)
        self.model.save("tcm.hd5")
        return hst
    
    def eval(self, batch_size: int, reduced_size=None, remove_nan=True):
        label_converter = lambda x: cv2.resize(x, (self.img_width, self.img_height))
        image_converter = lambda x: keras.applications.vgg16.preprocess_input(label_converter(x))

        training, dev, _ = getABSDDataMask(batch_size, label_converter=label_converter, image_converter=image_converter,
                                           reduced_size=reduced_size, remove_nan=remove_nan)
        return self.model.evaluate_generator(dev, verbose=1)

    def _input_converter(self, x):
        resized = cv2.resize(x, (self.img_width, self.img_height))
        noise = np.random.normal(0, 5, size=resized.shape)
        img = np.clip(resized + noise, 0, 255)
        return keras.applications.vgg16.preprocess_input(img)