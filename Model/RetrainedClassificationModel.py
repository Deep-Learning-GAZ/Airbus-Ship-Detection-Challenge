import cv2

import keras
from keras import Model
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l2

from Model import TrainableModel
from getABSDData import getABSDDataMask


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
        x = Flatten()(x)
        x = Dropout(0)(x)
        predictions = Dense(self.img_width * self.img_height, activation="relu")(x)

        # Creating the final model
        self.model = Model(input=model.input, output=predictions)

    def train(self, batch_size: int, l2_regularization: float = 0, dropout_drop_porb: float = 0, n_epoch: int = 3):
        label_converter = lambda x: cv2.resize(x, (self.img_width, self.img_width))
        image_converter = lambda x: keras.applications.vgg16.preprocess_input(label_converter(x))

        training, dev, _ = getABSDDataMask(1, label_converter=label_converter, image_converter=image_converter)

        callbacks = [EarlyStopping(patience=10), TensorBoard()]
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = l2(l2_regularization)
            if isinstance(layer, Dropout):
                layer.rate = dropout_drop_porb
        self.model.compile(loss="mean_squared_error", optimizer='adam')

        self.model.fit_generator(training, validation_data=dev, callbacks=callbacks, epochs=n_epoch, verbose=2)
