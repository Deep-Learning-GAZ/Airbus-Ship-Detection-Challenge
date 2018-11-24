import cv2

import keras
from keras import Model
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping, TensorBoard

from Model import TrainableModel
from getABSDData import getABSDDataMask


class RetrainedClassificationModel(TrainableModel):
    def __init__(self, name, img_width=224, img_height=224):
        super().__init__(name)
        self.img_width = img_width
        self.img_height = img_height
        model = keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
        model.layers.pop()
        model.layers.pop()

        # Adding custom Layers
        x = model.layers[-1].output
        x = Flatten()(x)
        predictions = Dense(img_width * img_height, activation="relu")(x)

        # Creating the final model
        self.model = Model(input=model.input, output=predictions)

        self.model.compile(loss="mean_squared_error", optimizer='adam',
                           metrics=["accuracy"])

    def train(self, batch_size: int, l2_regularization: float = 0, dropout_keep_porb: float = 0, n_epoch: int = 3):
        # Resize the data using the actual img_width and img_width:
        resizer = lambda x: cv2.resize(x, (self.img_width, self.img_width))

        # We can receive the training data using the getABSDDataMask function:
        training, dev, _ = getABSDDataMask(1, label_converter=resizer, image_converter=resizer)

        # The number of the epochs:
        n_epoch = 1000
        callbacks = [EarlyStopping(patience=10), TensorBoard()]
        # Learning:
        self.model.fit_generator(training, validation_data=dev, callbacks=callbacks, epochs=n_epoch, verbose=2)
