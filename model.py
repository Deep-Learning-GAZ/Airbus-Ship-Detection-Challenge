from keras.models import Model
from keras.layers import Input, Flatten, Add
from keras.layers.core import Activation, Reshape
from keras.layers.merge import Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization

from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


def segnet(
        input_shape,
        n_labels,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax",
        use_residual=False,
        use_argmax=True):
    # encoder
    inputs = Input(shape=input_shape)

    residual_connections = []
    
    conv_1 = Conv2D(64, kernel, padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Conv2D(64, kernel, padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    if use_argmax: pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)
    else: pool_1 = MaxPooling2D(pool_size)(conv_2)
    if use_residual: residual_connections.append(pool_1)
    
    conv_3 = Conv2D(128, kernel, padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Conv2D(128, kernel, padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    
    if use_argmax: pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)
    else: pool_2 = MaxPooling2D(pool_size)(conv_4)
    if use_residual: residual_connections.append(pool_2)
    
    conv_5 = Conv2D(256, kernel, padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Conv2D(256, kernel, padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Conv2D(256, kernel, padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    
    if use_argmax: pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)
    else: pool_3 = MaxPooling2D(pool_size)(conv_7)
    if use_residual: residual_connections.append(pool_3)
    
    conv_8 = Conv2D(512, kernel, padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Conv2D(512, kernel, padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Conv2D(512, kernel, padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)
    
    if use_argmax: pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)
    else: pool_4 = MaxPooling2D(pool_size)(conv_10)
    if use_residual: residual_connections.append(pool_4)
    
    conv_11 = Conv2D(512, kernel, padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Conv2D(512, kernel, padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Conv2D(512, kernel, padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)
    
    if use_argmax: pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    else: pool_5 = MaxPooling2D(pool_size)(conv_13)
    if use_residual: residual_connections.append(pool_5)
    print("Done building encoder..")
    
    # decoder
    
    if use_residual: pool_5 = Add()([pool_5, residual_connections[-1]])
    if use_argmax: unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])
    else: unpool_1 = UpSampling2D(pool_size)(pool_5)
    
    conv_14 = Conv2D(512, kernel, padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Conv2D(512, kernel, padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Conv2D(512, kernel, padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)
    
    if use_residual: conv_16 = Add()([conv_16, residual_connections[-2]])
    if use_argmax: unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])
    else: unpool_2 = UpSampling2D(pool_size)(conv_16)
    
    conv_17 = Conv2D(512, kernel, padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Conv2D(512, kernel, padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Conv2D(256, kernel, padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)
    
    if use_residual: conv_19 = Add()([conv_19, residual_connections[-3]])
    if use_argmax: unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])
    else: unpool_3 = UpSampling2D(pool_size)(conv_19)
    
    conv_20 = Conv2D(256, kernel, padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Conv2D(256, kernel, padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Conv2D(128, kernel, padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)
    
    if use_residual: conv_22 = Add()([conv_22, residual_connections[-4]])
    if use_argmax: unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])
    else: unpool_4 = UpSampling2D(pool_size)(conv_22)
    
    conv_23 = Conv2D(128, kernel, padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Conv2D(64, kernel, padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)
    
    if use_residual: conv_24 = Add()([conv_24, residual_connections[-5]])
    if use_argmax: unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])
    else: unpool_5 = UpSampling2D(pool_size)(conv_24)
    
    conv_25 = Conv2D(64, kernel, padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Conv2D(n_labels, 1, padding="same")(conv_25)
    conv_26 = BatchNormalization()(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Done building decoder..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model
