import numpy as np
import pandas as pd


def modelToTable(model):
    layer_names = [layer.name for layer in model.layers]
    layer_input_shapes = [layer.input_shape[1:] for layer in model.layers]
    layer_input_dimensions = [np.prod(layer.input_shape[1:]) for layer in model.layers]
    layers_info = pd.DataFrame(
        {'Name': layer_names, 'Input shape': layer_input_shapes, 'Input dimension': layer_input_dimensions})
    return layers_info
