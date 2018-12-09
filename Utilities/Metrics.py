from keras import backend as K
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np

def recall(y_true, y_pred):
	"""Recall metric.

	Only computes a batch-wise average of recall.

	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision(y_true, y_pred):
	"""Precision metric.

	Only computes a batch-wise average of precision.

	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1(y_true, y_pred):
    """F1 metric.

	Only computes a batch-wise average of f1 score.

	Computes the f1 score, a metric for quality of multi-label classification of
	all kinds, even with very specific class distribution.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))

def true_area(y_true, y_pred):
    """Area size of actual ship area.

	Only computes a batch-wise average of it.
    """
    area = K.sum(y_true)
    return area

def pred_area(y_true, y_pred):
    """Area size of predicted ship area.

	Only computes a batch-wise average of it.
    """
    area = K.sum(y_pred)
    return area

def iou(y_true, y_pred):
    """IoU(Intersect over Union) metric.

	Only computes a batch-wise average of IoU.

	Computes the IoU score, a metric for quality of segmentation classification models.
    """

    i = K.sum(K.round(K.clip(y_pred*y_true, 0, 1)))
    u = K.sum(K.round(K.clip(y_pred+y_true, 0, 1))) + K.epsilon()  # avoid division by zero
    return i/u

def f2(y_true, y_pred):
    """F2 metric.

	Only computes a batch-wise average of f2 score.

	Computes the f2 score, a metric for quality of multi-label classification of
	all kinds, even with very specific class distribution.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    return 5*((p*r)/(4*p+r+K.epsilon()))
    

class MetricsCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_f2s = []
        self.val_ious = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1(val_targ, val_predict)
        _val_f2 = f2(val_targ, val_predict)
        _val_iou = iou(val_targ, val_predict)
        _val_recall = recall(val_targ, val_predict)
        _val_precision = precision(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_f2s.append(_val_f2)
        self.val_ious.append(_val_iou)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_f2: %f — val_iou: %f — val_precision: %f — val_recall %f\n".format(_val_f1, _val_f2, _val_iou, _val_precision, _val_recall))
