from keras import backend as K
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np

# def precision(y_true, y_pred):
#     true_pos = K.sum((y_true > 0.5) & (y_pred > 0.5))
#     pred_pos = K.sum(y_pred > 0.5)
#     return true_pos / pred_pos

# def recall(y_true, y_pred):
#     true_pos = K.sum((y_true > 0.5) & (y_pred > 0.5))
#     pos = K.sum(y_true > 0.5)
#     return true_pos / pos

# def f1(y_true, y_pred):
#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     return 2 * (p * r) / (p + r)

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
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))

def true_area(y_true, y_pred):
    area = K.sum(y_true)
    return area

def pred_area(y_true, y_pred):
    area = K.sum(y_pred)
    return area

def iou(y_true, y_pred):
#     if K.sum(y_true) == K.sum(y_pred) == 0:
#         return 1.0
    
    i = K.sum(K.round(K.clip(y_pred*y_true, 0, 1)))
    u = K.sum(K.round(K.clip(y_pred+y_true, 0, 1))) + K.epsilon()  # avoid division by zero
    return i/u

def f2(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    return 5*((p*r)/(4*p+r+K.epsilon()))
    
    
#     thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
#     if K.sum(y_true) == K.sum(y_pred) == 0:
#         return 1.0
        
#     f2_total = 0
#     ious = {}
#     for t in thresholds:
#         tp,fp,fn = 0,0,0
        
#         fn = 
        
#         K.map_fn(,y_true)
        
        
#         for i,mt in enumerate(y_true):
#             found_match = False
#             for j,mp in enumerate(y_pred):
#                 key = 100 * i + j
#                 if key in ious.keys():
#                     miou = ious[key]
#                 else:
#                     miou = iou(mt, mp)
#                     ious[key] = miou  # save for later
#                 if miou >= t:
#                     found_match = True
#             if not found_match:
#                 fn += 1
                
#         for j,mp in enumerate(y_pred):
#             found_match = False
#             for i, mt in enumerate(y_true):
#                 miou = ious[100*i+j]
#                 if miou >= t:
#                     found_match = True
#                     break
#             if found_match:
#                 tp += 1
#             else:
#                 fp += 1
#         f2 = (5*tp)/(5*tp + 4*fn + fp)
#         f2_total += f2
    
#     return f2_total/len(thresholds)
    

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
#         print("Predict:")
#         print(val_predict)
#         print("Actual:")
#         print(val_targ)
        print(" — val_f1: %f — val_f2: %f — val_iou: %f — val_precision: %f — val_recall %f\n".format(_val_f1, _val_f2, _val_iou, _val_precision, _val_recall))
#         print("Predict:\n" + val_predict + "\nActual:\n" + val_targ)