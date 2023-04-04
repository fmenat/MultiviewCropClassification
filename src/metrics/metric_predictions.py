from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, accuracy_score, balanced_accuracy_score,roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import numpy as np

# ALL CLASSIFICATION METRICS HAS (y_pre, y_true) order!!
def OverallAccuracy():
	def metric(y_pred, y_true):
		return accuracy_score(y_true, y_pred)
	return metric

def AverageAccuracy():
	def metric(y_pred, y_true):
		return balanced_accuracy_score(y_true, y_pred)
	return metric

def F1Score(average):
    average = None if average=="none" else average
    def ins_func(y_pred, y_true):
        return f1_score(y_true, y_pred, average=average)
    return ins_func
def Precision(average):
    average = None if average=="none" else average
    def ins_func(y_pred, y_true):
        return precision_score(y_true, y_pred, average=average)
    return ins_func
def Recall(average):
    average = None if average=="none" else average
    def ins_func(y_pred, y_true):
        return recall_score(y_true, y_pred, average=average)
    return ins_func

def Kappa():
	def metric(y_pred, y_true):
		return cohen_kappa_score(y_true, y_pred)
	return metric

def ConfusionMatrix():
	def metric(y_pred, y_true):
		return confusion_matrix(y_true, y_pred)
	return metric

def get_n_data(y_pred,y_true):
    labels_n = np.unique(y_true)
    return {v: (np.sum(y_pred==v),np.sum(y_true==v)) for v in labels_n}

def get_n_true(y_pred,y_true):
    labels_n = np.unique(y_true)
    return [np.sum(y_true==v) for v in labels_n]

def get_n_pred(y_pred,y_true):
    labels_n = np.unique(y_true)
    return [ np.sum(y_pred==v) for v in labels_n]


#METRICS FOR GIVING PROBABILITIES AS OUTPUT
def ROC_AUC(average):
    average = None if average=="none" else average
    def ins_func(y_pred, y_true):
        return roc_auc_score(y_true, y_pred[:,1], average=average)
    return ins_func

def CatEntropy(): #normalized
    def metric(y_pred, y_true): #it only use y_pred
        if len(y_pred.shape) > 1:
            K = y_pred.shape[1]
            y_pred = np.clip(y_pred, 1e-10, 1.0)
            entropy = - (y_pred * np.log(y_pred)).sum(axis=-1) / np.log(K)
            return entropy.mean(axis=0)
    return metric

def P_max(): 
    def metric(y_pred, y_true): #it only use y_pred
        if len(y_pred.shape) == 2:
            N, K = y_pred.shape
            p_max_x =  np.max(y_pred, axis =-1)
            return p_max_x.mean(axis=0) 
    return metric

def LogP(): #un-normalized
    def metric(y_pred, y_true):
        if len(y_pred.shape) == 2:
            N, K = y_pred.shape
            y_pred = np.clip(y_pred, 1e-10, 1.0)
            return ( np.log(y_pred[np.arange(N), y_true]) ).mean(axis=0)
    return metric

def P_true(): 
    def metric(y_pred, y_true):
        if len(y_pred.shape) == 2:
            N, K = y_pred.shape
            return ( y_pred[np.arange(N), y_true] ).mean(axis=0) 
    return metric




def R2Score():
	def metric(y_pred, y_true):
		return r2_score(y_true, y_pred)
	return metric

def MAE():
	def metric(y_pred, y_true):
		return mean_absolute_error(y_true, y_pred)
	return metric

def RMSE():
	def metric(y_pred, y_true):
		return np.sqrt(mean_squared_error(y_true, y_pred))
	return metric

def MAPE():
	def metric(y_pred, y_true):
		return mean_absolute_percentage_error(y_true, y_pred)
	return metric
