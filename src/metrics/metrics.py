import numpy as np

from .metric_predictions import OverallAccuracy,F1Score,Precision,Recall,Kappa, ConfusionMatrix,AverageAccuracy,ROC_AUC
from .metric_predictions import R2Score, MAPE, RMSE, MAE
from .metric_predictions import get_n_data, get_n_true, get_n_pred
from .metric_predictions import CatEntropy,LogP,P_max, P_true

class BaseMetrics(object): 
	"""central metrics class to provide standard metric types"""

	def __init__(self, task_type="classification", metric_types=[]):
		"""build BaseMetrics

		Parameters
		----------
		metric_types : list of str, optional
		    declaration of metric types to be used, by default all are used
		"""
		self.task_type = task_type
		self.metric_types = [v.lower() for v in metric_types]
		self.metric_dict = {}

	def __call__(self, prediction, target):
		"""call forward for each metric in collection

		Parameters
		----------
		prediction : array_like (n_samples, n_outputs)
		    prediction tensor
		target : array_like     (n_samples, n_outputs)
		    ground truth tensor, in classification: n_outputs=1, currently working only =1
		"""
		if not isinstance(prediction, np.ndarray):
			prediction = np.asarray(prediction)
		if not isinstance(target, np.ndarray):
			target = np.asarray(target)
			
		if self.task_type=="classification" and hasattr(self,"aux_metric"):
			self.n_samples = self.aux_metric(prediction,target)	
		else:
			self.n_samples = []		

		#forward over all metrics
		return {name: func(prediction, target) for (name, func) in self.metric_dict.items()}

	def get_metric_types(self):
		"""return list of metric types inside collection

		Returns
		-------
		list of strings
		"""
		return list(self.metric_dict.keys())

	def reverse_forward(self, target,prediction):
		return self(prediction, target)



class ClassificationMetrics(BaseMetrics): 
    def __init__(self, metric_types=["OA","AA","KAPPA", "F1 MACRO","P MACRO","ENTROPY","LOGP"]): # "R MACRO" == "AA"
        """build ClassificationMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(ClassificationMetrics,self).__init__("classification", metric_types)
        self.aux_metric = get_n_data
        for metric in self.metric_types:
            if "oa"==metric:
                self.metric_dict["OA"] = OverallAccuracy()
            elif "aa"==metric or "r macro" == metric or "recall"==metric:
                self.metric_dict["AA"] = Recall("macro") #AverageAccuracy()
            elif "f1" in metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"F1 {avg_mode.upper()}"] = F1Score(avg_mode)        
            elif "p " in metric or "precision"==metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"P {avg_mode.upper()}"] = Precision(avg_mode)
            elif "r " in metric or "recall"==metric :
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"R {avg_mode.upper()}"] = Recall(avg_mode)    
            elif "kappa"==metric:
                self.metric_dict["KAPPA"] = Kappa()
            elif "confusion" in metric or "matrix" in metric:
                self.metric_dict["MATRIX"] = ConfusionMatrix()
            elif "ntrue"==metric:
                self.metric_dict["N TRUE"] = get_n_true
            elif "npred"==metric:
                self.metric_dict["N PRED"] = get_n_pred


class SoftClassificationMetrics(BaseMetrics): 
    def __init__(self, metric_types=["ENTROPY","LOGP", "PMAX", "PTRUE", "AUC"]):
        """build SoftClassificationMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(SoftClassificationMetrics,self).__init__("classification", metric_types)
        for metric in self.metric_types:
            if "auc" in metric or "roc-auc"==metric:
                avg_mode = metric.split(" ")[1] if len(metric.split(" "))!=1 else "macro"
                self.metric_dict[f"AUC {avg_mode.upper()}"] = ROC_AUC(avg_mode)   
            elif "entropy" == metric:
                self.metric_dict["ENTROPY"] = CatEntropy()
            elif "logp"==metric:
                self.metric_dict["LOGp"] = LogP()
            elif "pmax"==metric:
                self.metric_dict["Pmax"] = P_max()
            elif "ptrue"==metric:
                self.metric_dict["Ptrue"] = P_true()
            


class RegressionMetrics(BaseMetrics): 
    def __init__(self, metric_types=["R2","RMSE","MAE", "MAPE"]):
        """build RegressionMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(RegressionMetrics,self).__init__("regression", metric_types)  

        for metric in self.metric_types:
            if "r2"==metric:
                self.metric_dict["R2"] = R2Score()
            elif "mae"==metric:
                self.metric_dict["MAE"] = MAE()
            elif "rmse"==metric:
                self.metric_dict["RMSE"] = RMSE()
            elif "mape"==metric:
                self.metric_dict["MAPE"] = MAPE()