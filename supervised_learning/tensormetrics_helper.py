import tensorflow as tf
import ipdb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, accuracy_score, roc_auc_score

def tf_metrics(y_true, y_pred):
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	f1_score_value = f1_score(y_true, y_pred)
	confusion_matrix_value = confusion_matrix(y_true, y_pred)
	fpr, tpr, tresholds = roc_curve(y_true, y_pred)
	roc_auc = roc_auc_score(y_true, y_pred)
	return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score_value, "confusion_matrix": confusion_matrix_value, 'roc_auc_score': roc_auc}