import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score 
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import sys

method = sys.argv[1]
train = sys.argv[2] == 'train'
thresh = 10

# Read in data

y_test = np.load(method+'/y_test.npy')
y_train = np.load(method+'/y_train.npy')
y_train_pred = np.load(method+'/y_train_pred.npy')
y_test_pred = np.load(method+'/y_test_pred.npy')

# Calculate Youden's J Statistic

y_test_thresh = (y_test >= thresh).astype('int')
y_train_thresh = (y_train >= thresh).astype('int')

print(y_train_thresh.sum())
fpr = {}
tpr = {}
thresholds ={}
roc_auc = {}

# Compute False Positive and True Positive Rates for each class
for i in range(10):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_train_thresh[i], y_train_pred[i], drop_intermediate=False)
    roc_auc[i] = auc(fpr[i], tpr[i])

J_stats = [None]*10
opt_thresholds = [None]*10

# Compute Youden's J Statistic for each trial
for i in range(10):
    J_stats[i] = tpr[i] - fpr[i]
    opt_thresholds[i] = thresholds[i][np.argmax(J_stats[i])]

print(opt_thresholds)

# Thresh hold train and test using the optimal threshold values
y_train_pred_thresh = np.copy(y_train_pred)
y_test_pred_thresh = np.copy(y_test_pred)
for i in range(10):
	y_train_pred_thresh[i] = (y_train_pred_thresh[i] >= opt_thresholds[i]).astype('int')
	y_test_pred_thresh[i] = (y_test_pred_thresh[i] >= opt_thresholds[i]).astype('int')


def class_met(true,pred,thresh):

	roc_auc = []
	pr_auc = []
	prec = []
	recall = []
	f1 = []
	bacc = []
	kappa = []

	for i in range(10):
		roc_auc.append(roc_auc_score(true[i],pred[i]))
		p, r, thresholds  = precision_recall_curve(true[i],pred[i])
		pr_auc.append(auc(r,p))
		prec.append(precision_score(true[i],thresh[i]))
		recall.append(recall_score(true[i],thresh[i]))
		f1.append(f1_score(true[i],thresh[i]))
		bacc.append(balanced_accuracy_score(true[i],thresh[i]))
		kappa.append(cohen_kappa_score(true[i],thresh[i])) 
	
	print("ROC-AUC: {:0.2f}+-{:0.2f}".format(round(np.mean(roc_auc),2),round(np.std(roc_auc),2)))
	print("PR-AUC: {:0.2f}+-{:0.2f}".format(round(np.mean(pr_auc),2),round(np.std(pr_auc),2)))
	print("Recall: {:0.2f}+-{:0.2f}".format(round(np.mean(recall),2),round(np.std(recall),2)))
	print("Precision: {:0.2f}+-{:0.2f}".format(round(np.mean(prec),2),round(np.std(prec),2)))
	print("F1: {:0.2f}+-{:0.2f}".format(round(np.mean(f1),2),round(np.std(f1),2)))
	print("BACC: {:0.2f}+-{:0.2f}".format(round(np.mean(bacc),2),round(np.std(bacc),2)))
	print("Kappa: {:0.2f}+-{:0.2f}".format(round(np.mean(kappa),2),round(np.std(kappa),2)))


def reg_met(true,pred):

	mae = []
	r2 = []
	pr2 = []
	sr2 = []
	mape = []
	mdae = []
	rmse = []

	for i in range(10):

		mae.append(mean_absolute_error(true[i],pred[i]))
		r2.append(r2_score(true[i],pred[i]))
		pr2.append(pearsonr(true[i],pred[i])[0])
		sr2.append(spearmanr(true[i],pred[i])[0])
		mape.append(mean_absolute_percentage_error(true[i],pred[i]))
		mdae.append(median_absolute_error(true[i],pred[i]))
		rmse.append(mean_squared_error(true[i],pred[i])**0.5)


	print("R2: {:0.2f}+-{:0.2f}".format(round(np.mean(r2),2),round(np.std(r2),2)))
	print("RMSE: {:0.2f}+-{:0.2f}".format(round(np.mean(rmse),2),round(np.std(rmse),2)))
	print("MAE: {:0.2f}+-{:0.2f}".format(round(np.mean(mae),2),round(np.std(mae),2)))
	print("MDAE: {:0.2f}+-{:0.2f}".format(round(np.mean(mdae),2),round(np.std(mdae),2)))

if train:
  print('\nTRAIN Regression Metrics:')
  reg_met(y_train,y_train_pred)
if not train:
  print('\nTest Regression Metrics:')
  reg_met(y_test,y_test_pred)
if train:
  print('\nTRAIN Classification Metrics:')
  class_met(y_train_thresh,y_train_pred,y_train_pred_thresh)
if not train:
  print('\nTest Classification Metrics:')
  class_met(y_test_thresh,y_test_pred,y_test_pred_thresh)
