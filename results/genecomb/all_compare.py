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
from scipy.stats import ttest_ind,shapiro,mannwhitneyu
import sys

method_one = sys.argv[1]
method_two= sys.argv[2]
thresh = 30

def get_method_results(method):
	# Read in data

	y_test = np.load(method+'/y_test.npy')
	y_train = np.load(method+'/y_train.npy')
	y_train_pred = np.load(method+'/y_train_pred.npy')
	y_test_pred = np.load(method+'/y_test_pred.npy')

	# Calculate Youden's J Statistic

	y_test_thresh = (y_test >= thresh).astype('int')
	y_train_thresh = (y_train >= thresh).astype('int')

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
		
		return [roc_auc,pr_auc,recall,prec,f1,bacc,kappa]	


	def reg_met(true,pred):

		mae = []
		r2 = []
		mape = []
		mdae = []
		rmse = []

		for i in range(10):

			mae.append(mean_absolute_error(true[i],pred[i]))
			r2.append(r2_score(true[i],pred[i]))
			mape.append(mean_absolute_percentage_error(true[i],pred[i]))
			mdae.append(median_absolute_error(true[i],pred[i]))
			rmse.append(mean_squared_error(true[i],pred[i])**0.5)


		return [r2,rmse,mae,mdae]


	return reg_met(y_test,y_test_pred) + class_met(y_test_thresh,y_test_pred,y_test_pred_thresh)

method_one_metrics = get_method_results(method_one)
method_two_metrics = get_method_results(method_two)

print()
print()
print('roc_auc p-value')
test = ttest_ind(method_one_metrics[4],method_two_metrics[4])
print("Red: " + str(test.statistic > 0) + ", Significant: " + str(test.pvalue < 0.05))
print()


print('pr_auc p-value')
test = ttest_ind(method_one_metrics[5],method_two_metrics[5])
print("Red: " + str(test.statistic > 0) + ", Significant: " + str(test.pvalue < 0.05))
print()

print('recall p-value')
test = ttest_ind(method_one_metrics[6],method_two_metrics[6])
print("Red: " + str(test.statistic > 0) + ", Significant: " + str(test.pvalue < 0.05))
print()

print('precision p-value')
test = ttest_ind(method_one_metrics[7],method_two_metrics[7])
print("Red: " + str(test.statistic > 0) + ", Significant: " + str(test.pvalue < 0.05))
print()

print('f1 p-value')
test = ttest_ind(method_one_metrics[8],method_two_metrics[8])
print("Red: " + str(test.statistic > 0) + ", Significant: " + str(test.pvalue < 0.05))
print()

print('bacc p-value')
test = ttest_ind(method_one_metrics[9],method_two_metrics[9])
print("Red: " + str(test.statistic > 0) + ", Significant: " + str(test.pvalue < 0.05))
print()

print('kappa p-value')
test = ttest_ind(method_one_metrics[10],method_two_metrics[10])
print("Red: " + str(test.statistic > 0) + ", Significant: " + str(test.pvalue < 0.05))
print()

print()
print()

print('r2 p-value')
test = ttest_ind(method_one_metrics[0],method_two_metrics[0])
print("Red: " + str(test.statistic > 0) + ", Significant: " + str(test.pvalue < 0.05))
print()

print('rmse p-value')
test = ttest_ind(method_one_metrics[1],method_two_metrics[1])
print("Red: " + str(test.statistic < 0) + ", Significant: " + str(test.pvalue < 0.05))
print()

print('mae p-value')
test = ttest_ind(method_one_metrics[2],method_two_metrics[2])
print("Red: " + str(test.statistic < 0) + ", Significant: " + str(test.pvalue < 0.05))
print()

print('mdae p-value')
test =  ttest_ind(method_one_metrics[3],method_two_metrics[3])
print("Red: " + str(test.statistic < 0) + ", Significant: " + str(test.pvalue < 0.05))
print()
print()
