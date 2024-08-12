import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_absolute_percentage_error as  r2_score
from sklearn.metrics import precision_recall_curve
import sys
from scipy.stats import ttest_ind,shapiro,mannwhitneyu

method_1 = sys.argv[1]
method_2 = sys.argv[2]
thresh = 30

# Read in data

def get_method_results(method):
  y_test = np.squeeze(np.load(method+'/y_test.npy'))
  y_test_pred = np.squeeze(np.load(method+'/y_test_pred.npy'))
  y_test = np.load(method+'/y_test.npy')
  y_test_pred = np.load(method+'/y_test_pred.npy')

  return class_met(y_test > thresh,y_test_pred),reg_met(y_test ,y_test_pred)

def class_met(true,pred):

	pr_auc = []

	for i in range(10):
		p, r, thresholds  = precision_recall_curve(true[i],pred[i])
		pr_auc.append(auc(r,p))
	
	return pr_auc

def reg_met(true,pred):

	r2 = []

	for i in range(10):

		r2.append(r2_score(true[i],pred[i]))


	return r2

method_one_pr,method_one_r2 = get_method_results(method_1)
method_two_pr,method_two_r2 = get_method_results(method_2)
pr_diff = [abs(x-y) for x,y in zip(method_one_pr,method_two_pr)]
r2_diff = [abs(x-y) for x,y in zip(method_one_r2,method_two_r2)]

print()
print(method_1)
print("PR: {:0.2f}+-{:0.2f}".format(round(np.mean(method_one_pr),2),round(np.std(method_one_pr),2)))
print("r2: {:0.2f}+-{:0.2f}".format(round(np.mean(method_one_r2),2),round(np.std(method_one_r2),2)))

print()
print(method_2)
print("PR: {:0.2f}+-{:0.2f}".format(round(np.mean(method_two_pr),2),round(np.std(method_two_pr),2)))
print("r2: {:0.2f}+-{:0.2f}".format(round(np.mean(method_two_r2),2),round(np.std(method_two_r2),2)))

print()
print("DIFF")
print("PR: {:0.2f}+-{:0.2f}".format(round(np.mean(pr_diff),2),round(np.std(pr_diff),2)))
print(ttest_ind(method_one_pr,method_two_pr,alternative = 'greater'))
print()
print("r2: {:0.2f}+-{:0.2f}".format(round(np.mean(r2_diff),2),round(np.std(r2_diff),2)))
print(ttest_ind(method_one_r2,method_two_r2,alternative = 'greater'))
print()
