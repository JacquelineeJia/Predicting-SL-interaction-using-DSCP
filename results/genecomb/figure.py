import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import r2_score
from sklearn.metrics import precision_recall_curve
import sys
import matplotlib.pyplot as plt
import seaborn as sns


thresh = 30

# Read in data

def get_method_results(method):
  y_test = np.squeeze(np.load(method+'/y_test.npy'))
  y_test_pred = np.squeeze(np.load(method+'/y_test_pred.npy'))
  y_test = (np.load(method+'/y_test.npy')>0).astype(int)
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


def print_fig(methods,display,fname):
  
	pr_mean = []
	r2_mean = []
	pr_std = []
	r2_std = []
	for i in range(len(methods)):
	  method = methods[i]
	  display = displays[i]
	  method_pr,method_r2 = get_method_results(method)
	  pr_mean.append(np.mean(method_pr))
	  r2_mean.append(np.mean(method_r2))
	  pr_std.append(np.std(method_pr))
	  r2_std.append(np.std(method_r2))

	plt.yticks(ticks = np.arange(0,1,0.01),minor = True)
	plt.ylim(0,1)
	plt.bar(range(len(displays)),pr_mean,color = 'firebrick',yerr = pr_std,edgecolor = 'black')
	plt.bar(range(len(displays)+1,len(displays)*2+1),r2_mean,color = 'darkcyan',yerr = r2_std,edgecolor = 'black')
	plt.xticks(list(range(len(displays))) + list(range(len(displays)+1,len(displays)*2+1)),displays*2,rotation = 315,ha = 'left')
	plt.legend(['PR-AUC', 'R2'],ncols = 2,frameon = False)
	sns.despine(fig = plt.gcf())
	plt.savefig(fname,bbox_inches = 'tight')
	plt.close()

methods = ['dscp','deepsynergy','dtf']
displays = ['DSCP','DeepSynergy','DTF']
print_fig(methods,displays,"Genecomb.png")

methods = ['dscp','deepsynergy','dtf']
displays = ['DSCP','DeepSynergy','DTF','CoSTCo','CP-WOPT','ProDeepSyn']
print_fig(methods,displays,"merick_dscp.png")
