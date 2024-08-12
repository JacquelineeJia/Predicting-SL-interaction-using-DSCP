'''
This file compares each algorithm to its ID variant
'''

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import cohen_kappa_score as kappa
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import sys
from sklearn.metrics import roc_curve, auc

dataset = sys.argv[1]

method_one_list = {'merick':['deepsynergy',],
                   'merick_lodo':['deepsynergy'],
                   'merick_loclo':['deepsynergy'],
                   'almanac':['deepsynergy'],
                   'drugcomb':['deepsynergy']}[dataset]
method_two_list = {'merick':['deepsynergyid'],
                   'merick_lodo':['deepsynergyid'],
                   'merick_loclo':['deepsynergyid'],
                   'almanac':['deepsynergyid'],
                   'drugcomb':['deepsynergyid']}[dataset]
thresh = {'merick':30,
                   'merick_lodo':30,
                   'merick_loclo':30,
                   'almanac':10,
                   'drugcomb':10,}[dataset]


def threshold(method):

	y_test = np.load(dataset+'/'+method+'/y_test.npy')[0]
	y_train = np.load(dataset+'/'+method+'/y_train.npy')[0]
	y_train_pred = np.median(np.load(dataset+'/'+method+'/y_train_pred.npy'),axis = 0)
	y_test_pred = np.median(np.load(dataset+'/'+method+'/y_test_pred.npy'),axis = 0)
	y_train_pred = np.load(dataset+'/'+method+'/y_train_pred.npy')[0]
	y_test_pred = np.load(dataset+'/'+method+'/y_test_pred.npy')[0]

	min_ = np.min(y_train)
	max_ = np.max(y_train-min_)
 
	y_test_thresh = (y_test >= thresh).astype('int')
	y_train_thresh = (y_train >= thresh).astype('int')
	
	y_train_pred-=min_
	y_train_pred/=max_


	kappas = []
	for t in np.arange(np.min(y_train_pred),np.max(y_train_pred),0.0001):
	  kappas.append(kappa(y_train_thresh,y_train_pred > t))
	opt_thresholds = np.argmax(kappas)*0.0001

	opt_thresholds*=max_
	opt_thresholds+=min_
	#print(opt_thresholds)
	
	# Thresh hold train and test using the optimal threshold values
	y_test_pred_thresh = np.copy(y_test_pred)
	y_test_pred_thresh = (y_test_pred_thresh > opt_thresholds).astype('int')

	return y_test_thresh,np.squeeze(y_test_pred_thresh)

def threshold(method):

	y_test = np.load(dataset+'/'+method+'/y_test.npy')[0]
	y_train = np.load(dataset+'/'+method+'/y_train.npy')[0]
	y_train_pred = np.median(np.load(dataset+'/'+method+'/y_train_pred.npy'),axis = 0)
	y_test_pred = np.median(np.load(dataset+'/'+method+'/y_test_pred.npy'),axis = 0)

	# Calculate Youden's J Statistic

	y_test_thresh = (y_test >= thresh).astype('int')
	y_train_thresh = (y_train >= thresh).astype('int')

	fpr = {}
	tpr = {}
	thresholds ={}
	roc_auc = {}

	fpr, tpr, thresholds = roc_curve(y_train_thresh, y_train_pred, drop_intermediate=False)
	roc_auc = auc(fpr, tpr)

	J_stats = None
	opt_thresholds = None

	J_stats = tpr - fpr
	opt_thresholds = thresholds[np.argmax(J_stats)]	
	
	# Thresh hold train and test using the optimal threshold values
	y_test_pred_thresh = np.copy(y_test_pred)
	y_test_pred_thresh = (y_test_pred_thresh > opt_thresholds).astype('int')

	return y_test_thresh,np.squeeze(y_test_pred_thresh)

def get_reg_sig(method_one,method_two):
	def get_method_results(method):
	  y_test_pred = np.median(np.squeeze(np.load(dataset+'/'+method+'/y_test_pred.npy')),axis = 0)
	  y_test = np.squeeze(np.load(dataset+'/'+method+'/y_test.npy'))[0]

	  return np.abs((y_test_pred - y_test))

	method_one_test_pred = get_method_results(method_one)
	method_two_test_pred = get_method_results(method_two)
 
	results = wilcoxon(method_one_test_pred,method_two_test_pred,alternative = 'less')

	return results.pvalue

def get_mape(method_one,method_two):
	def get_method_results(method):
	  y_test_pred = np.median(np.squeeze(np.load(dataset+'/'+method+'/y_test_pred.npy')),axis = 0)
	  y_test = np.squeeze(np.load(dataset+'/'+method+'/y_test.npy'))[0]

	  method_mape = np.abs((y_test_pred - y_test)).mean()
	  return method_mape

	method_one_mape = get_method_results(method_one)	
	method_two_mape = get_method_results(method_two)
	diff = abs(method_one_mape - method_two_mape)
      
	return method_one_mape,method_two_mape,diff

def get_class_sig(method_one,method_two):

	test,method_one_pred = threshold(method_one)
	test,method_two_pred = threshold(method_two)

	method_one_pred = test == method_one_pred
	method_two_pred = test == method_two_pred

	table = [[0,0],[0,0]]

	table[0][0] = np.logical_and(method_one_pred,method_two_pred).sum()
	table[1][0] = np.logical_and(np.logical_not(method_one_pred),method_two_pred).sum()
	table[0][1] = np.logical_and(method_one_pred,np.logical_not(method_two_pred)).sum()
	table[1][1] = np.logical_and(np.logical_not(method_one_pred),np.logical_not(method_two_pred)).sum()

	results = mcnemar(table)
	return results.pvalue

def get_f1(method_one,method_two):

	test,method_one_pred = threshold(method_one)
	test,method_two_pred = threshold(method_two)
	method_one_f1 = kappa(test,method_one_pred)
	method_two_f1 = kappa(test,method_two_pred)
	diff = abs(method_one_f1-method_two_f1)

	return method_one_f1,method_two_f1,diff

def class_row(method_one,method_two):
  method_one_f1,method_two_f1,diff= get_f1(method_one,method_two)
  sig = get_class_sig(method_one,method_two) 
  print("Method: {}, Published F1 Score: {}, Random F1 Score: {}, Difference: {}, Significance: {}".format(
                 method_one,
                 round(method_one_f1,2),
                 round(method_two_f1,2),
                 round(diff,2),
                 round(sig,2)))

def reg_row(method_one,method_two):
  method_one_mape,method_two_mape,diff= get_mape(method_one,method_two)
  sig = get_reg_sig(method_one,method_two) 
  print("Method: {}, Published MAPE Score: {}, Random MAPE Score: {}, Difference: {}, Significance: {}".format(
                 method_one,
                 round(method_one_mape,2),
                 round(method_two_mape,2),
                 round(diff,2),
                 round(sig,2)))


print()
print('Classification Table')
for one,two in zip(method_one_list,method_two_list):
  class_row(one,two)

print()
print('Regression Table')
for one,two in zip(method_one_list,method_two_list):
  reg_row(one,two)
