import os
import json
import pickle
import re
import csv
import collections
import string

import numpy as np
import pandas as pd


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import DutchStemmer
from nltk import ngrams, skipgrams, FreqDist, pos_tag
from nltk.corpus import wordnet, stopwords

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, f_regression
from sklearn.feature_selection import mutual_info_classif

from scipy.stats import pearsonr

import warnings
warnings.filterwarnings("ignore")

# Save and load pickle objects
def save_object(obj, fpath):
	with open(fpath, 'wb') as o:
		pickle.dump(obj, o)

def load_object(fpath):
	with open(fpath, 'rb') as i:
		return pickle.load(i)

# Split files for cross validation
def cross_val_files(fpath):
	f = open(fpath, encoding='utf-8')
	lines = f.read().split('\n')
	lines = [line for line in lines[1:-1]]
	f.close()

	folds = []
	fold1 = lines[:100]
	fold2 = lines[100:200]
	fold3 = lines[200:300]
	fold4 = lines[300:400]
	fold5 = lines[400:500]
	fold6 = lines[500:600]
	fold7 = lines[600:700]
	fold8 = lines[700:800]
	fold9 = lines[800:900]
	fold10 = lines[900:1000]
	folds.extend([fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10])
	return folds

# Load fold of gold dataset
def load_data(data_file, fold):
	folds = cross_val_files(data_file)
	lines = []
	for f in fold:
		lines = lines + folds[f]
	lines = [line.split('\t') for line in lines]
	return lines

# Load all data in gold dataset file
def load_alldata(data_file):
	f = open(data_file, encoding='utf-8')
	lines = f.read().split('\n')
	lines = [line.split('\t') for line in lines[1:-1]]
	f.close()
	return lines

# Load files with predictions
def load_predictions(f_pred_dir, model, testfold, mode):
	all_folds = [0,1,2,3,4,5,6,7,8,9]
	tfold = 'testfold' + str(testfold+1) + '_'
	dfolds = [('devfold' + str(f+1) + '_') for f in all_folds if f != testfold]

	all_preds = []

	if mode == 'train':
		fs_pred = []
		for devfold in dfolds:
			for filename in os.listdir(f_pred_dir):
				if 'dev.' in filename:
					if tfold in filename:
						if devfold in filename:
							if model in filename:
								if 'withlex' not in filename:
									fs_pred.append(f_pred_dir + '/' + str(filename))
		for file in fs_pred:
			f = open(file, 'rt', encoding='utf-8')
			lines = f.read().split('\n')
			lines = [line.split(',"') for line in lines[1:-1]]
			preds = [line[1].strip('"') for line in lines]
			f.close()
			for pred in preds:
				all_preds.append(pred)
	else:
		for filename in os.listdir(f_pred_dir):
			if 'test.' in filename:
				if model in filename:
					if tfold in filename:
						if 'withlex' not in filename:
							fs_pred = f_pred_dir + '/' + str(filename)
							#print(fs_pred)
		f = open(fs_pred, 'rt', encoding='utf-8')
		lines = f.read().split('\n')
		lines = [line.split(',"') for line in lines[1:-1]]
		preds = [line[1].strip('"') for line in lines]
		f.close()
		for pred in preds:
			all_preds.append(pred)
	
	all_preds = [pred.split(',') for pred in all_preds]
	
	new_all_preds = []
	for pred in all_preds:
		pred = [float(p) for p in pred]
		new_all_preds.append(pred)

	return new_all_preds

# Make dictionary with instance id, text instance, gold label and categorical and dimensional predictions
def make_data_dict(data_file, f_pred_dir1, f_pred_dir2, model1, model2, testfold, mode):
	all_folds = [0,1,2,3,4,5,6,7,8,9]
	if mode == 'train':
		fold = [f for f in all_folds if f != testfold]
	else:
		fold = [testfold]

	data_dict = {}

	data = load_data(data_file, fold)
	preds1 = load_predictions(f_pred_dir1, model1, testfold, mode)
	preds2 = load_predictions(f_pred_dir2, model2, testfold, mode)



	data_dict['id'] = []
	data_dict['text'] = []
	data_dict['y'] = []
	data_dict['predictions1'] = []
	data_dict['predictions2'] = []

	for element in data:
		data_dict['id'].append(element[0])
		data_dict['text'].append(element[1])
		data_dict['y'].append(element[2])
	data_dict['predictions1'] = preds1
	data_dict['predictions2'] = preds2
	return data_dict

# Function to calculate cost-corrected accuracy
def cost_corr_acc(y_true, y_pred):
	conf_m = np.array(confusion_matrix(y_true, y_pred))
	cost_m = np.array([[0, 2/3, 2/3, 2/3, 2/3, 2/3], [2/3, 0, 1/3, 1, 1, 1/3], [2/3, 1/3, 0, 1, 1, 1/3], [2/3, 1, 1, 0, 1/3, 1], [2/3, 1, 1, 1/3, 0, 1], [2/3, 1/3, 1/3, 1, 1, 0]])
	cost = np.sum(np.multiply(conf_m, cost_m))/np.sum(conf_m)
	ccacc = 1 - cost
	return ccacc

# 
def eval_all(train_dict, test_dict, dataset, feats):
	dims = {'tweets_cat': 6, 'tweets_vad': 3, 'captions_cat': 6, 'captions_vad': 3}

	x_train_pred1 = train_dict['predictions1'] # categorical predictions
	x_train_pred2 = train_dict['predictions2'] # dimensional predictions
	train_feats_dict = {'predictions1': x_train_pred1, 'predictions2': x_train_pred2}
	train_feats = [train_feats_dict[feat] for feat in feats]
	x_train = np.concatenate(tuple(train_feats), axis=1)
	y_train = train_dict['y']

	x_dev_pred1 = test_dict['predictions1'] # categorical predictions
	x_dev_pred2 = test_dict['predictions2'] # dimensional predictions
	test_feats_dict = {'predictions1': x_dev_pred1, 'predictions2': x_dev_pred2}
	test_feats = [test_feats_dict[feat] for feat in feats]
	x_dev = np.concatenate(tuple(test_feats), axis=1)
	y_dev = test_dict['y']

	if dataset in ['tweets_vad', 'captions_vad']:
		dimensions = dims[dataset]
			
		y_train_dict = {}
		for i in range(dimensions):
			labels = []
			for label in y_train:
				labels.append(float(label.split(',')[i]))
			y_train_dict['y_train_' + str(i)] = labels
			
		y_dev_dict = {}
		for i in range(dimensions):
			labels = []
			for label in y_dev:
				labels.append(float(label.split(',')[i]))
			y_dev_dict['y_dev_' + str(i)] = labels
		
		accuracies = []
		final_label = []
		for i in range(dimensions):
			classifier = LinearRegression()
			classifier.fit(x_train, y_train_dict['y_train_' + str(i)])
			pred = classifier.predict(x_dev)
			metric = pearsonr(y_dev_dict['y_dev_' + str(i)], pred)[0]
			accuracies.append(metric)
			final_label.append(pred)
	
		final_label = np.asarray(final_label).T
		pred = final_label
		avg_accuracy = np.mean(accuracies)
		print(accuracies[0], accuracies[1], accuracies[2])
		metric = accuracies

	if dataset in ['tweets_cat', 'captions_cat']
		classifier = LinearSVC(random_state = 7, class_weight = 'balanced')
		classifier.fit(np.asarray(x_train), np.asarray(y_train))
		pred = classifier.predict(np.asarray(x_dev))
		metric = np.mean(pred == y_dev)
		micro_f1 = f1_score(np.asarray(y_dev), np.asarray(pred), average='micro')
		macro_f1 = f1_score(np.asarray(y_dev), np.asarray(pred), average='macro')
		ccacc = cost_corr_acc(y_dev, pred)
		print(macro_f1, micro_f1, ccacc)
		metric = micro_f1
	return metric, y_dev, pred




dataset = 'captions_cat'

data_file = 'subtitles_cat_all.txt'

f_pred_dir1 = "predictions/captions_cat/"
f_pred_dir2 = "predictions/captions_vad/"

model1 = 'robbert'
model2 = 'robbert'

all_metrics = []
all_pred = []
all_true = []

feats = [
'predictions1',
'predictions2'
]

for testfold in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:	
	train_dict = make_data_dict(data_file, f_pred_dir1, f_pred_dir2, model1, model2, testfold, mode='train')
	test_dict = make_data_dict(data_file, f_pred_dir1, f_pred_dir2, model1, model2, testfold, mode='test')
	metric, y_dev, pred = eval_all(train_dict, test_dict, dataset, feats)
	all_metrics.append(metric)
	for element in y_dev:
		all_true.append(element)
	for element in pred:
		all_pred.append(element)
	print('\n')

# target task is classification
if dataset in ['tweets_cat', 'captions_cat']:
	micro_f1 = f1_score(np.asarray(all_true), np.asarray(all_pred), average='micro')
	macro_f1 = f1_score(np.asarray(all_true), np.asarray(all_pred), average='macro')
	ccacc = cost_corr_acc(all_true, all_pred)
	print('***')
	print(macro_f1, micro_f1, ccacc)
	print(classification_report(all_true, all_pred))
	print(confusion_matrix(all_true, all_pred))
	for element in zip(all_true, all_pred):
		print(element[0], element[1])

# target task is regression
else:
	r = []
	for i in range(3):
		true = [float(element.split(',')[i]) for element in all_true]
		pred = [element[i] for element in all_pred]
		r.append(pearsonr(true, pred)[0])
	print(r)