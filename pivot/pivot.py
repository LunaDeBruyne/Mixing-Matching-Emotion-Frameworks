from scipy.spatial import distance
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC
import collections
import numpy as np

import os


# Valence, arousal and dominance scores for the terms anger, fear, happiness, love, and sadness in Mehrabian & Russell (1974). Neutral is considered the center of the VAD space.

# anger
v_anger = -0.51
a_anger = 0.59
d_anger = 0.25

# fear
v_fear = -0.64
a_fear = 0.60
d_fear = -0.43

# happy
v_joy = 0.81
a_joy = 0.51
d_joy = 0.46

# love
v_love = 0.82
a_love = 0.65
d_love = -0.05

# sadness
v_sadness = -0.63
a_sadness = -0.27
d_sadness = -0.33

# neutral
v_neutral = 0
a_neutral = 0
d_neutral = 0


# Scale the scores by Mehrabian & Russell to be in the [0-1] interval instead of [-1,1]
def scale(n):
	nn = ((n+1)/2)*1
	return nn


# function to map VAD scores to one of the categories anger, fear, joy, love, sadness or neutral, based on the smalles cosine distance.
def vad2cat_cos(vad):
	dists = {}
	dists['anger'] = distance.cosine([scale(v_anger), scale(a_anger), scale(d_anger)], vad)
	dists['fear'] = distance.cosine([scale(v_fear), scale(a_fear), scale(d_fear)], vad)
	dists['joy'] = distance.cosine([scale(v_joy), scale(a_joy), scale(d_joy)], vad)
	dists['love'] = distance.cosine([scale(v_love), scale(a_love), scale(d_love)], vad)
	dists['sadness'] = distance.cosine([scale(v_sadness), scale(a_sadness), scale(d_sadness)], vad)
	dists['neutral'] = distance.cosine([scale(v_neutral), scale(a_neutral), scale(d_neutral)], vad)
	return min(dists, key=dists.get)

# rule-based mapping from VAD scores to one of the categories anger, fear, joy, love, sadness or neutral.
	if vad[0] < .5 and vad[1] > .5 and vad[2] > .5:
		label = 'anger'
	elif vad[0] < .5 and vad[1] > .5 and vad[2] < .5:
		label = 'fear'
	elif vad[0] > .5 and vad[1] > .5 and vad[2] > .5:
		label = 'joy'
	elif vad[0] < .5 and vad[1] < .5 and vad[2] < .5:
		label = 'sadness'
	else:
		label = vad2cat_cos(vad)
	return label

# load files with predictions / gold standard
def load(file):
	f = open(file, 'rt', encoding='utf-8')
	lines = f.read().split('\n')
	lines = lines[1:-1]
	return lines

# put predicted VAD scores in list
def vad_preds(fs_pred):
	label_dict = {'anger': 1, 'fear': 2, 'joy': 3, 'love': 4, 'sadness': 5, 'neutral': 0}
	pred = []
	for f in fs_pred:
		lines = load(f)
		pred.extend(lines)
	pred_label_dict = {}
	for instance in pred:
		pred_label_dict[instance.split(',')[0]] = ','.join(instance.split(',')[1:])
	od_pred = collections.OrderedDict(sorted(pred_label_dict.items()))
	pred_labels = [instance.strip('"') for instance in od_pred.values()]
	new_pred_labels = []
	for instance in pred_labels:
		instance2 = instance.split(',')
		instance3 = []
		for element in instance2:
			instance3.append(float(element))
		new_pred_labels.append(label_dict[vad2cat(instance3)])
	return new_pred_labels

# put gold VAD scores in list
def vad_true(f_true):
	label_dict = {'anger': 1, 'fear': 2, 'joy': 3, 'love': 4, 'sadness': 5, 'neutral': 0}

	true = load(f_true)
	
	true_label_dict = {}
	for instance in true:
		true_label_dict[instance.split('\t')[0]] = [float(element) for element in (','.join(instance.split('\t')[2:])).split(',')]
	od_true = collections.OrderedDict(sorted(true_label_dict.items()))
	pred_labels = [item for item in od_true.values()]
	new_pred_labels = []
	for instance in pred_labels:
		new_pred_labels.append(label_dict[vad2cat(instance)])
	return new_pred_labels

# put gold categorical labels in list
def cat_true(f_true):
	true = load(f_true)
	
	true_label_dict = {}
	for instance in true:
		true_label_dict[instance.split('\t')[0]] = instance.split('\t')[2]
	od_true = collections.OrderedDict(sorted(true_label_dict.items()))
	true_labels = [int(item) for item in od_true.values()]
	return true_labels


# function to calculate F1 for individual class
def calc_f1(true, pred, c):
	TP = FP = FN = 0
	for item in list(zip(pred, true)):
		if item[0] == c and item[1] == c:
			TP += 1
		elif item[0] == c and item[1] != c:
			FP += 1
		elif item[0] != c and item[1] == c:
			FN += 1
	if (TP+FP) != 0:
		P = TP/(TP+FP)
	else:
		P = 0
	if (TP+FN) != 0:	
		R = TP/(TP+FN)
	else:
		R = 0
	if P+R != 0:
		F = (2*P*R)/(P+R)
	else:
		F = 0
	return F

# function to calculate cost-corrected accuracy
def cost_corr_acc(true_labels, pred_labels):
	conf_m = np.array(confusion_matrix(true_labels, pred_labels))
	cost_m = np.array([[0, 2/3, 2/3, 2/3, 2/3, 2/3], [2/3, 0, 1/3, 1, 1, 1/3], [2/3, 1/3, 0, 1, 1, 1/3], [2/3, 1, 1, 0, 1/3, 1], [2/3, 1, 1, 1/3, 0, 1], [2/3, 1/3, 1/3, 1, 1, 0]])
	cost = np.sum(np.multiply(conf_m, cost_m))/np.sum(conf_m)
	ccacc = 1 - cost
	return ccacc

# calculate accuracy, 
def cat_metrics(true_labels, pred_labels):
	macro_f1 = f1_score(true_labels, pred_labels, average='macro')
	acc = np.mean(np.asarray(true_labels) == np.asarray(pred_labels))
	ccacc = cost_corr_acc(true_labels, pred_labels)
	acc_per_class = []
	f1_per_class = []
	for i in range(6):
		f1_per_class.append(calc_f1(true_labels, pred_labels, i))
	classif_report = classification_report(true_labels, pred_labels)
	return macro_f1, acc, ccacc, f1_per_class


f_vad = "subtitles_vad_all.txt"
f_cat = "subtitles_cat_all.txt"

pred_labels = vad_true(f_vad)
true_labels = cat_true(f_cat)

print(cat_metrics(true_labels, pred_labels)[0]) # F1
print(cat_metrics(true_labels, pred_labels)[1]) # acc
print(cat_metrics(true_labels, pred_labels)[2]) # ccacc
print("***")
# F1 per class
#print(cat_metrics(true_labels, pred_labels)[3][0])
#print(cat_metrics(true_labels, pred_labels)[3][1])
#print(cat_metrics(true_labels, pred_labels)[3][2])
#print(cat_metrics(true_labels, pred_labels)[3][3])
#print(cat_metrics(true_labels, pred_labels)[3][4])
#print(cat_metrics(true_labels, pred_labels)[3][5])