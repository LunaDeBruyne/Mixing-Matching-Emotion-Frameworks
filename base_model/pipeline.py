# Split files for cross validation
def cross_val_files(fpath):
	f = open(fpath, encoding='utf-8')
	lines = f.read().split('\n')
	lines = lines[1:-1]
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

import os


def cat_loop(fpath, weights, modelname, lexicon, train1, test1, output_dir1, pred_path1, epochs, gpu_id):

	folds = cross_val_files(fpath)

	# test on fold t
	for t in range(10): # outer loop = isolating test fold
		print('On testfold' + str(t+1))

		# train on everything except t
		f_train = open(train1 + 'trainfold_bertje.txt', 'w', encoding='utf-8')
		f_train.write('id\ttext\tlabel\n')
		for tr in range(10):
			if tr != t:
				for element in folds[tr]:
					f_train.write(element + '\n')
		f_train.close()
		train =  train1 + 'trainfold_bertje.txt'
		test = test1 + 'fold' + str(t+1) + '_bertje.txt'
		
		output_dir =  output_dir1
		
		
		# Change config file
		
		f = open("config_cat_default.json", 'r')
		default = f.read()
		f.close()
		
		default = default.replace('"weights": "default"', '"weights": "' + weights + '"')
		default = default.replace('"lexicon": "default"', '"lexicon": "' + lexicon + '"')
		default = default.replace('"train": "default"', '"train": "' + train + '"')
		default = default.replace('"test": "default"', '"test": "' + test + '"')
		default = default.replace('"output_dir": "default"', '"output_dir": "' + output_dir + '"')
		default = default.replace('"epochs": "default"', '"epochs": ' + str(epochs))
		
		f = open("config_cat_run.json", 'w')
		f.write(default)
		f.close()
		
		
		# train
		os.system("CUDA_VISIBLE_DEVICES=" + gpu_id + " python -m torch.distributed.launch --nproc_per_node 1 ../transformers_withoutvalid/predict.py config_cat_run.json")
		
		# inference on test
		modelpath = output_dir + 'model.pth'
		
		if lexicon == "False":
			pred_path_test =  pred_path1 + modelname + "_testfold" + str(t+1) + "_test.txt"
		else:
			pred_path_test =  pred_path1 + modelname + "_testfold" + str(t+1) + "_withlex_test.txt"
		
		f = open("config_cat_run.json", 'r')
		run = f.read()
		f.close()
		
		run = run.replace('"pred_path": "default"', '"pred_path": "' + pred_path_test + '"')
		
		f = open("config_cat_run.json", 'w')
		f.write(run)
		f.close()
		
		os.system("CUDA_VISIBLE_DEVICES=" + gpu_id + " python -m torch.distributed.launch --nproc_per_node 1 ../transformers_withoutvalid/predict.py config_cat_run.json --infer " + modelpath)
		
		
		# delete models
		
		# AANPASSEN 'output_dir'
		os.system("rm " + output_dir + "/*")
		
		
		# clear variables
		del train, test
		del output_dir
		del modelpath
		del default, run
		del pred_path_test

	return None

# gold dataset
fpath =  "tweets_cat_all.txt"

weights = "pdelobelle/robbert-v2-dutch-base"
modelname = "robbert"

epochs = 5

lexicon = "False"

# beginning of filenames made in cat_loop
train1 =  "tweets_cat_"
test1 = "tweets_cat_"

# path for saving output and predictions
output_dir1 =  "output/"
pred_path1 =  "predictions/tweets_cat/"


gpu_id = '2'

#cat_loop(fpath, weights, modelname, lexicon, train1, test1, output_dir1, pred_path1, epochs, gpu_id)


def vad_loop(fpath, weights, modelname, lexicon, train1, test1, output_dir1, pred_path1, epcohs, gpu_id):

	folds = cross_val_files(fpath)

	# test on fold t
	for t in range(10): # outer loop = isolating test fold
		print('On testfold' + str(t+1))

		# train on everything except t
		f_train = open(train1 + 'trainfold_bertje.txt', 'w', encoding='utf-8')
		f_train.write('id\ttext\tlabel\n')
		for tr in range(10):
			if tr != t:
				for element in folds[tr]:
					f_train.write(element + '\n')
		f_train.close()
		train =  train1 + 'trainfold_bertje.txt'
		test = test1 + 'fold' + str(t+1) + '_bertje.txt'
		
		output_dir =  output_dir1
		
		
		# Change config file
		
		f = open("config_vad_default.json", 'r')
		default = f.read()
		f.close()
		
		default = default.replace('"weights": "default"', '"weights": "' + weights + '"')
		default = default.replace('"lexicon": "default"', '"lexicon": "' + lexicon + '"')
		default = default.replace('"train": "default"', '"train": "' + train + '"')
		default = default.replace('"test": "default"', '"test": "' + test + '"')
		default = default.replace('"output_dir": "default"', '"output_dir": "' + output_dir + '"')
		default = default.replace('"epochs": "default"', '"epochs": ' + str(epochs))
		
		f = open("config_vad_run.json", 'w')
		f.write(default)
		f.close()
		
		
		# train
		os.system("CUDA_VISIBLE_DEVICES=" + gpu_id + " python -m torch.distributed.launch --nproc_per_node 1 ../transformers_withoutvalid/predict.py config_vad_run.json")
		
		# inference on test
		modelpath = output_dir + 'model.pth'
	
		if lexicon == "False":
			pred_path_test =  pred_path1 + modelname + "_testfold" + str(t+1) + "_test.txt"
		else:
			pred_path_test =  pred_path1 + modelname + "_testfold" + str(t+1) + "_withlex_test.txt"
		
		f = open("config_vad_run.json", 'r')
		run = f.read()
		f.close()
		
		run = run.replace('"pred_path": "default"', '"pred_path": "' + pred_path_test + '"')
		
		f = open("config_vad_run.json", 'w')
		f.write(run)
		f.close()
		
		os.system("CUDA_VISIBLE_DEVICES=" + gpu_id + " python -m torch.distributed.launch --nproc_per_node 1 ../transformers_withoutvalid/predict.py config_vad_run.json --infer " + modelpath)
		
		
		# delete models
		
		# AANPASSEN 'output_dir'
		os.system("rm " + output_dir + "/*")
		
		
		# clear variables
		del train, test
		del output_dir
		del modelpath
		del default, run
		del pred_path_test

	return None

# gold dataset
fpath = '/home/luna/transformers/BERTICON/Data/subtitles_vad_all_bertje.txt'

weights = "pdelobelle/robbert-v2-dutch-base"
modelname = "robbert"

epochs = 10

lexicon = "False"

# beginning of filenames made in vad_loop
train1 =  "tweets_vad_"
test1 = "tweets_vad_"

# path for saving output and predictions
output_dir1 =  "output/"
pred_path1 =  "tweets_vad/"

gpu_id = '2'

#vad_loop(fpath, weights, modelname, lexicon, train1, test1, output_dir1, pred_path1, epochs, gpu_id)
