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


def mt_loop(fpath1, fpath2, weights, modelname, lexicon, train1, train2, test1, test2, output_dir1, pred_path1, pred_path2, epochs, gpu_id):

	folds1 = cross_val_files(fpath1)
	folds2 = cross_val_files(fpath2)

	# test on fold t
	for t in range(10): # outer loop = isolating test fold
		print('On testfold' + str(t+1))

		# train on everything except t
		f_train1 = open(train1 + 'trainfold_bertje.txt', 'w', encoding='utf-8')
		f_train1.write('id\ttext\tlabel\n')
		for tr in range(10):
			if tr != t:
				for element in folds1[tr]:
					f_train1.write(element + '\n')
		f_train1.close()

		f_train2 = open(train2 + 'trainfold_bertje.txt', 'w', encoding='utf-8')
		f_train2.write('id\ttext\tlabel\n')
		for tr in range(10):
			if tr != t:
				for element in folds2[tr]:
					f_train2.write(element + '\n')
		f_train2.close()


		train_1 = train1 + 'trainfold_bertje.txt'
		train_2 = train2 + 'trainfold_bertje.txt'
		test_1 = test1 + 'fold' + str(t+1) + '_bertje.txt'
		test_2 = test2 + 'fold' + str(t+1) + '_bertje.txt'
		
		output_dir =  output_dir1
		
		
		# Change config file
		f = open("config_mt_default.json", 'r')
		default = f.read()
		f.close()
		
		default = default.replace('"weights": "default"', '"weights": "' + weights + '"')
		default = default.replace('"lexicon": "default"', '"lexicon": "' + lexicon + '"')
		default = default.replace('"train1": "default"', '"train1": "' + train_1 + '"')
		default = default.replace('"train2": "default"', '"train2": "' + train_2 + '"')
		default = default.replace('"test1": "default"', '"test1": "' + test_1 + '"')
		default = default.replace('"test2": "default"', '"test2": "' + test_2 + '"')
		default = default.replace('"output_dir": "default"', '"output_dir": "' + output_dir + '"')
		default = default.replace('"epochs": "default"', '"epochs": ' + str(epochs))
		
		f = open("config_mt_run.json", 'w')
		f.write(default)
		f.close()
		
		
		# train
		os.system("CUDA_VISIBLE_DEVICES=" + gpu_id + " python -m torch.distributed.launch --nproc_per_node 1 ../MT2/predict.py config_mt_run.json")
		
		# inference on test
		modelpath = output_dir + 'model.pth'
		
		if lexicon == "False":
			pred_path_test1 =  pred_path1 + modelname + "_testfold" + str(t+1) + "_test.txt"
			pred_path_test2 =  pred_path2 + modelname + "_testfold" + str(t+1) + "_test.txt"
		else:
			pred_path_test1 =  pred_path1 + modelname + "_testfold" + str(t+1) + "_withlex_test.txt"
			pred_path_test2 =  pred_path2 + modelname + "_testfold" + str(t+1) + "_withlex_test.txt"
		
		f = open("config_mt_run.json", 'r')
		run = f.read()
		f.close()
		
		run = run.replace('"pred_path1": "default"', '"pred_path1": "' + pred_path_test1 + '"')
		run = run.replace('"pred_path2": "default"', '"pred_path2": "' + pred_path_test2 + '"')

		f = open("config_mt_run.json", 'w')
		f.write(run)
		f.close()
		
		os.system("CUDA_VISIBLE_DEVICES=" + gpu_id + " python -m torch.distributed.launch --nproc_per_node 1 ../MT2/predict.py config_mt_run.json --infer " + modelpath)
		
		
		# delete models
		
		# AANPASSEN 'output_dir'
		os.system("rm " + output_dir + "/*")
		
		
		# clear variables
		del train_1, train_2, test_1, test_2
		del output_dir
		del modelpath
		del default, run
		del pred_path_test1, pred_path_test2

	return None

# gold datasets
fpath1 = 'tweets_vad_all.txt'
fpath2 =  "tweets_cat_all.txt"

weights = "pdelobelle/robbert-v2-dutch-base"
modelname = "robbert"

epochs = 10

lexicon = "False"

# beginning of filenames made in mt_loop
train1 = "tweets_vad_"
train2 = "tweets_cat_"
test1 =  "tweets_vad_"
test2 =  "tweets_cat_"

# path for saving output and predictions
output_dir1 =  "/home/luna/transformers3_8/PIVOT/output/"
pred_path1 =  "predictions/tweets_mt_vad75/"
pred_path2 =  "predictions/tweets_mt_cat25/"


gpu_id = '2'

mt_loop(fpath1, fpath2, weights, modelname, lexicon, train1, train2, test1, test2, output_dir1, pred_path1, pred_path2, epochs, gpu_id)