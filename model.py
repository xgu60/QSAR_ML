
import numpy as np 
import pandas as pd 

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from plot import plot_corr


def preprocess():
		
	df = pd.read_csv("data/fingerprinters_4096.csv")
	df["ic50"] = - np.log10(df["ic50"] / 1E9)
	#df["ic50"] = - df["ic50"]
	df.to_csv('data/processed_data.csv', index=False)
	return df		

def shuffle_dataframe(dataframe, shuffle, seed, pctl):
	if shuffle:
		dataframe = dataframe.sample(frac=1, random_state=seed)
	#df.to_csv('data/randomized_data.csv', index=False)	
	train = dataframe.iloc[:int(dataframe.shape[0] * pctl), 1:]
	#train.to_csv('data/training.csv', index=False)	
	validation = dataframe.iloc[int(dataframe.shape[0] * pctl) : , 1:]
	#validation.to_csv('data/validation.csv', index=False)
	return train, validation
	


def extractData(dataframe):
	df = dataframe
	x = df.iloc[:, 1:].values
	y = df.iloc[:, 0].values
	return x, y


def pearson_r_square(data_pred, data_label):
	label_mean = np.mean(data_label)
	pred_mean = np.mean(data_pred)
	numerator = np.sum((data_pred - data_label) ** 2)
	denominator = np.sum((data_label - label_mean) ** 2)
	return 1 - numerator / denominator


def calculate_SSA(pred, label, thres, low, high, step):
	#label and pred are numpy arrays	
	data = np.column_stack((label, pred))
	ROC_data = []
	for pic50 in np.arange(low, high, step):
		TP = data[np.where((data[:, 0] >= thres) * (data[:, 1] >= pic50))].shape[0]		
		FP = data[np.where((data[:, 0] < thres) * (data[:, 1] >= pic50))].shape[0]
		TN = data[np.where((data[:, 0] < thres) * (data[:, 1] < pic50))].shape[0]
		FN = data[np.where((data[:, 0] >= thres) * (data[:, 1] < pic50))].shape[0]		
		ROC_data.append([pic50, TP / (TP + FN), FP / (TN + FP), (TP + TN) / pred.shape[0]])
	return np.array(ROC_data)

def apply_pca(train, validation, n_components=50):

	pca = PCA(n_components)
	train_x, train_y = extractData(train)
	val_x, val_y = extractData(validation)
	pca.fit(train_x)
	train_x = pca.transform(train_x)
	val_x = pca.transform(val_x)

	train = np.column_stack((train_y, train_x))	
	val = np.column_stack((val_y, val_x))

	train = pd.DataFrame(train)
	#train.to_csv('data/training.csv', index=False)
	val = pd.DataFrame(val)
	#val.to_csv('data/validation.csv', index=False)
	return train, val

def regression_model(regressor, train, validation, visual=True, **regressor_param):
	train_x, train_y = extractData(train)
	val_x, val_y = extractData(validation)

	rg = regressor(**regressor_param)
	rg.fit(train_x, train_y)
	pred_train = rg.predict(train_x)
	pred_val = rg.predict(val_x)	

	train_r2 = pearson_r_square(pred_train, train_y)
	val_r2 = pearson_r_square(pred_val, val_y)
	roc_data = calculate_SSA(pred_val, val_y, 5, 2, 10, 0.1)
	auc = np.trapz(roc_data[::-1, 1], roc_data[::-1, 2])

	#print(lr.coef_)
	
	#print("pearson r squared for training: " + str(train_r2))
	#print("pearson r squared for validation: " + str(val_r2))
	#print("AUC for validation: " + str(auc))

	if visual:	
		plot_corr(pred_train, train_y, pred_val, val_y,
		 train_r2, val_r2, auc, roc_data)
		#df_roc = pd.DataFrame(roc_data)
		#df_roc.to_csv("{}_ROC_Table.csv".format(regressor.__name__), index=False)

	return rg, train_r2, val_r2, auc

def fit(regressor, train, **regressor_param):
	train_x, train_y = extractData(train)
	rg = regressor(**regressor_param)
	rg.fit(train_x, train_y)
	return rg

def predict(model, test):
	test_x, test_y = extractData(test)
	pred = model.predict(test_x)
	return pred



def exp(regressor, shuffle=True, randomSeeds=range(0,100), 
		train_pctl=0.7, train_vis=False, save=True, **regressor_param):
	df = preprocess()
	train_r2s = []
	val_r2s = []
	aucs = []
	for seed in randomSeeds:
		train, validation = shuffle_dataframe(dataframe=df, shuffle=shuffle, 
											seed=seed, pctl=train_pctl)		
		model, train_r2, val_r2, auc = regression_model(regressor, 
														train, validation, 
														visual=train_vis, 
														**regressor_param)
		train_r2s.append(train_r2)
		val_r2s.append(val_r2)
		aucs.append(auc)	
	
	#print(train_r2s)
	train_r2_mean = np.mean(train_r2s)	
	print("train r squared mean: ")
	print(train_r2_mean)

	#print(val_r2s)
	val_r2_mean = np.mean(val_r2s)	
	print("validation r squared mean: ")
	print(val_r2_mean)

	#print(aucs)
	auc_mean = np.mean(aucs)	
	print("AUC mean: ")
	print(auc_mean)	

	if save:
		res = {"seeds": randomSeeds, 
			"train_r2" : train_r2s, 
			"val_r2" : val_r2s, 
			"auc" : aucs}
		df = pd.DataFrame(res, columns=["seeds", "train_r2", "val_r2", "auc"])
	
		#save to csv file
		df.loc["mean"] = df.mean()
		df.to_csv("exp_results_seeds.csv")

	return train_r2_mean, val_r2_mean, auc_mean

	
	
	
	



		
	
	
	




	 
		    

    

    





