from model import preprocess, fit, predict, apply_pca, pearson_r_square, calculate_SSA
from plot import plot_corr2
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import pandas as pd

def exp(regressor, pca_n_components=None, **regressor_param):
	df = preprocess()
	df = df.iloc[:, 1:]
	#print(df.head())
	sample_num = df.shape[0]
	val_pred = []
	
	for row in range(sample_num):
		val = df.iloc[[row]]
		train = df.drop([row])
		if pca_n_components != None:
			train, val = apply_pca(train, val, n_components=pca_n_components)
		model = fit(regressor, train, **regressor_param)
		pred = predict(model, val)
		val_pred.append(pred[0])
			
	real_label = df.iloc[:, 0].values
	val_pred = np.array(val_pred)
	r2 = pearson_r_square(val_pred, real_label)
	print("pearson r square: {}".format(r2))
	roc_data = calculate_SSA(val_pred, real_label, 5, 2, 10, 0.1)
	auc = np.trapz(roc_data[::-1, 1], roc_data[::-1, 2])
	print("AUC: {}".format(auc))
	plot_corr2(val_pred, real_label, r2, auc, roc_data)
	
	



if __name__ == "__main__":	
	exp(LinearRegression, 100)
	
	
	




	 
		    

    

    





