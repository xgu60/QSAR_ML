from model import preprocess, shuffle_dataframe, regression_model, exp, apply_pca
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import pandas as pd

def lr():
	exp(LinearRegression, shuffle=True, randomSeeds=range(0, 1), 
		train_pctl=0.7, train_vis=True, save=True)
		
def rf():
	exp(RandomForestRegressor, shuffle=True, 
		randomSeeds=range(0,100), train_pctl=0.7, 
		train_vis=False, save=True, n_estimators=100)

def gb():
	exp(GradientBoostingRegressor, shuffle=True, randomSeeds=range(0, 100),
		train_pctl=0.7, train_vis=False, save=True)
		
def knn():
	exp(KNeighborsRegressor, shuffle=True, randomSeeds=range(0, 1), 
		train_pctl=0.7, train_vis=True, save=False, n_neighbors=30, 
		weights="distance")
		
def ridge():
	exp(Ridge, shuffle=True, randomSeeds=range(6, 7), 
		train_pctl=0.7, train_vis=True, save=False, alpha=6)
		
def lasso():
	exp(Lasso, shuffle=True, randomSeeds=range(6, 7), 
		train_pctl=0.7, train_vis=True, save=False, alpha=0.015)
		

def ridge_train():
	train_r2 = []
	val_r2 = []
	aucs = []
	alphas = [1e-10, 1e-5, 0.01, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 128, 258, 512]
	for alpha in alphas:
		train, val, auc = exp(Ridge, shuffle=True, randomSeeds=range(0, 100), 
		train_pctl=0.7, train_vis=False, save=True, alpha=alpha)
		train_r2.append(train)
		val_r2.append(val)
		aucs.append(auc)
	
	res = {"alpha" : alphas, 
			"train" : train_r2, 
			"validation" : val_r2,
			"auc" : aucs}
	df = pd.DataFrame(res, columns=["alpha", "train", "validation", "auc"])	
	df.to_csv("exp_results_ridge.csv")
	
def lasso_train():
	train_r2 = []
	val_r2 = []
	aucs = []
	alphas = [1e-2, 1.25e-2, 1.5e-2, 1.75e-2, 2e-2, 2.25e-2, 2.5e-2, 2.75e-2, 3e-2,
				3.5e-2, 4.0e-2, 5.0e-2, 1e-1, 5e-1, 1.0, 5, 10, 20, 50]
	for alpha in alphas:
		train, val, auc = exp(Lasso, shuffle=True, randomSeeds=range(6,7), 
		train_pctl=0.7, train_vis=False, save=True, alpha=alpha)
		train_r2.append(train)
		val_r2.append(val)
		aucs.append(auc)
	
	res = {"alpha" : alphas, 
			"train" : train_r2, 
			"validation" : val_r2,
			"auc" : aucs}
	df = pd.DataFrame(res, columns=["alpha", "train", "validation", "auc"])	
	df.to_csv("exp_results_lasso.csv")



if __name__ == "__main__":	
	#lr()
	ridge()
	#knn()
	#rf()
	#gb()
	
	



		
	
	
	




	 
		    

    

    





