from model import preprocess, shuffle_dataframe, regression_model, apply_pca
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import pandas as pd

def exp(regressor, shuffle=True, randomSeeds=range(0,1), train_pctl=0.7, 
	train_vis=True, save=True, pca_n=50, **regressor_param):
	df = preprocess()
	train_r2s = []
	val_r2s = []
	aucs = []
	for seed in randomSeeds:
		train, validation = shuffle_dataframe(dataframe=df, 
											shuffle=shuffle, 
											seed=seed, 
											pctl=train_pctl)
		train, validation = apply_pca(train, validation, 
										n_components=pca_n)
		model, train_r2, val_r2, auc = regression_model(regressor, train, 
												validation,
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


if __name__ == "__main__":	
	train_r2 = []
	val_r2 = []
	aucs = []
	#pca_n = np.arange(1, 40) * 50
	#alphas = [5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 0.01, 0.011, 0.012, 0.013, 0.014, 
				# 0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
	pca_n = 200
	alpha = 0.01
	
	#for alpha in alphas:
		#print("alpha is: {}".format(alpha))
	train, val, auc = exp(Lasso, shuffle=True, 
								randomSeeds=range(0, 100),
								train_pctl=0.7, 
								train_vis=False, 
								save=True, 
								pca_n=pca_n,
								alpha=alpha)
	train_r2.append(train)
	val_r2.append(val)
	aucs.append(auc)
	
	# res = {"alpha" : alphas, 
			# "train" : train_r2, 
			# "validation" : val_r2,
			# "auc" : aucs}
	# df = pd.DataFrame(res, columns=["alpha", "train", "validation", "auc"])	
	# df.to_csv("exp_results_pca.csv")

	


		
	
	
	




	 
		    

    

    





