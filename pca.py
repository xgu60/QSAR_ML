from model import preprocess, shuffle_dataframe, apply_pca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
	df = preprocess()
	train, val = shuffle_dataframe(dataframe=df, shuffle=True, 
										seed=2, pctl=0.7)
	train, val = apply_pca(train, val, n_components=2)
	#print(train.head(), validation.head())
	train_pos = train[train.iloc[:, 0] >= 5]
	train_neg = train[train.iloc[:, 0] < 5]

	val_pos = val[val.iloc[:, 0] >= 5]
	val_neg = val[val.iloc[:, 0] < 5]
	#print(train_pos.head(), val_neg.head())

	plt.subplot(1, 2, 1)
	plt.xlabel('pca_component_1', fontsize=12)
	plt.ylabel('pca_component_2', fontsize=12)
	#plt.ylim((0,1))
	plt.scatter(train_pos.iloc[:, 1], train_pos.iloc[:, 2], 
				c='b', label="potent")
	plt.scatter(train_neg.iloc[:, 1], train_neg.iloc[:, 2], 
				c='r', label="not potent", alpha=0.5)	
	plt.legend(loc='upper right', shadow=True)	
	plt.subplot(1, 2, 2)
	plt.xlabel('pca_component_1', fontsize=12)
	plt.ylabel('pca_component_2', fontsize=12)
	#plt.ylim((0,1))
	plt.scatter(val_pos.iloc[:, 1], val_pos.iloc[:, 2], 
				c='b', label="potent")
	plt.scatter(val_neg.iloc[:, 1], val_neg.iloc[:, 2], 
				c='r', label="not potent", alpha=0.5)
	plt.legend(loc='upper right', shadow=True)
	plt.tight_layout()	
	plt.show()