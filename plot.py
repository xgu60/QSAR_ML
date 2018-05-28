import pandas as pd 
import matplotlib.pyplot as plt

def plot_corr(pred_train, train_y, pred_val, val_y, train_r2, val_r2, auc, roc_data):
	plt.subplot(221)
	#plot training and validation
	plt.ylabel('Real_pIC50', fontsize=16)
	plt.xlabel('Predict_pIC50', fontsize=16)		
	#plt.grid(True)	
	plt.scatter(pred_train, train_y, c='r', alpha=0.5, label='training')
	plt.scatter(pred_val, val_y, c='b', label='validation')
	plt.legend(loc='lower right', shadow=True)
	plt.text(2.5, 8.0, "Training $R^2$ = %4.2f"%train_r2, size=14, color="red")
	plt.text(2.5, 7.5, "Validation $R^2$ = %4.2f"%val_r2, size=14, color="blue")		
	#plt.show()
	plt.subplot(222)
	#plot validation alone
	plt.ylabel('Real_pIC50', fontsize=16)
	plt.xlabel('Predict_pIC50', fontsize=16)		
	#plt.grid(True)	
	plt.scatter(pred_val, val_y, c='b', label='validation')
	plt.legend(loc='lower right', shadow=True)
	plt.text(2.5, 7.0, "Validation $R^2$ = %4.2f"%val_r2, size=16, color="blue")		
	#plt.show()
	plt.subplot(223)
	#plot roc and auc for validation
	plt.xlabel('FPR (1 - specificity)', fontsize=20)
	plt.ylabel('TPR (sensitivity)', fontsize=20)
	#plt.grid(True)	
	plt.plot(roc_data[:, 2], roc_data[:, 1], 'g-')
	plt.plot([0, 1], [0, 1], 'r-')
	plt.text(0.6, 0.1, "AUC = %4.2f"%auc, size=20)
	plt.tight_layout()
	plt.show()
	
def plot_corr2(pred_val, val_y, val_r2, auc, roc_data):
	plt.subplot(121)
	#plot validation alone
	plt.ylabel('Real_pIC50', fontsize=16)
	plt.xlabel('Predict_pIC50', fontsize=16)		
	#plt.grid(True)	
	plt.scatter(pred_val, val_y, c='b', label='validation', alpha=0.5)
	#plt.legend(loc='lower right', shadow=True)
	plt.text(2, 8.5, "Leave_one_out $R^2$ = %4.2f"%val_r2, size=12, color="blue")		
	#plt.show()
	plt.subplot(122)
	#plot roc and auc for validation
	plt.xlabel('FPR (1 - specificity)', fontsize=20)
	plt.ylabel('TPR (sensitivity)', fontsize=20)
	#plt.grid(True)	
	plt.plot(roc_data[:, 2], roc_data[:, 1], 'g-')
	plt.plot([0, 1], [0, 1], 'r-')
	plt.text(0.6, 0.1, "AUC = %4.2f"%auc, size=20)
	plt.tight_layout()
	plt.show()


def lr_pca():
	df = pd.read_csv("exp_results_pca.csv")
	df = df[:20]
	plt.xlabel('pca_n_components', fontsize=20)
	plt.ylabel('results', fontsize=20)
	plt.ylim((0,1.1))
	plt.scatter(df.iloc[:, 1], df.iloc[:, 2], c='r', alpha=0.8)
	plt.scatter(df.iloc[:, 1], df.iloc[:, 3], c='b', alpha=0.8)
	plt.scatter(df.iloc[:, 1], df.iloc[:, 4], c='g', alpha=0.8)
	plt.plot(df.iloc[:, 1], df.iloc[:, 2], c='r')
	plt.plot(df.iloc[:, 1], df.iloc[:, 3], c='b')
	plt.plot(df.iloc[:, 1], df.iloc[:, 4], c='g')
	plt.text(150, 1.02, "training $R^2$", size=12, color="r")
	plt.text(50, 0.35, "validation $R^2$", size=12, color="b")
	plt.text(50, 0.87, "AUC", size=12, color="g")
	plt.show()

def seeds_mean():
	df = pd.read_csv("exp_results_seeds.csv")
	plt.xlabel('experiments', fontsize=20)
	plt.ylabel('results', fontsize=20)
	plt.ylim((-0.1,1.2))
	plt.plot(df.iloc[:, 2], c='r', label="training $R^2$", alpha=0.8)
	plt.plot(df.iloc[:, 3], c='b', label="validation $R^2$", alpha=0.8)
	plt.plot(df.iloc[:, 4], c='g', label="AUC", alpha=0.8)
	plt.text(1, 1.1, "training $R^2$ mean = %4.2f"%df.iloc[-1, 2], size=12, color="r")
	plt.text(1, 0.5, "validation $R^2$ mean = %.2g"%df.iloc[-1, 3], size=12, color="b")
	plt.text(1, 1.0, "AUC mean = %4.2f"%df.iloc[-1, 4], size=12, color="g")
	plt.show()

def ridge_alpha():
	df = pd.read_csv("exp_results_ridge.csv")
	df = df[:]
	plt.xlabel('alpha', fontsize=20)
	plt.ylabel('results', fontsize=20)
	plt.ylim((0,1.1))
	plt.xscale("log")
	plt.scatter(df.iloc[:, 1], df.iloc[:, 2], c='r', alpha=0.8)
	plt.scatter(df.iloc[:, 1], df.iloc[:, 3], c='b', alpha=0.8)
	plt.scatter(df.iloc[:, 1], df.iloc[:, 4], c='g', alpha=0.8)
	plt.plot(df.iloc[:, 1], df.iloc[:, 2], c='r')
	plt.plot(df.iloc[:, 1], df.iloc[:, 3], c='b')
	plt.plot(df.iloc[:, 1], df.iloc[:, 4], c='g')
	plt.text(1E-9, 1.05, "training $R^2$", size=12, color="r")
	plt.text(1E-9, 0.40, "validation $R^2$", size=12, color="b")
	plt.text(1E-9, 0.87, "AUC", size=12, color="g")
	plt.show()

def lasso_alpha():
	df = pd.read_csv("exp_results_lasso.csv")
	df = df[:]
	plt.xlabel('lasso_alpha', fontsize=20)
	plt.ylabel('results', fontsize=20)
	plt.ylim((-0.1,1.1))
	plt.xscale("log")
	plt.scatter(df.iloc[:, 1], df.iloc[:, 2], c='r', alpha=0.8)
	plt.scatter(df.iloc[:, 1], df.iloc[:, 3], c='b', alpha=0.8)
	plt.scatter(df.iloc[:, 1], df.iloc[:, 4], c='g', alpha=0.8)
	plt.plot(df.iloc[:, 1], df.iloc[:, 2], c='r')
	plt.plot(df.iloc[:, 1], df.iloc[:, 3], c='b')
	plt.plot(df.iloc[:, 1], df.iloc[:, 4], c='g')
	plt.text(1, 0.3, "training $R^2$", size=12, color="r")
	plt.text(1, 0.2, "validation $R^2$", size=12, color="b")
	plt.text(1E-2, 0.9, "AUC", size=12, color="g")
	plt.show()

def roc():
	df1 = pd.read_csv("lr_roc.csv")
	df2 = pd.read_csv("ridge_roc.csv")
	df3 = pd.read_csv("lasso_roc.csv")

	#plot roc and auc for validation
	plt.xlabel('FPR (1 - specificity)', fontsize=20)
	plt.ylabel('TPR (sensitivity)', fontsize=20)
	#plt.grid(True)	
	plt.plot(df1.iloc[:, 2], df1.iloc[:, 1], 'm-', label="pca")
	plt.plot(df2.iloc[:, 2], df2.iloc[:, 1], 'b-', label="ridge")
	plt.plot(df3.iloc[:, 2], df3.iloc[:, 1], 'g--', label="lasso")
	plt.plot([0, 1], [0, 1], 'r-')	
	plt.legend(loc='lower right', shadow=True)
	plt.show()	
	

if __name__ == "__main__":
	#lr_pca()
	#ridge_alpha()
	seeds_mean()
	#lasso_alpha()
	#roc()
	

	