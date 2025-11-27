
# import the required libraries
import sweetviz as sv
from sklearn.model_selection import train_test_split

print("SweetViz Version : {}".format(sv.__version__))

import os
import joblib
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('investigator_ftldlbd_nacc71.csv', low_memory=False)

df.head(5)

# analyzing the dataset
# report = sv.analyze([df, 'Train'], pairwise_analysis='off')


# report.show_html('uds_report.html', open_browser=False)

#!pip install ydata-profiling
import pandas as pd
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Your Data Profile Report")
profile

from sklearn.decomposition import PCA
#import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def mean_centering(df):
    for variable in df.columns:
        df[variable] = df[variable] - np.mean(df[variable])


def replace(df):
    for column in df:
        if df[column].isna().any():
            mean = np.mean(df[column])
            df[column] = df[column].replace(np.nan, np.mean(df[column]))


def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = ys)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


df = df.select_dtypes(include=['float64', 'int64'])
#print(df.head)
#replace(df)
#df = df.dropna()
#print(df.head)
mean_centering(df)
df = df.fillna(0)

print(df.head)

matrix = df.to_numpy()





pca = PCA().fit(matrix)
pca_matrix = pca.transform(matrix)
eigenvalues = pca.explained_variance_

plt.figure(figsize=(8, 6))
plt.scatter(pca_matrix[:, 0], pca_matrix[:, 1])
plt.xlabel(f"Principal Component 1 ({round(eigenvalues[0]/sum(eigenvalues), 3)*100})")
plt.ylabel(f"Principal Component 2 ({round(eigenvalues[1]/sum(eigenvalues), 3) *100})")
plt.title("Scores Plot (PCA)")
plt.grid(True)

plt.savefig("pca_na_removed_CSF_scores.png")
#plt.savefig("pca_na_replaced_with_mean_CSF.png")


#Call the function. Use only the 2 PCs.
myplot(pca_matrix[:,0:2],np.transpose(pca.components_[0:2, :]), df.columns)
plt.savefig("pca_na_removed_CSF_loadings.png")
#plt.savefig("pca_na_replaced_with_mean_CSF_loadings.png")

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = ys)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

#Call the function. Use only the 2 PCs.
myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()
