import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pickle

if len(sys.argv) < 2:
    print("no input csv file")
    exit(0)



df = pd.read_csv(sys.argv[1])

# print(df, df.columns[0:len(df.columns)])

xcols = [ i for i in df.columns]
targ = xcols.pop()
# print(xcols)
X = df.loc[:,xcols ].values
print(X.shape)
Y = df.loc[:,targ].values
print(Y.shape)

X = StandardScaler().fit_transform(X)
print(X)
# pca = PCA(n_components=128)
# pcaofX = pca.fit_transform(X)
# print("shapeofX after pca",pcaofX.shape, ", cum Sum of variance ratio",pca.explained_variance_ratio_.cumsum()[-1])

pcaofX = X

X_train, X_test, Y_train, Y_test = train_test_split(pcaofX, Y, test_size=0.1,random_state=109)

print(X_train.shape)

svm_model = svm.SVC(kernel="linear")
svm_model.fit(X_train,Y_train)

pred = svm_model.predict(X_test)
print(pred)

print("Accuracy:",metrics.accuracy_score(Y_test, pred))


pickle.dump(svm_model, open("svm_model.sav", 'wb'))

