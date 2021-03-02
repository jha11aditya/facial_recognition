import os
import sys
import pandas as pd
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pickle
from skimage import img_as_ubyte

tempdir = "predictor_temp"

if len(sys.argv) < 3:
    print("no input image")
    exit(0)

if len(sys.argv) < 2:
    print("no input model")
    exit(0)

svm_model = pickle.load(open(sys.argv[1], 'rb'))

os.system('mkdir -p ' + tempdir)
os.system("python3 face_cutter.py " + sys.argv[2] + " " + tempdir )
img_files = [name for name in os.listdir(tempdir) if  not os.path.isdir(os.path.join(tempdir, name)) ]
print(img_files)
cutf = tempdir +"/"+img_files[0]
os.system("python3 hogger.py " + cutf + " " + tempdir )
os.system("rm " + cutf)
img_files = [name for name in os.listdir(tempdir) if  not os.path.isdir(os.path.join(tempdir, name)) ]
hogf = tempdir + "/" + img_files[0]

imgdat = imread(hogf, as_gray=True)
os.system("rm -rf " + tempdir)
imgdat = resize(imgdat, (64,64))

imgdat = img_as_ubyte(imgdat)


flat_imgdat = np.array( imgdat ).flatten()
print(flat_imgdat.shape)
X = np.array(flat_imgdat)
X = X.reshape(-1,1)
X = StandardScaler().fit_transform(X)
print(X.shape,X)
# pca = PCA(n_components=128)
# pcaofX = pca.fit_transform(X)


res = svm_model.predict(X.T)

print("Prediction:",res)

