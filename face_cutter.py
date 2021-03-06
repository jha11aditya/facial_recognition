from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import sys
import cv2
import os
import numpy as np
import ntpath



CASCADE_PATH = "fr_env/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml"

def display_image(img):
    plt.imshow(img)
    plt.show()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)







if len(sys.argv) < 2:
    print("no input image")
    exit(1)
if len(sys.argv) < 3:
    print("no output folder")
    exit(1)
op_path = sys.argv[2]
print(sys.argv[1])
raw_img = cv2.imread(sys.argv[1])
print(raw_img.shape)
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
print(img.shape)

# resized_img = resize(img,(1000,1000))
resized_img =img
print(resized_img.shape)
# display_image(resized_img)


face_classifier = cv2.CascadeClassifier(CASCADE_PATH)

# resized_img = np.array(resized_img, dtype='uint8')
faces = face_classifier.detectMultiScale(resized_img, 1.3,3) ### (inp_img, scaling_factor, minNeighbours )
print(len(faces))
if len(faces) == 0:
    print("No faces found")
ind = 0
for (x,y,w,h) in faces:
    cv2.rectangle(resized_img, (x,y), (x+w,y+h), (127,0,255), 2 )
    cropped = resized_img[y:y+h, x:x+w]
    # cv2.imshow('Face', cropped)
    cv2.imwrite( op_path + "/faceof_"+str(ind)+path_leaf(sys.argv[1]),cropped)
    # print("***", op_path, op_path + "/faceof_"+str(ind)+path_leaf(sys.argv[1]))
    ind+=1
    # cv2.waitKey(0)

cv2.destroyAllWindows()