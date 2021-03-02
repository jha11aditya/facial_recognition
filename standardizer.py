import os
import sys
import cv2
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
class dataset:
    def __init__(self, da, ta ):
        self.point = da
        self.target = ta

if len(sys.argv) < 2:
    print("no input data folder")
    exit(0)

rdat = sys.argv[1]
print(rdat, [name for name in os.listdir(rdat) if os.path.isdir(os.path.join(rdat, name))])

subj_names = [name for name in os.listdir(rdat) if os.path.isdir(os.path.join(rdat, name)) ]

dataset_list = []
for d in subj_names:
    inpath = rdat + "/" + d
    img_files = [name for name in os.listdir(inpath) if  not os.path.isdir(os.path.join(inpath, name)) ]
    # print(img_files)
    for img in img_files:
        inimg = inpath + "/" + img
        # print(inimg)
        imgdat = imread(inimg, as_gray=True)
        print(imgdat[23][1])
        imgdat = resize(imgdat, (64,64))
        # print(imgdat[0])
        imgdat = img_as_ubyte(imgdat)
        # print(imgdat[0])
        # imsave(  "test1.png", img_as_ubyte(imgdat))
        
        # print(imgdat.shape)

        flat_imgdat = np.array( imgdat ).flatten()
        # print(flat_imgdat.shape)
        # print(flat_imgdat)
        dataset_list.append( dataset(flat_imgdat, d) )
        # break
    # break

for ele in dataset_list:
    print(ele.point, ele.target)
    
f = open("full_data.csv", "w")
for i in range(len(dataset_list[0].point)):
    f.write("col"+str(i)+",")
f.write("target\n")
for ele in dataset_list:
    # print(ele.point, ele.target)
    for e in ele.point:
        f.write(str(e)+",")
    f.write(ele.target + "\n")