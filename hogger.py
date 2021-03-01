from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np

def display_image(img):
    plt.imshow(img)
    plt.show()








if len(sys.argv) < 2:
    print("no input image")
    exit(1)

img = imread(sys.argv[1], as_gray=True)
print(img.shape)

##resizing to 1:2 aspect ratio for easier calc

# resized_img = resize(img,(1024,512))
resized_img = img
print(resized_img.shape)
# display_image(resized_img)

# calc hog
fd, hog_img = hog(resized_img,
 orientations=8,
 pixels_per_cell=(2,2),
 cells_per_block=(1,1),
 visualize=True,
 multichannel=False )

# display_image(hog_img)
hog_img = np.array(hog_img, dtype='uint8')
imsave("hog_of" + sys.argv[1], hog_img)
