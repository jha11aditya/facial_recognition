from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import sys


def display_image(img):
    plt.imshow(img)
    plt.show()

if len(sys.argv) < 2:
    print("no input image")
    exit(1)


img = imread(sys.argv[1], as_gray=True)
print(img.shape)

##resizing to 1:2 aspect ratio for easier calc

resized_img = resize(img,(1024,512))
print(resized_img.shape)
display_image(resized_img)

fd, hog_img = hog(resized_img,
 orientations=9,
 pixels_per_cell=(4,4),
 cells_per_block=(2,2),
 visualize=True,
 multichannel=False )

display_image(hog_img)
