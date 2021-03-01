from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


ap = argparse.ArgumentParser()

ap.add_argument("-i","--dataset",required=True, help="path to input data images folder")
ap.add_argument("-e","--encodings",required=True, help="path to pickle file of face embeddings to be created")
# ap.add_argument("-d","--detec",required=True, help="path to input data images")

args = vars(ap.parse_args())
