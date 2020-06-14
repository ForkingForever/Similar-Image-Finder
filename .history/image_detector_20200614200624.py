from skimage.metrics import structural_similarity as ssim
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--threshold", type=float, default=0.9,
                help="threshold")
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(path1,path2,imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    tres = args['threshold']
  

    if s >= tres:
        print("Image ''".format([path1,path2,str(m), str(tres), str(s)]))
        twin = np.hstack([imageA, imageB])
        cv2.imshow('', twin)
        cv2.waitKey(0)


imagePaths = list(paths.list_images(args['dataset']))
companies = ['dhl', 'paypal', 'wellsfargo']
all_data = []


for path in imagePaths:
    company = ''
    for c in companies:
        if c in path:
            company = c
    all_data.append({'comp': c, 'path': path})


for image in all_data:
    try:
        p1 = cv2.imread(image['path'])
        p1 = cv2.resize(p1, (300, 300))
        p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
        for i in all_data:
            if i['path']!=image['path']:
                p2 = cv2.imread(i['path'])
                p2 = cv2.resize(p2, (300, 300))
                p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
                compare_images(image['path'],i['path'],p1, p2)
    except Exception as e:

        print(str(e))
