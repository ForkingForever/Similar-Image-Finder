from skimage.metrics import structural_similarity as ssim
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--threshold", type=float, default=0.7,
                help="threshold")
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



def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    print("".join(['']))
    tres = args['threshold']

    if m > tres:
        print(m)
        twin = np.hstack([imageA, imageB])
        cv2.imshow('', twin)
        cv2.waitKey(0)

imagePaths = list(paths.list_images('example_images'))
companies = ['dhl', 'paypal', 'wellsfargo']


all_data = []

root = '.'
parent = 'dataset'

for path in imagePaths:
    for company in companies:
        if company in path:
            all_data.append({'comp': company, 'path': path})

print(all_data)
for image in all_data:
    try:
        p1 = cv2.imread(image['path'])
        p1 = cv2.resize(p1, (500, 500))
        p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
        for i in all_data:
            p2 = cv2.imread(image['path'])
            p2 = cv2.resize(p2, (500, 500))
            p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
            compare_images(p1, p2)
    except Exception as e:
        print(str(e))
