from skimage.metrics import structural_similarity as ssim
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

    tres = args['threshold']

    if m > tres:
        print(m)
        twin = np.hstack([imageA, imageB])
        cv2.imshow('', twin)
        cv2.waitKey(0)


companies = ['dhl', 'paypal', 'wellsfargo']
file_types = ['.jpg','.jpeg', '.png', '.bla']


image_paths = []

root = '.'
parent = 'example_images'

for path, subdirs, files in os.walk(os.path.join(root, parent)):
    for name in files:
        if name.endswith(tuple(file_types)):
            path = os.path.join(path, name)
            for company in companies:
                if company in path:
                    image_paths.append({'comp': company, 'path': path.replace('.jpg','.jpeg')})
cv2.imread('example_images/dhl/dhl-1.jpg')
print(image_paths)
try:
    for image in image_paths:
        try:
            p1 = cv2.imread(image['path'])
        p1 = cv2.resize(p1, (300, 400))
        p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
        for i in image_paths:
            p2 = cv2.imread(image['path'])
            p2 = cv2.resize(p2, (300, 400))
            p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
            compare_images(p1, p2)
except Exception as e:
    print(str(e))
