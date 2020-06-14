from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    


companies = ['dhl', 'paypal', 'wellsfargo']
file_types = ['.jpg', '.png', '.bla']


image_paths = []

root = '.'
parent = 'example_images'

for path, subdirs, files in os.walk(os.path.join(root, parent)):
    for name in files:
        if name.endswith(tuple(file_types)):
            path = os.path.join(path, name)
            for company in companies:
                if company in path:
                    image_paths.append({'comp': company, 'path': path})


for image in image_paths:
    p1 = cv2.imread(image['path'])
    p1 = cv2.resize(p1, (300, 300))
    for i in image.image_paths:
        p2 = cv2.imread(image['path'])
        p2 = cv2.resize(p2, (300, 300))