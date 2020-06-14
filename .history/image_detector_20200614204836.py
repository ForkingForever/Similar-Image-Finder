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


class Utility:
    totalFound = 0
    totalCompare = 0
    searching = False
    companyResults = {}

    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
        return err

    def compare_images(self, im1, im2, imageA, imageB):
        # compute the mean squared error and structural similarity
        # index for the images
        m = self.mse(imageA, imageB)
        s = ssim(imageA, imageB)
        tres = args['threshold']
        self.totalCompare += 1
        if s >= tres:
            print("Image[{c1}] '{p1}' compared to Image[{c2}] '{p2}' Simility:{sim}".format(c1=im1['comp'], c2=im2['comp'],p1=im1['path'], p2=im2['path'], sim=str(s)))
            twin = np.hstack([imageA, imageB])
            cv2.imshow('', twin)
            cv2.waitKey(0)
            self.searching = False
            self.totalFound += 1
            companyTotal = self.companyResults.get(im1['comp'],[])
            companyTotal+=1
            self.companyResults[im1['comp']] = companyTotal
        elif self.searching is False:
            print('Searching...')
            self.searching = True


imagePaths = list(paths.list_images(args['dataset']))
companies = ['dhl', 'paypal', 'wellsfargo']
all_data = []

for path in imagePaths:
    company = ''
    for c in companies:
        if c in path:
            company = c
    all_data.append({'comp': company, 'path': path})

u = Utility()

for image in all_data:
    try:
        p1 = cv2.imread(image['path'])
        p1 = cv2.resize(p1, (300, 300))
        p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
        for i in all_data:
            if i['path'] != image['path']:
                p2 = cv2.imread(i['path'])
                p2 = cv2.resize(p2, (300, 300))
                p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)

                u.compare_images(image, i, p1, p2)
    except Exception as e:
        print(str(e))

print("{total} times compared. {found} similarity found by over {tr} treshold.".format(total=u.totalCompare,found=u.totalFound,tr=args['threshold']))


for key,value in u.companyResults:
    print("Company:{c} Total:{")