# author: Alberto del Rio Ponce
# date: March 2019

# USAGE
# python index.py --dataset images --index index_descriptor.cpickle --descriptor descriptor

# import the necessary packages
from pyimagesearch.rgbhistogram import RGBHistogram
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
import numpy as np
import argparse
import pickle
import glob
import cv2
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required=True,
	help="Name of index")
ap.add_argument("-e", "--descriptor", required=True,
	help="Name of descriptor", choices=["rgb", "lbp", "hog"])
args = vars(ap.parse_args())

# initialize the index dictionary to store our our quantifed
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
index = {}

# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
option = args["descriptor"]
if option == "rgb":
	print("RGB Descriptor")
	desc = RGBHistogram([8, 8, 8])
	# size = len(glob.glob(args["dataset"] + os.sep + "*.png"))

	# use glob to grab the image paths and loop over them
	for imagePath in glob.glob(args["dataset"] + os.sep + "*.png"):
		# load the image, describe it using our RGB histogram
		# descriptor, and update the index
		image = cv2.imread(imagePath)
		# print(os.path.basename(k))

		# Print each image to get the working status
		print(imagePath)
		print("-------------")
		# extract our unique image ID (i.e. the filename)
		k = imagePath[imagePath.rfind(os.sep) + 1:]
		# print(k)

		features = desc.describe(image)
		index[k] = features

elif option == "lbp":
	print("LBP Descriptor")
	# initialize the local binary patterns descriptor along with
	# the data and label lists
	desc = LocalBinaryPatterns(24, 8)
	# data = []
	# labels = []
	# size = len(glob.glob(args["dataset"] + os.sep + "*.png"))

	# Training
	# loop over the training images
	for imagePath in glob.glob(args["dataset"] + os.sep + "*.png"):
		# load the image, convert it to gray scale, and describe it
		image = cv2.imread(imagePath)
		print(imagePath)
		print("-------------")
		# extract our unique image ID (i.e. the filename)
		k = imagePath[imagePath.rfind(os.sep) + 1:]
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		features = desc.describe(gray)
		# print(features)
		index[k] = features

elif option == "hog":
	print("HOG Descriptor")
	# initialize the histogram of oriented gradients descriptor
	winSize = (64, 64);	blockSize = (16, 16); blockStride = (8, 8);	cellSize = (8, 8)
	nbins = 9; derivAperture = 1; winSigma = 4.; histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01; gammaCorrection = 0; nlevels = 64
	winStride = (8, 8);	padding = (8, 8); locations = ((10, 20),)
	hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
    	histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
	# size = len(glob.glob(args["dataset"] + os.sep + "*.png"))

	# Training
	# loop over the training images
	for imagePath in glob.glob(args["dataset"] + os.sep + "*.png"):
		# load the image
		image = cv2.imread(imagePath)
		# Print each image to get the working status
		print(imagePath)
		print("-------------")

		# extract our unique image ID (i.e. the filename)
		k = imagePath[imagePath.rfind(os.sep) + 1:]

		# Resize to avoid run out of memory and slowness
		img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
		hist = hog.compute(img, winStride, padding, locations)
		#(hist, _) = np.histogram(desc.ravel())
		eps = 1e-7
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		index[k] = hist

# we are now done indexing our image -- now we can write our
# index to disk
# print(len(index))
outputFile = open(args["index"]+".pkl", "wb")
pickle.dump(index, outputFile)
outputFile.close()

pickle.dumps(index)
# show how many images we indexed
print("done...indexed %d images" % (len(index)))
