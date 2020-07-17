# author: Alberto del Rio
# date: March 2019

# USAGE
# python search.py --dataset images --index index_descriptor.cpickle --descriptor descriptor

# import the necessary packages
from pyimagesearch.searcher import Searcher
import numpy as np
import pandas as pd
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required=False,
	help="Name of index")
ap.add_argument("-a", "--all", required=False,
	help="Search with all descriptors")
args = vars(ap.parse_args())

# Configure the search: 1 for combining descriptors, 0 using the specified descriptor by CLI
if args['all']:
	search = 1
else:
	search = 0

if search == 0:
	# load the index and initialize our searcher
	file = open(args["index"]+".pkl", 'rb')
	index = pickle.load(file)
	searcher = Searcher(index)

	# loop over images in the index -- we will use each one as
	# a query image
	for (query, queryFeatures) in index.items():
		# perform the search using the current query
		# Very slow for hog searching due to histogram length
		results = searcher.search(queryFeatures)

		# load the query image and display it
		path = args["dataset"] + os.sep + query
		print(path)
		queryImage = cv2.imread(path)
		queryImage = cv2.resize(queryImage, (450, 360))
		cv2.putText(queryImage, query, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					1.0, (0, 0, 255), 3)
		cv2.imshow("Query", queryImage)
		print("query: %s" % query)

		# initialize the two montages to display our results --
		# we have a total of 25 images in the index, but let's only
		# display the top 10 results; 5 images per montage, with
		# images that are 400x166 pixels
		montageA = np.zeros((166 * 5, 400, 3), dtype="uint8")
		montageB = np.zeros((166 * 5, 400, 3), dtype="uint8")

		# loop over the top ten results
		for j in range(0, 10):
			# grab the result (we are using row-major order) and
			# load the result image
			(score, imageName) = results[j]
			path = args["dataset"] + os.sep + "%s" % imageName
			result = cv2.resize(cv2.imread(path), (400, 166))
			cv2.putText(result, imageName + ': ' + str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
						1.0, (0, 0, 255), 3)
			#cv2.putText(result, score, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			#			1.0, (0, 0, 255), 3)
			print("\t%d. %s : %.3f" % (j + 1, imageName, score))

			# check to see if the first montage should be used
			if j < 5:
				montageA[j * 166:(j + 1) * 166, :] = result

			# otherwise, the second montage should be used
			else:
				montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result

		# show the results
		cv2.imshow("Results 1-5", montageA)
		cv2.imshow("Results 6-10", montageB)
		cv2.waitKey(0)

elif search == 1:

	# load the index and initialize the searchers
	file_rgb = open("index_rgb.cpickle" + ".pkl", 'rb')
	index_rgb = pickle.load(file_rgb)
	searcher_rgb = Searcher(index_rgb)

	file_lbp = open("index_lbp.cpickle" + ".pkl", 'rb')
	index_lbp = pickle.load(file_lbp)
	searcher_lbp = Searcher(index_lbp)

	file_hog = open("index_hog.cpickle" + ".pkl", 'rb')
	index_hog = pickle.load(file_hog)
	searcher_hog = Searcher(index_hog)

	for (query_rgb, queryFeatures_rgb) in index_rgb.items():
		results_rgb = searcher_rgb.search(queryFeatures_rgb)
		for (query_lbp, queryFeatures_lbp) in index_lbp.items():
			if query_rgb == query_lbp:
				results_lbp = searcher_lbp.search(queryFeatures_lbp)
				for (query_hog, queryFeatures_hog) in index_hog.items():
					if query_hog == query_lbp and query_hog == query_rgb:
						results_hog = searcher_hog.search(queryFeatures_hog)
						# break
			# break
		rgb_df = pd.DataFrame(results_rgb, columns=['RGB', 'Name'])
		rgb_df.set_index('Name', inplace=True)

		lbp_df = pd.DataFrame(results_lbp, columns=['LBP', 'Name'])
		lbp_df.set_index('Name', inplace=True)

		hog_df = pd.DataFrame(results_hog, columns=['HOG', 'Name'])
		hog_df.set_index('Name', inplace=True)

		result = pd.concat([rgb_df, lbp_df, hog_df], axis=1, sort=True)

		# Normalize values
		for feature_name in result.columns:
			max_value = result[feature_name].max()
			min_value = result[feature_name].min()
			result[feature_name] = (result[feature_name] - min_value) / (max_value - min_value)

		# Extract the average between descriptors and sort the dataframe
		result['Average'] = result.mean(numeric_only=True, axis=1)
		final = result.sort_values(by=['Average'])
		#print(final.head())

		# load the query image and display it
		path = args["dataset"] + os.sep + query_rgb
		#print(path)
		# cv2.waitKey(0)
		queryImage = cv2.imread(path)
		queryImage = cv2.resize(queryImage, (450, 360))
		cv2.putText(queryImage, query_rgb, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					1.0, (0, 0, 255), 3)
		cv2.imshow("Query", queryImage)
		print("query: %s" % query_rgb)

		# initialize the two montages to display our results --
		# we have a total of 25 images in the index, but let's only
		# display the top 10 results; 5 images per montage, with
		# images that are 400x166 pixels
		montageA = np.zeros((166 * 5, 400, 3), dtype="uint8")
		montageB = np.zeros((166 * 5, 400, 3), dtype="uint8")

		# loop over the top ten results
		for j in range(0, 10):
			# grab the result (we are using row-major order) and
			# load the result image
			score = final['Average'][j]
			imageName = final.index[j]
			#(score, imageName) = results[j]
			path = args["dataset"] + os.sep + "%s" % imageName
			imgresult = cv2.resize(cv2.imread(path), (400, 166))
			cv2.putText(imgresult, imageName + ': ' + str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
						1.0, (0, 0, 255), 3)
			# cv2.putText(result, score, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			#			1.0, (0, 0, 255), 3)
			print("\t%d. %s : %.3f" % (j + 1, imageName, score))

			# check to see if the first montage should be used
			if j < 5:
				montageA[j * 166:(j + 1) * 166, :] = imgresult

			# otherwise, the second montage should be used
			else:
				montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = imgresult

		# show the results
		cv2.imshow("Results 1-5", montageA)
		cv2.imshow("Results 6-10", montageB)
		cv2.waitKey(0)