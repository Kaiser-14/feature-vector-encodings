# author: Alberto del Rio
# date: March 2019

# USAGE python search_external.py --dataset images --index index.cpickle --query queries/rivendell-query.png
# --descriptor descriptor

# import the necessary packages
from pyimagesearch.rgbhistogram import RGBHistogram
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from pyimagesearch.searcher import Searcher
import numpy as np
import pandas as pd
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help="Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required=False,
	help="Name of index")
ap.add_argument("-q", "--query", required=True,
	help="Path to query image")
ap.add_argument("-e", "--descriptor", required=False,
	help="Name of descriptor", choices=["rgb", "lbp", "hog"])
ap.add_argument("-a", "--all", required=False,
	help="Search with all descriptors")
args = vars(ap.parse_args())

# Configure the search: 1 for combining descriptors, 0 using the specified descriptor by CLI
if args['all']:
    search = 1
else:
	search = 0

directory = args["query"]
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        queryPath = os.path.join(directory, filename)
        print(queryPath)
        queryImage = cv2.imread(queryPath)
        queryImage = cv2.resize(queryImage, (450, 360))
        cv2.putText(queryImage, queryPath, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        cv2.imshow("Query", queryImage)
        print("query: %s" % queryPath)
        if search == 0:
            if args["descriptor"] == "rgb":
                desc = RGBHistogram([8, 8, 8])
                queryFeatures = desc.describe(queryImage)
            elif args["descriptor"] == "lbp":
                desc = LocalBinaryPatterns(24, 8)
                gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
                queryFeatures = desc.describe(gray)
            elif args["descriptor"] == "hog":
                winSize = (64, 64); blockSize = (16, 16); blockStride = (8, 8);
                cellSize = (8, 8); nbins = 9; derivAperture = 1; winSigma = 4.;
                histogramNormType = 0; L2HysThreshold = 2.0000000000000001e-01;
                gammaCorrection = 0; nlevels = 64; winStride = (8, 8);
                padding = (8, 8); locations = ((10, 20),)

                hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
                queryFeatures = hog.compute(queryImage, winStride, padding, locations)
                eps = 1e-7
                queryFeatures = queryFeatures.astype("float")
                queryFeatures /= (queryFeatures.sum() + eps)

            # load the index perform the search
            file = open(args["index"] + ".pkl", 'rb')
            index = pickle.load(file)
            searcher = Searcher(index)
            results = searcher.search(queryFeatures)

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
                path = args["dataset"] + os.sep + "%s" % (imageName)
                result = cv2.resize(cv2.imread(path), (400, 166))
                cv2.putText(result, imageName + ': ' + str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 3)
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
            desc_rgb = RGBHistogram([8, 8, 8])
            queryFeatures_rgb = desc_rgb.describe(queryImage)

            file_lbp = open("index_lbp.cpickle" + ".pkl", 'rb')
            index_lbp = pickle.load(file_lbp)
            searcher_lbp = Searcher(index_lbp)
            desc_lbp = LocalBinaryPatterns(24, 8)
            gray_lbp = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
            queryFeatures_lbp = desc_lbp.describe(gray_lbp)

            file_hog = open("index_hog.cpickle" + ".pkl", 'rb')
            index_hog = pickle.load(file_hog)
            searcher_hog = Searcher(index_hog)
            winSize = (64, 64); blockSize = (16, 16); blockStride = (8, 8);
            cellSize = (8, 8); nbins = 9; derivAperture = 1; winSigma = 4.;
            histogramNormType = 0; L2HysThreshold = 2.0000000000000001e-01;
            gammaCorrection = 0; nlevels = 64; winStride = (8, 8);
            padding = (8, 8); locations = ((10, 20),)

            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                    histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
            queryFeatures_hog = hog.compute(queryImage, winStride, padding, locations)
            eps = 1e-7
            queryFeatures_hog = queryFeatures_hog.astype("float")
            queryFeatures_hog /= (queryFeatures_hog.sum() + eps)

            results_rgb = searcher_rgb.search(queryFeatures_rgb)
            results_lbp = searcher_lbp.search(queryFeatures_lbp)
            results_hog = searcher_hog.search(queryFeatures_hog)

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
            # print(final.tail())

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
                # (score, imageName) = results[j]
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

