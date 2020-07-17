# Image Descriptors

Image Search Engine using images of a specific dataset, using a combination of Feature Vector Encodings such as:

- RGB Histogram
- Local Binary Pattern (LBP)
- Histogram of Oriented Gradients (HOG)

## Requirements
- Python
- OpenCV
- Numpy
- Scikit-image

## Description

The idea of this project is to use some descriptors (color, shape, texture...) for an image search creating a Machine learning system, testing the combination of all of them. They will be classified thanks to the the histogram descriptors mentioned (RGB, LBP, HOG) in a Supervised Learning way.

The whole process consists first creating an index for each descriptor and then searching through it. The dataset can be found in https://www.vicos.si/Downloads/FIDS30 (download and place it in a folder called /images_all/), but in order to facilitate the training process I made a subset of them (folder /images_small/). Take into account that running the program for the entire dataset with all the descriptors could take several minutes to complete. As soon as you have an index for the specific descriptor, you can make the image search for the own dataset or even search external images.

## Usage

Combine the options as you wish, but the process is first creating and index, and then search using it, providing the dataset you want to use. In order to search using the combination of the three descriptors you have to previously create the specific indexes.

### Create specific descriptors to search
```
python index.py --dataset images_small --index index_rgb.cpickle --descriptor rgb

python index.py --dataset images_small --index index_lbp.cpickle --descriptor lbp

python index.py --dataset images_all --index index_hog.cpickle --descriptor hog (Note that this one use the big dataset)
```

### Search using one descriptor

```
python search.py --dataset images_small --index index_rgb.cpickle

python search.py --dataset images_all --index index_lbp.cpickle

python search.py --dataset images_all --index index_hog.cpickle
```

### Search external dataset using one descriptor

```
python search_external.py --dataset images_small --index index_rgb.cpickle --query images_ext --descriptor rgb

python search_external.py --dataset images_all --index index_lbp.cpickle --query images_ext --descriptor lbp

python search_external.py --dataset images_all --index index_hog.cpickle --query images_ext --descriptor hog
```

### Search combining descriptors

```
python search.py --dataset images_small --all yes

python search_external.py --dataset images_all --query images_ext --all yes (Note using complete dataset)
```