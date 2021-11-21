I am Simon Ziegler, cofounder of the Cladiac GmbH and software developer in the area of machine learning and computer vision. Here i want to share some sources.

This repository is for a python source for cell segmentation in microscopy images. The software was tested with those libraries:

OpenCV 4.2.0
Scikit-Image 0.17.2

Note: older versions of Scikit-Image gave wrong results in computing local maxima which is used for clustering.


The software is started with:

python cell_segmentation.py DIRECTORY

DIRECTORY is the directory which contains the input images. There are different channels:

Main channel: "DAPI" 
Red channel: "DsRed" or "Cy5"
Green channel: "Alexa"

The channel of an image is determined by having the channel name in the filename. Files are sorted by name and grouped by there ordering.


Result of the segmentation procedure is a directory with the name "result_TIMESTAMP" which contains the result files. Those are named as the channel files with the ending "MASK" and contain an single-channel integer image with pixel values corresponding to the segment (cell).

The "documentation.pdf" file is a description of how to use the software.


