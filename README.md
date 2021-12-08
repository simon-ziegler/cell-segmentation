This repository is for a python source for cell segmentation in microscopy images. The software was tested with the libraries in "requirements.txt". To install the libraries, use this command:

python -m pip install -r requirements.txt

Note: older versions of Scikit-Image gave wrong results in computing local maxima which is used for clustering. The applications are running with Python 3.7.5.


The cell segmentation software is started with:

python cell_segmentation.py DIRECTORY

DIRECTORY is the directory which contains the input images. There are different channels:

Main channel (basis of cell identification): "<span>&#95;</span>DAPI"<br> 
Red channel: "<span>&#95;</span>DsRed" or "<span>&#95;</span>Cy5"<br>
Green channel: "<span>&#95;</span>Alexa"<br>

The channel of an image is determined by having the channel name in the filename. Files are sorted by name and grouped by there ordering.


Result of the segmentation procedure is a directory with the name "TIMESTAMP_result" which contains the result files. Those are named as the channel files with the ending "<span>&#95;</span>MASK" and contain an single-channel integer image with pixel values corresponding to the segment (cell).

The "documentation.pdf" file is a description of how to use the segmentation software.

<br><br>
The cell classification software is started with:

python cell_intensity.py DIRECTORY

Here, DIRECTORY is the name of the output directory of the cell segmentation. For the results, a directory named "TIMESTAMP_classified" contains the result files. Those are by last part in the name:

"<span>&#95;</span>class_green" : classes marked in green channel<br>
"<span>&#95;</span>class_red" : classes marked in red channel<br>
"<span>&#95;</span>ID" : all localized cells with ID number<br>
"<span>&#95;</span>info" : table with cell information<br>
"<span>&#95;</span>scaled_green" : original green channel image intensity scaled<br>
"<span>&#95;</span>scaled_red" : original red channel image intensity scaled<br>

The "documentation_classification.pfd" describes how to use the classification software.
