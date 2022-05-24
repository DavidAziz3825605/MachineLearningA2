********************************************************
********************************************************
COSC2673 - Traffic Sign Data set
********************************************************
********************************************************

This data set has been prepared using the BelgiumTSC Benchmark training data set.
The original images have been converted to greyscale and downsampled to a consistent 28x28 dimension.

The target classification classes of the images have been adjusted, with original classes being merged. 

Information about the original data set us retained below.





********************************************************
BelgiumTSC - The Belgium Traffic Sign Classification Benchmark
********************************************************

This archive contains the training set of the 
"Belgium Traffic Sign Classification Benchmark".

This training set is a subset of the BelgiumTS - Belgium Traffic Signs database:
http://homes.esat.kuleuven.be/~rtimofte/
http://www.vision.ee.ethz.ch/~timofter/


and follows the structure used in GTSRB Challenge proposed for IJCNN 2011:
http://benchmark.ini.rub.de/


**********************************************
Archive content
**********************************************
This archive contains the following structure:

There is one directory for each of the 62 classes (00000 - 00061).
Each directory contains the corresponding training images and one 
text file with annotations, eg. GT-00000.csv. 

In total are 4591 images for training. On average for each physically distinct
traffic sign there are 3 images available.


**********************************************
Image format and naming
**********************************************
The images are PPM images (RGB color). Files are numbered in two parts:

   XXXXX_YYYYY.ppm

The first part, XXXXX, represents the pole number from where is in the full
annotations of BelgiumTS database. All images of one class 
with identical pole numbers originate from the same pole and more represent
one single physical traffic sign.

The second part, YYYYY, is a running number for the views where the traffic
sign is annotated. There is no temporal order
of the images, but they were where extracted in the order of the annotations on
the original pole from BelgiumTS.


**********************************************
Annotation format
**********************************************

<removed - this does not apply for COSC2673>


**********************************************
How to cite?
**********************************************

@inproceedings{Timofte-BMVC-2011,
author = {Radu Timofte and Luc Van Gool},
title  = {Sparse representation based projections},
booktitle = {British Machine Vision Conference},
year   = {2011},
}

@article{Timofte-MVA-2011,
  author = {Radu Timofte and Karel Zimmermann and Luc {Van Gool}},
  title = {Multi-view Traffic Sign Detection, Recognition, and 3D Localisation},
  journal = {Machine Vision and Applications},
  year = {2011}, 
  doi  = {10.1007/s00138-011-0391-3},
}


**********************************************
Further information
**********************************************
For more information please visit the website at

http://homes.esat.kuleuven.be/~rtimofte/
http://www.vision.ee.ethz.ch/~timofter/

If you have any questions, do not hesitate to contact us 
    
	Radu.Timofte@esat.kuleuven.be
	Radu.Timofte@vision.ee.ethz.ch

**********************************************
Centre for Processing Speech and Images (PSI/ESAT)
VISion in Industry, Communications, and Services (VISICS)

Katholieke Universiteit Leuven
Belgium
**********************************************
