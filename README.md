# CBIR-SYSTEM
This project presents an efficient content-based image retrieval technique for searching similar images using a hybrid feature which is the combination of two feature that is LBP (Local Binary Pattern) and DCT (Discrete Cosine Transform). 
# Software-tools
  - Python 3.7
  - OpenCV 3.4.1
  - Numpy
  - OS
  - XlsxWriter
  - Pandas
  - Matplotlib
  - Random
  - Pickle
  - scikit-learn
# Database Description
For the evaluation of the proposed method a general purpose Corel database containing 1000 images of 10 different classes in jpg format of size 80×120 or 120×80 is used. The image set comprises 100 images in each of 10 different classes as butterflies, flowers and cat.
# Feature Description
The effectiveness of CBIR Systems depends on the performance of the algorithm for extracting the features of an image. In order to increase the performance of the system, we have to choose features correspondingly. The descriptors for finding the feature vector using visual content can be local or global. So here the local feature that we have used to extract feature is LBP (Local binary pattern) and the global feature that we have used is DCT (Discrete Cosine Transform). We are combining both to get a hybrid feature. Which we can use during the retrieval process.
