# PCA_face-recognition

### A Nearest-Neighbour Face Recognition Algorithm based on Eigenfaces

We will be using a database consisting of 10 images each of 40 different people. The image resolution is 32x32, corresponding to the extrinsic dimension of 1024. The goal is to correctly identify the person from a test image using a low-dimensional eigenface representation.

The datasets are faces_train.txt and faces_train.txt. Each row in these files contain 32x32 = 1024 pixel values. We will be using 7 images per person for training and 3 images for testing. The first 7 images in the training data and the first 3 images in the test data is of person A and so on. The labels for the two datasets are given in faces_train_labels.txt and faces_test_labels.txt.

### Inputs and Outputs

The program should take a command-line parameter that contains the name of the file containing the training data, training labels, the number of principal components to use (K), the name of the file containing the training data and test labels.

For example (with K=10):
```
python eigenfacesassignment.py faces_train.txt faces_train_labels.txt 10 faces_test.txt faces_test_labels.txt
```

The outputs would be:
* The picture of the mean face and top 5 eigenfaces computed by PCA.
* Project and reconstruct one of the face images in test set using different values of K. Show the results for 4 different K’s.
* A plot of the nearest-neighbour (1NN) classification rate (on the test data) as a function of K.
* Pictures of incorrectly classified faces and their nearest neighbours from the training set for K=100.
