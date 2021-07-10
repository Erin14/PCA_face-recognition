import sys
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

__author__ = "Ruiling Chen"
__version__ = "0.1.0"


def PCA(face_data, meanface, eigfaces = None, n_components = None):
    '''
    Perform PCA on face_data.

    :param face_data: the dataset that we want to perform PCA.
    :param meanface: the meanface of training dataset.
    :param eigfaces: if the input face_data is test data, then this is the eigen faces from training dataset.
    :param n_components: the number of components.
    :return: all of the eigen faces and the data projection in the face space.
    '''
    data_centered = face_data - meanface
    eig_vectors = None
    # calculate the eigen faces for training dataset
    if eigfaces is None:
        CovMatrix = np.dot(data_centered.T, data_centered)
        _, eig_vectors = np.linalg.eigh(CovMatrix)

        eig_vectors = eig_vectors.T  # the output eigen vectors from np.linalg.eigh() are stored as columns
        eig_vectors = eig_vectors[::-1] # reorder
        eigfaces = eig_vectors[0:n_components]

    data_projection = np.transpose(np.dot(eigfaces, data_centered.T))
    return eig_vectors, data_projection

def reconstruct(original_data, meanface, eigfaces):
    '''
    Reconstruct the approximated faces depends on the given mean face and eigen faces.

    :param original_data: the original data that we want to reconstruct.
    :param meanface: the mean face of training dataset.
    :param eigfaces: the eigen faces of the face space.
    :return: the reconstruction data.
    '''
    data_centered = original_data - meanface
    weights = np.dot(data_centered, eigfaces.T)
    recon_data = meanface + np.dot(weights, eigfaces)
    return recon_data

def show_meanface(meanface):
    '''
    Show the picture of the mean face.

    :param meanface: the mean face data.
    '''
    plt.title('mean face')
    plt.imshow(meanface.reshape((32, 32), order='F'), cmap=plt.cm.gray)

def show_top_five_eiegenfaces(eigfaces):
    '''
    Show the top 5 eigenfaces.

    :param eigfaces: all of the eigen faces.
    '''
    plt.figure(figsize=(15, 3))
    for eigfacesIndex in range(5):
        plt.subplot(1, 5, eigfacesIndex + 1)
        plt.title('Top ' + str(eigfacesIndex + 1) + ' eigenface')
        plt.imshow(eigfaces[eigfacesIndex].reshape((32, 32), order='F'), cmap=plt.cm.gray)

def show_reconstructed_face(original_face, eigfaces, meanface, Ks):
    '''
    Show the reconstructed face of the original_face.

    :param original_face: the face we want to reconstruct.
    :param eigfaces: the eigen faces.
    :param meanface: the mean face.
    :param Ks: a list of K as the number of components.
    '''
    img_position = 1
    plt.figure(figsize=(20, 5))
    for k in Ks:
        recon_face = reconstruct(original_face, meanface, eigfaces[0:k])
        plt.subplot(1, 4, img_position)
        plt.title('Reconstructed number 80 face for K = ' + str(k))
        plt.imshow(recon_face.reshape((32, 32), order='F'), cmap=plt.cm.gray)
        img_position += 1

def my_classification(train_data, train_label, test_data):
    '''
    For every test data, calculate the minimum distance between them and the training data.
    Assign the corresponding label to my_label as our classification result.

    :param train_data: the training dataset
    :param train_label: the training label
    :param test_data: the test dataset
    :return: our classification result
    '''
    my_label = np.zeros(len(test_data), dtype = 'int') # store the identified result with 1NN
    for faceIndex in range(len(test_data)):
        # calculate the Euclidean distance between the test face and every training face in the face space.
        dists = [np.linalg.norm(train_data[trainIndex] - test_data[faceIndex])
                 for trainIndex in range(len(train_label))]
        # get the index of the minimum distance
        min_index = np.where(dists == min(dists))
        my_label[faceIndex] = train_label[min_index]
    return my_label

def plot_classification_rate(all_eigfaces, train_data, train_label, test_data, test_label):
    '''
    Plot the classification rate against K value (the number of components).

    :param all_eigfaces: all of the eigen faces.
    :param train_data: the training dataset.
    :param train_label: the training label.
    :param test_data: the test dataset.
    :param test_label: the test label.
    '''
    meanface = train_data.mean(axis=0)
    Ks = []  # store K values (from 1 to 300)
    accus = []  # store accuracy rate

    for K in range(300):
        K += 1  # loop for K from 1 to 300
        _, train_projection = PCA(train_data, meanface, all_eigfaces[0:K])
        _, test_projection = PCA(test_data, meanface, all_eigfaces[0:K])

        classification_label = my_classification(train_projection, train_label, test_projection)
        accu = sum(classification_label == test_label) / len(test_label)
        Ks.append(K)
        accus.append(accu)
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Ks, accus)
    ax.set_xlabel('K')
    ax.set_ylabel('accuracy rate')
    plt.show()

def show_incorrectly_classified_faces(eigfaces, meanface, train_data, train_label, test_data, test_label):
    '''
    Show the incorrectly classified faces and the nearest neighbour faces.

    :param eigfaces: top 100 eigen faces
    :param meanface: the mean face of training dataset
    :param train_data: the training data
    :param train_label: the training label
    :param test_data: the test data
    :param test_label: the test label
    '''
    _, train_projection = PCA(train_data, meanface, eigfaces)
    _, test_projection = PCA(test_data, meanface, eigfaces)

    classification_label = my_classification(train_projection, train_label, test_projection)
    theface_index = []  # the index of faces that are incorrectly classified
    nearest_index = []  # their nearest neighbours from the training set
    for labelIndex in range(len(classification_label)):
        if classification_label[labelIndex] != test_label[labelIndex]:
            theface_index.append(test_label[labelIndex])
            nearest_index.append(classification_label[labelIndex])
    plt.figure(figsize = (10, 5 * len(theface_index)))
    for index in range(len(theface_index)):
        face_position = 2 * index + 1
        plt.subplot(len(theface_index), 2, face_position)
        plt.imshow(test_data[theface_index[index]].reshape((32, 32), order='F'), cmap=plt.cm.gray)
        plt.subplot(len(theface_index), 2, face_position + 1)
        plt.imshow(train_data[nearest_index[index]].reshape((32, 32), order='F'), cmap=plt.cm.gray)
    plt.show()

def main():
    # deal with the input parameters
    train_datafile, train_labelfile, K, test_datafile, test_labelfile = sys.argv[1:6]

    # train_datafile, train_labelfile, K, test_datafile, test_labelfile = "faces_train.txt", "faces_train_labels.txt", 10, "faces_test.txt", "faces_test_labels.txt"
    
    K = int(K)
    train_data = np.genfromtxt(train_datafile, dtype='float')
    train_label = np.genfromtxt(train_labelfile, dtype='int')
    test_data = np.genfromtxt(test_datafile, dtype='float')
    test_label = np.genfromtxt(test_labelfile, dtype='int')

    meanface = train_data.mean(axis=0)
    all_eigfaces, train_projection = PCA(train_data, meanface, n_components = K)

    # show the mean face and the top 5 eigenfaces
    show_meanface(meanface)
    show_top_five_eiegenfaces(all_eigfaces)
    # show number 80 reconstructed face (number 79 in python) for K = 4, 16, 64, 128
    show_reconstructed_face(train_data[79], all_eigfaces, meanface, Ks=[4, 16, 64, 128])

    # plot the classification rate against K value (from 1 to 300)
    plot_classification_rate(all_eigfaces, train_data, train_label, test_data, test_label)

    # Show the incorrectly classified faces and the nearest neighbour faces with K = 100
    show_incorrectly_classified_faces(all_eigfaces[0:100], meanface, train_data, train_label, test_data, test_label)

    _, test_projection = PCA(test_data, meanface, all_eigfaces[0:K])
    classification_label = my_classification(train_projection, train_label, test_projection)
    accuracy = sum(classification_label == test_label) / len(test_label)
    print("The classification rate for K = {} is {}.".format(K, accuracy))

if __name__ == '__main__':
    main()



