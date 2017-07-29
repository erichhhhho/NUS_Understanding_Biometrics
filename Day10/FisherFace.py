import os
import numpy as np
import scipy.linalg as linalg
import cv2
import operator
from PIL import Image
from matplotlib import pyplot




def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1, col))

    for i in range(col):
        r[0, i] = linalg.norm(x[:, i])
    return r


def myLDA(A, Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection
    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim, datanum = A.shape
    totalMean = np.mean(A, 1)
    partition = [np.where(Labels == label)[0] for label in classLabels]
    classMean = [(np.mean(A[:, idx], 1), len(idx)) for idx in partition]

    # compute the within-class scatter matrix
    W = np.zeros((dim, dim))
    for idx in partition:
        W += np.cov(A[:, idx], rowvar=1) * len(idx)

    # compute the between-class scatter matrix
    B = np.zeros((dim, dim))
    for mu, class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset, offset) * class_size

    # solve the generalized eigenvalue problem for discriminant directions
    ew, ev = linalg.eig(B, W)

    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind, val in sorted_pairs[:classNum - 1]]
    LDAW = ev[:, selected_ind]
    Centers = [np.dot(mu, LDAW) for mu, class_size in classMean]
    Centers = np.array(Centers).T
    return LDAW, Centers, classLabels


def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A

    # Note: "lambda" is a Python reserved word


    # compute mean, and subtract mean from every column
    [r, c] = A.shape
    m = np.mean(A, 1)
    A = A - np.tile(m, (c, 1)).T
    B = np.dot(A.T, A)
    [d, v] = linalg.eig(B)

    # sort d in descending order
    order_index = np.argsort(d)
    order_index = order_index[::-1]
    d = d[order_index]
    v = v[:, order_index]

    # compute eigenvectors of scatter matrix
    W = np.dot(A, v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1

    LL = d[0:-1]

    W = W2[:, 0:-1]  # omit last column, which is the nullspace

    return W, LL, m


def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = []  # Label will store list of identity label

    # browsing the directory
    for f in os.listdir(directory):
        if not f[-3:] == 'bmp':
            continue
        infile = os.path.join(directory, f)
        im = cv2.imread(infile, 0)
        # turn an array into vector
        im_vec = np.reshape(im, -1)
        A.append(im_vec)
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A, dtype=np.float32)
    faces = faces.T
    idLabel = np.array(Label)

    return faces, idLabel


def float2uint8(arr):
    mmin = arr.min()
    mmax = arr.max()
    arr = (arr - mmin) / (mmax - mmin) * 255
    arr = np.uint8(arr)
    return arr


if __name__ == '__main__':
    # write your code here
    directory = 'F:/Desktop/NUS Summer/HW5/face/train'
    K = 30
    faces, idLabel = read_faces(directory)
    W, LL, m = myPCA(faces)
    We = W[:, :K]

    # """Calculate the mean of columns in test dataset m_TestData"""
    # test=[]
    # for f in os.listdir(directory):
    #     if not f[-3:] =='bmp':
    #         continue
    #     infile = os.path.join(directory, f)
    #     im = cv2.imread(infile, 0)
    #     # turn an array into vector
    #     im_vec = np.reshape(im, -1)
    #     test.append(im_vec)
    #
    # TestData=np.array(test, dtype=np.float32).T
    # m_TestData = np.mean(TestData, 1)


    """Calculate the Z matrix, whose dimesion is Df-by-10 and contains the mean feaure vector
    of each person.(ith column corresponds to person with label i) e.g K=30"""
    Z = []
    count = 0
    temp = np.array([0])
    for f in os.listdir(directory):
        if not f[-3:] == 'bmp':
            continue
        infile = os.path.join(directory, f)
        im = cv2.imread(infile, 0)
        # turn an array into vector
        im_vec = np.reshape(im, -1)
        y = np.dot(We.transpose(), im_vec - m)
        temp = temp + y
        count += 1
        if count == 12:
            temp = temp / 12
            Z.append(temp)
            temp = np.array([0])
            count = 0

    Z = np.array(Z, dtype=np.float32)
    Z = np.array(Z).transpose()

    """Perform identification of test data"""
    directory = 'F:/Desktop/NUS Summer/HW5/face/test'
    Label = []  # Label will store list of identity label
    OutputLabel = []  # Label will store list of result label
    for f in os.listdir(directory):
        if not f[-3:] == 'bmp':
            continue
        infile = os.path.join(directory, f)
        im = cv2.imread(infile, 0)
        # turn an array into vector
        im_vec = np.reshape(im, -1)

        name = f.split('_')[0][-1]
        Label.append(int(name))

        # Project to the PCA feature space
        y = np.dot(We.transpose(), im_vec - m)
        #print(m)
        # Search for the closest template on Z(Using Euclidean distance metric)
        count = 0
        result = Z[:, 0]

        min = 10000000
        for col in Z.transpose():
            count += 1
            if np.linalg.norm(col - y) < min:
                # nearest distance min
                min = np.linalg.norm(col - y)
                # identification result label
                temp = count - 1
                # match template on Z
                result = col

        OutputLabel.append(temp)
        ResultLabel = np.array(OutputLabel)
        Label1 = np.array(Label)

    """Draw the Confusion Matrix"""
    confusion_array = np.zeros([10, 10])
    Result = ResultLabel.reshape((10, 12))
    Labels = Label1.reshape((10, 12))

    for j in range(0, 10):
        for items in Result[j]:
            if items == j:
                confusion_array[j][j] += 1
            elif items == 0:
                confusion_array[j][0] += 1
            elif items == 1:
                confusion_array[j][1] += 1
            elif items == 2:
                confusion_array[j][2] += 1
            elif items == 3:
                confusion_array[j][3] += 1
            elif items == 4:
                confusion_array[j][4] += 1
            elif items == 5:
                confusion_array[j][5] += 1
            elif items == 6:
                confusion_array[j][6] += 1
            elif items == 7:
                confusion_array[j][7] += 1
            elif items == 8:
                confusion_array[j][8] += 1
            elif items == 9:
                confusion_array[j][9] += 1
    print(confusion_array)

    """2.Draw the mean face and top 8 eigenfaces"""
    meanface=float2uint8(m.reshape((160, 140)))
    meanfaceimage=Image.fromarray(meanface)
    eigenfaces=np.array([float2uint8(We[:,0].reshape((160, 140)))]*9)
    eigenfacesimage=np.array([3]*9)

    for i in range(0,9):
        eigenfaces[i]=float2uint8(We[:,i].reshape((160, 140)))

    axes=pyplot.subplot(331)
    pyplot.imshow(Image.fromarray(eigenfaces[0]), 'gray')
    pyplot.title('Eigenface1')
    axes.set_xticks([])
    axes.set_yticks([])

    axes =pyplot.subplot(332)
    pyplot.imshow(Image.fromarray(eigenfaces[1]), 'gray')
    pyplot.title('Eigenface2')
    axes.set_xticks([])
    axes.set_yticks([])

    axes =pyplot.subplot(333)
    pyplot.imshow(Image.fromarray(eigenfaces[2]), 'gray')
    pyplot.title('Eigenface3')
    axes.set_xticks([])
    axes.set_yticks([])

    axes =pyplot.subplot(334)
    pyplot.imshow(Image.fromarray(eigenfaces[3]), 'gray')
    pyplot.title('Eigenface4')
    axes.set_xticks([])
    axes.set_yticks([])

    axes =pyplot.subplot(335)
    pyplot.imshow(Image.fromarray(eigenfaces[4]), 'gray')
    pyplot.title('Eigenface5')
    axes.set_xticks([])
    axes.set_yticks([])

    axes =pyplot.subplot(336)
    pyplot.imshow(Image.fromarray(eigenfaces[5]), 'gray')
    pyplot.title('Eigenface6')
    axes.set_xticks([])
    axes.set_yticks([])

    axes =pyplot.subplot(337)
    pyplot.imshow(Image.fromarray(eigenfaces[6]), 'gray')
    pyplot.title('Eigenface7')
    axes.set_xticks([])
    axes.set_yticks([])

    axes =pyplot.subplot(338)
    pyplot.imshow(Image.fromarray(eigenfaces[7]),'gray')
    pyplot.title('Eigenface8')
    axes.set_xticks([])
    axes.set_yticks([])

    axes =pyplot.subplot(339)
    pyplot.imshow(meanfaceimage, 'gray')
    pyplot.title('Mean')
    axes.set_xticks([])
    axes.set_yticks([])

    pyplot.show()
    #
    # for x in range(0,9):
    #     We[:,x].reshape((160, 140))
