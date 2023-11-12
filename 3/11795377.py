'''
Name: Gabriel Vicente Rodrigues
NUSP: 11795377
Course: Image Processing and Analysis (2023)
Year: 2023
Semester: 1
Title: Assigment 3
'''

###################################################################### 
# imports

import numpy as np
import imageio.v3 as iio
from scipy.ndimage import convolve

###################################################################### 
# utils

def normalize(img, factor):
    img = np.array(img, dtype=np.float32)
    min = img.min()
    max = img.max()

    imageNorm = (img - min) / (max - min)
    imageNorm = imageNorm * factor
    return imageNorm

def luminance(img):
    image = np.array(img, copy=True).astype(np.float32)

    newImage = np.zeros((image.shape[0], image.shape[1]))

    newImage = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    
    newImage = np.floor(newImage)

    return normalize(newImage, 255)

def euclidianDist(a, b):
    dist = np.sqrt(np.sum(np.square(a-b)))
    return dist




###################################################################### 
# operations

WSx = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

WSy = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

def hog(image):
    gx = convolve(image, WSx)
    gy = convolve(image, WSy)

    M = np.sqrt(np.square(gx) + np.square(gy)) / np.sum(np.sqrt(np.square(gx) + np.square(gy)))

    np.seterr(divide='ignore', invalid='ignore')
    angles = np.arctan(gy/gx) + (np.pi/2)
    angles = np.degrees(angles)
    angles = (angles/20).astype(np.uint8)

    d = np.zeros((9,), dtype=np.float64)

    # for each bin
    for i in range(9):
        #sum from M in indexes where angle is equal to bin
        sumBin = np.sum(M[np.where(angles == i)])
        d[i] = sumBin
    return d


class KNN:
    def __init__(self, k) -> None:
        self.k = k

    def fit(self, X, y) -> None:
        self.X = np.array(X, dtype=np.float32, copy=True)
        self.y = np.array(y, dtype=np.float32, copy=True)


    def predict(self, X_test) -> np.ndarray:
        
        X_test_copy = np.array(X_test, dtype=np.float32, copy=True)

        results = np.zeros(X_test_copy.shape[0], dtype=np.uint8)

        # for each entry
        for i in range(X_test_copy.shape[0]):
            distances = []

            # calculate distance between entry and data stored
            for j in range(self.X.shape[0]):
                d = euclidianDist(self.X[j], X_test_copy[i])
                distances.append({'distance': d, 'label': self.y[j]})

            #sort using distance
            distances = sorted(distances, key=lambda d: d['distance'])

            # sum the first k using the label
            sum = 0
            for j in range(self.k):
                sum+=distances[j]['label']

            # if the sum is greater than 1, than 1 is choosen as the label for the test
            if(sum >1):
                results[i] = 1

        return results







###################################################################### 


# Reading Inputs

inputImageNamesNH = str(input()).replace('\r', '').split(' ')
inputImageNamesH = str(input()).replace('\r', '').split(' ')
inputImageNamesT = str(input()).replace('\r', '').split(' ')

imagesNH = []
for imageName in inputImageNamesNH:
    image = iio.imread(imageName)
    imagesNH.append(luminance(image))


imagesH = []
for imageName in inputImageNamesH:
    image = iio.imread(imageName)
    imagesH.append(luminance(image))

imagesT = []
for imageName in inputImageNamesT:
    image = iio.imread(imageName)
    imagesT.append(luminance(image))


descriptorsTrain = []
descriptorsTest = []

labelTrain = []

for image in imagesNH:
    labelTrain.append(0)
    descriptorsTrain.append(hog(image))

for image in imagesH:
    labelTrain.append(1)
    descriptorsTrain.append(hog(image))

for image in imagesT:
    descriptorsTest.append(hog(image))


knn = KNN(
    k=3
)

knn.fit(descriptorsTrain, labelTrain)

y_predicted = knn.predict(descriptorsTest).astype(str)

print(" ".join(y_predicted))


