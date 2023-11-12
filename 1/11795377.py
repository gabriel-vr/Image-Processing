'''
Name: Gabriel Vicente Rodrigues
NUSP: 11795377
Course: Image Processing and Analysis (2023)
Year: 2023
Semester: 1
Title: Assigment 1
'''

###################################################################### 
# imports

import numpy as np
import imageio



###################################################################### 
# utils

def cumulativeHistogram(image, rangeValues):
    hist = np.zeros(rangeValues, dtype=np.uint64)
    for i in range(rangeValues):
        hist[i] = np.sum(image == i)
        #acumulates the values from previous positions into actual position
        if i > 0:
            hist[i]+=hist[i-1]
    return hist


def superresolution(images):
    M = images[0].shape[0]
    N = images[0].shape[1]
    superImage = np.zeros((M*2, N*2), dtype=np.uint8)

    for x in range(superImage.shape[0]):
        for y in range(superImage.shape[1]):
            #This logic selects wich image is responsible for the actual pixel of the super image
            #If x is pair, it is between images 0 and 1 and y decides
            #if x is odd, it is between images 2 and 3 and y decides 
            idxImage = x%2
            idxImage += y%2

            #As the coordinates of the super image are doubled, we have to divide them to correctly
            #access the low resolution image
            superImage[x, y] = images[idxImage][int(x/2), int(y/2)]
    return superImage

def compareImages(img1, img2):
    actualSum = 0.0
    M = img1.shape[0]
    N = img1.shape[1]
    for x in range(M):
        for y in range(N):
            actualSum+= np.power((float(img1[x, y]) - float(img2[x, y])), 2.0)
    return np.power(actualSum/(M * N), 0.5)



###################################################################### 
#Transformations

def gammaTransformation(img, g):
    enhancedImage = (255 *  np.power(img.astype(np.float64) / 255.0, 1/g)).astype(np.uint8)
    return enhancedImage


def cumulativeHistogramTransformation(image, hist,rangeValues, qImages = 1):
    result = np.zeros(image.shape, dtype=np.uint8)
    M = image.shape[0]
    N = image.shape[1]

    for i in range(rangeValues):
        # the qImages is used here to normalize the result according to the amount of images used
        # to create the histogram

        s = (((rangeValues-1)/float(M*N*qImages))*hist[i]).astype(np.uint8)

        result[np.where(image == i)] = s
    
    return result





###################################################################### 
# operations

def op0(images):
    return images

def op1(images):
    enhancedImages = []
    for image in images:
        hist = cumulativeHistogram(image, 256)
        enhancedImages.append(cumulativeHistogramTransformation(image, hist,256))
    return enhancedImages

def op2(images):
    enhancedImages = []
    hist = np.zeros(256, dtype=np.uint64)
    for image in images:
        actualHist = cumulativeHistogram(image, 256)
        hist+=actualHist
    for image in images:
        enhancedImages.append(cumulativeHistogramTransformation(image, hist, 256, 4))
    return enhancedImages


def op3(images, g):
    enhancedImages = []
    for image in images:
        enhancedImages.append(gammaTransformation(image, g))
    return enhancedImages


operations = {
    0: op0,
    1: op1,
    2: op2,
    3: op3
}



###################################################################### 


# Reading Inputs

lowResolutionImageBaseName = str(input())
highResolutionImageFileName = str(input())
enhancementMethodId = int(input())
enhancementMethodParam = float(input())





# reading images

lowResolutionImages = []
for i in range(4):
    lowResolutionImages.append(imageio.imread("{}{}.png".format(lowResolutionImageBaseName, i)))

highResolutionImage = imageio.imread(highResolutionImageFileName)



#calling operations

if(enhancementMethodId == 3):
    results = operations[enhancementMethodId](lowResolutionImages, enhancementMethodParam)
else:    
    results = operations[enhancementMethodId](lowResolutionImages)


superImage = superresolution(results)

print(round(compareImages(superImage, highResolutionImage), 4))