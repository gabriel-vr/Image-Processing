'''
Name: Gabriel Vicente Rodrigues
NUSP: 11795377
Course: Image Processing and Analysis (2023)
Year: 2023
Semester: 1
Title: Assigment 2
'''

###################################################################### 
# imports

import numpy as np
import imageio.v3 as iio


###################################################################### 
# utils

def compareImages(img1, img2):
    actualSum = 0.0
    M = img1.shape[0]
    N = img1.shape[1]
    for x in range(M):
        for y in range(N):
            actualSum+= np.power((float(img1[x, y]) - float(img2[x, y])), 2.0)
    return np.power(actualSum/(M * N), 0.5)


###################################################################### 
#Filters


def lowPass(shape, parameters):
    P = shape[0]
    Q = shape[1]
    H = np.zeros(shape, dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            d = np.sqrt(np.power(u-P/2, 2) + np.power(v-Q/2, 2))
            if d <= parameters[0]:
                H[u,v] = 1 
    
    return H

def highPass(shape, parameters):
    P = shape[0]
    Q = shape[1]
    H = np.ones(shape, dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            d = np.sqrt(np.power(u-P/2, 2) + np.power(v-Q/2, 2))
            if d <= parameters[0]:
                H[u,v] = 0
    
    return H

# lowpass with r0 - low pass with r1
def bandPass(shape, parameters):
    inR = parameters[0] if parameters[0] < parameters[1] else parameters[1]
    outR = parameters[1] if parameters[0] < parameters[1] else parameters[0]
    return lowPass(shape, [outR]) - lowPass(shape, [inR])

# 1 - laplacian low pass
def laplacianHighPass(shape, parameters):
    P = shape[0]
    Q = shape[1]
    H = np.zeros(shape, dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            H[u,v] = 1- (-4*np.power(np.pi,2)*(np.power(u - (P/2.0), 2) + np.power(v-(Q/2),2)))
    
    return H

def gaussianLowPass(shape, parameters):
    std1 = parameters[0]
    std2 = parameters[1]
    P = shape[0]
    Q = shape[1]
    H = np.zeros(shape, dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            x = (np.power(u-P/2, 2)/(2*np.power(std1, 2))) + (np.power(v-Q/2, 2)/(2*np.power(std2, 2)))
            H[u,v] = np.exp(-x)
    
    return H

def butterworthLowPass(shape, parameters):
    d0 = parameters[0]
    n = parameters[1]
    P = shape[0]
    Q = shape[1]
    H = np.zeros(shape, dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            d = np.sqrt(np.power(u-P/2, 2) + np.power(v-Q/2, 2))
            H[u,v] = 1/(1+np.power(d/d0, 2*n))
    
    return H


def  butterworthHighPass(shape, parameters):
    
    return 1 - butterworthLowPass(shape, parameters)

def noFrequency(shape, parameters):
    H = np.zeros(shape, dtype=np.float32)
    return H





filters = {
    0: lowPass,
    1: highPass,
    2: bandPass,
    3: laplacianHighPass,
    4: gaussianLowPass,
    5: butterworthLowPass,
    6: butterworthHighPass,
    7: noFrequency,
    8: noFrequency
}


###################################################################### 
# operations

def filterImage(image, filterIdx,parameters):
    F = np.fft.fftshift(np.fft.fft2(image))
    H = filters[filterIdx](F.shape, parameters)
    resultF = F * H
    response = np.fft.ifft2(np.fft.ifftshift(resultF)).real

    """
        Very important:
        The next step is needed because the response variable is not in the range 0 - 255, so we need to do this scalling
    """
    response = (255*(response - np.min(response)) / (np.max(response) - np.min(response))).astype(np.uint8)

    return response




###################################################################### 


# Reading Inputs

inputImageName = str(input()).replace('\r', '')
expectedImageName = str(input()).replace('\r', '')
filterIdx = int(input())
filterParams = []

while True:
    try:
        value = float(input())
        filterParams.append(value)

    except EOFError:
        break



inputImage = iio.imread(inputImageName)
expectedImage = iio.imread(expectedImageName)


#calling operations
result = filterImage(inputImage, filterIdx,filterParams)


print(round(compareImages(result, expectedImage), 4))