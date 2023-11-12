'''
Name: Gabriel Vicente Rodrigues
NUSP: 11795377
Course: Image Processing and Analysis (2023)
Year: 2023
Semester: 1
Title: Assigment 4
'''

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
 
# utils

def printPoints(points: list):
    printStrings = []
    for point in points:
        printStrings.append("({} {})".format(point[0], point[1]))
    
    print(' '.join(printStrings))




def fillFlood(image: np.ndarray, initialPoint: tuple, c: int):
    M, N = image.shape

    #vector that holds points to be looked
    pointsToView = [initialPoint]

    #vector that holds points of the region
    pointsInRegion = []

    color = image[initialPoint[0], initialPoint[1]]
    
    while(len(pointsToView) > 0):
        
        actualPoint = pointsToView.pop()

        # this test is important for points that are reoeted in the pointsToView array
        # ex: [(4, 4), (4, 5), (4, 5)]
        if(image[actualPoint[0], actualPoint[1]] != color):
            continue

        image[actualPoint[0], actualPoint[1]] = (color + 1)%2
        
        pointsInRegion.append(actualPoint)
        
        if(actualPoint[0] + 1 < M and image[actualPoint[0] + 1, actualPoint[1]] == color):
            pointsToView.append((actualPoint[0] + 1, actualPoint[1]))


        if(actualPoint[0] - 1 >=0 and image[actualPoint[0] - 1, actualPoint[1]] == color):
            pointsToView.append((actualPoint[0] - 1, actualPoint[1]))

        
        if(actualPoint[1] + 1 < N and image[actualPoint[0], actualPoint[1] + 1] == color):
            pointsToView.append((actualPoint[0], actualPoint[1] + 1))

        
        if(actualPoint[1] - 1 >=0 and image[actualPoint[0], actualPoint[1] - 1] == color):
            pointsToView.append((actualPoint[0], actualPoint[1] - 1))


        if c == 8:
            if(actualPoint[0] + 1 < M and actualPoint[1] + 1 < N and image[actualPoint[0] + 1, actualPoint[1] + 1] == color):
                pointsToView.append((actualPoint[0] + 1, actualPoint[1] + 1))
        
            if(actualPoint[0] - 1 >=0 and actualPoint[1] -1 >= 0 and image[actualPoint[0] - 1, actualPoint[1] - 1] == color):
                pointsToView.append((actualPoint[0] - 1, actualPoint[1] - 1))

            
            if(actualPoint[1] + 1 < N and actualPoint[0] - 1 >= 0 and image[actualPoint[0] - 1, actualPoint[1] + 1] == color):
                pointsToView.append((actualPoint[0] - 1, actualPoint[1] + 1))

            
            if(actualPoint[1] - 1 >=0 and actualPoint[0] + 1 < M and image[actualPoint[0] + 1, actualPoint[1] - 1] == color):
                pointsToView.append((actualPoint[0] + 1, actualPoint[1] - 1))
    
    
    return pointsInRegion



# Reading Inputs

inputImageName = str(input()).replace('\r', '')
coordinateX = int(input())
coordinateY = int(input())
connectivity = int(input())

image = (iio.imread(inputImageName) > 127).astype(np.uint8)

points = fillFlood(image, (coordinateX, coordinateY), connectivity)


points.sort()

printPoints(points)

