# ------------------------- IMPORTS -------------------------

import cv2
import time
import numpy as np
import math
import os
import sys
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib import colors

# ------------------------ FUNCTIONS ------------------------

# 1.1 GLOBAL MAP
def showWebcam(n):
    cam = cv2.VideoCapture(n)
    while True:
        ret_val, img = cam.read()
        ROI = img[0:1000,0:1000]
        cv2.imshow('my webcam', ROI)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def takePicture(n):
    cam = cv2.VideoCapture(n)
    check, frame = cam.read()
    ROI = frame[0:1000,0:1000]

    if check == True :
        img_name = 'images/image_0.png'
        cv2.imwrite(img_name,ROI)
    else :
        print("Error during the webcam opening")

    cam.release()
    return check

def loadImages(check):
    if check:
        mapImg = cv2.imread('images/image_0.png', cv2.IMREAD_COLOR)
        mapBw = cv2.imread('images/image_0.png', cv2.IMREAD_GRAYSCALE)
        eq = cv2.equalizeHist(mapBw)
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
        eq = clahe.apply(mapBw)
        
        return mapImg, mapBw, eq


# 1.2 FILTERING GLOBAL MAP
def globalFilter(eq):
    # Pre-processing the image through filtering and thresholding
    bilateral = cv2.bilateralFilter(eq,9,25,25)
    thresh = cv2.threshold(bilateral, 90, 255, cv2.THRESH_BINARY_INV)[1]

    # Applying morphological operators to the image
    kernelSquare13 = np.ones((13,13),np.float32)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelSquare13)
    
    return bilateral, thresh, opened


# 1.3 FILTERING OUT OBSTACLES
def obstacleFilter(opened):
    # Tophat to remove obstacles
    kernelRound70 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(70,70))
    tophat = cv2.morphologyEx(opened, cv2.MORPH_TOPHAT, kernelRound70)
    tophat = cv2.erode(tophat, None, iterations=2)

    # Morphological operators to seperate circles
    kernelRound40 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40))
    kernelRound25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    bigCircles = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernelRound40)
    bigCircles = cv2.erode(bigCircles, None, iterations=2)
    smallCircles = cv2.morphologyEx(tophat, cv2.MORPH_TOPHAT, kernelRound40)
    smallCircles = cv2.morphologyEx(smallCircles, cv2.MORPH_OPEN, kernelRound25)
    smallCircles = cv2.erode(smallCircles, None, iterations=2)
    circles = cv2.addWeighted(smallCircles,1,bigCircles,1,0)
    
    return tophat, circles


# 1.4 THYMIO DETECTION
def green_thym_pos(mapImg):
    # this part find the green thymio    
    RED_MIN = np.array([70, 50, 50],np.uint8)
    RED_MAX = np.array([100, 255, 255],np.uint8)

    hsv_img = cv2.cvtColor(mapImg, cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, RED_MIN, RED_MAX)
    
    kernelRound15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    frame_threshed = cv2.erode(frame_threshed, kernelRound15, iterations=1)

    kernelRound100 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
    frame_threshed = cv2.morphologyEx(frame_threshed, cv2.MORPH_CLOSE, kernelRound100)
    
    kernelRoundOP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
    frame_threshed = cv2.morphologyEx(frame_threshed, cv2.MORPH_OPEN, kernelRoundOP)
    
    kernelRoundER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
    frame_threshed = cv2.erode(frame_threshed, kernelRoundER, iterations=1)

    pos_x=np.argmax(np.argmax(frame_threshed,1))
    pos_y=np.argmax(np.argmax(frame_threshed,0))
    #print(pos_x,pos_y)

    return pos_x,pos_y

def angleX(x, y):
    if (x==0 and y==0):
        angle = 0
    elif (x == 0):
        if y < 0:
            angle = 90
        else:
            angle = 270
    else:
        if x > 0:
            angle = (np.arctan(y/x)*180/np.pi)%360
        else:
            angle = (180+np.arctan(y/x)*180/np.pi)%360
            
    return angle

def thymioDetection(circles, mapImg):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 75;
    params.maxThreshold = 200;
    params.filterByConvexity = False
    params.filterByCircularity = True
    params.minCircularity = 0.85
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 20000
    params.filterByColor = True
    params.blobColor = 255;

    # Detect Blobs
    detector = cv2.SimpleBlobDetector_create(params)

    # Generate keypoints
    keypoints = detector.detect(circles)
    pts = cv2.KeyPoint_convert(keypoints)
    thymioPos = np.zeros((2,3))
    smallCircles = np.zeros((2,2))

    # Get position (x, y, teta)
    if len(pts) == 4:
        n = 3
        count = 0
        while n != 0:
            for i in range(n):
                dist = math.sqrt((pts[n][0] - pts[i][0])**2 + (pts[n][1] - pts[i][1])**2)
                if dist < 110:
                    if keypoints[n].size < keypoints[i].size:
                        smallCircles[count][0] = int(round(pts[n][0]))
                        smallCircles[count][1] = int(round(pts[n][1]))
                        thymioPos[count][0] = int(round(pts[i][0]))
                        thymioPos[count][1] = int(round(pts[i][1]))
                        thymioPos[count][2] = int(round(angleX(pts[n][0]-pts[i][0], pts[n][1]-pts[i][1])))
                    else:
                        smallCircles[count][0] = int(round(pts[i][0]))
                        smallCircles[count][1] = int(round(pts[i][1]))
                        thymioPos[count][0] = int(round(pts[n][0]))
                        thymioPos[count][1] = int(round(pts[n][1]))
                        thymioPos[count][2] = int(round(angleX(pts[i][0]-pts[n][0], pts[i][1]-pts[n][1])))
                    count += 1                   
            n -= 1
    
    # switch thmio0 is red
    pos_x,pos_y= green_thym_pos(mapImg)
    dist = math.sqrt((thymioPos[0][0] - pos_y)**2 + (thymioPos[0][1] - pos_x)**2)
    #print("dist", dist)
    if dist > 120:
        #print(thymioPos)
        thymioPos = thymioPos[::-1]
        smallCircles= smallCircles[::-1]
        #print(thymioPos)
    
    dist = math.sqrt((thymioPos[0][0] - pos_y)**2 + (thymioPos[0][1] - pos_x)**2)
    #print("dist", dist)
    
    thymioGPos = thymioPos[0]
    thymioRPos = thymioPos[1]
    
    mapWithShapes = mapImg.copy()
    cv2.circle(mapWithShapes,(int(thymioGPos[0]),int(thymioGPos[1])), 60, (0,255,0), 5)
    cv2.arrowedLine(mapWithShapes, (int(thymioGPos[0]),int(thymioGPos[1])),
                   (int(smallCircles[0][0]),int(smallCircles[0][1])), (0,255,0), 5, 8, 0, 0.4)
    cv2.circle(mapWithShapes,(int(thymioRPos[0]),int(thymioRPos[1])), 60, (255,0,255), 5)
    cv2.arrowedLine(mapWithShapes, (int(thymioRPos[0]),int(thymioRPos[1])),
                   (int(smallCircles[1][0]),int(smallCircles[1][1])), (255,0,255), 5, 8, 0, 0.4)
    
    return mapWithShapes, thymioGPos, thymioRPos, smallCircles


# 1.5 REMOVE THYMIOS AND DILATE MAP
def obstacleDetection(thresh):
    # Removing the Robots
    kernelSquare60 = np.ones((60,60),np.float32)
    obstacles = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelSquare60)

    # Applying morphological operators to the image
    morph = cv2.dilate(obstacles, None, iterations=75)
    
    return obstacles, morph


# 1.6 CREATING GRID
def getGridSize(mapBw):
    ratio = 20
    gridSize = int(len(mapBw)/ratio)
    return gridSize, ratio

def createEmptyPlot(gridSize):
    """
    Helper function to create a figure of the desired dimensions & grid
    
    :param grid_size: dimension of the map along the x and y dimensions
    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=(7,7))
    
    major_ticks = np.arange(0, gridSize+1, 5)
    minor_ticks = np.arange(0, gridSize+1, 1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([-1,gridSize])
    ax.set_xlim([-1,gridSize])
    ax.grid(True)
    
    return fig, ax

def fillGrid(gridSize, ratio, obstacles, morph, show):
    # Morphed Obstacle Data
    dataMorph = np.empty((gridSize,gridSize))
    m = 0
    n = 0
    for i in range(len(morph)):
        if (i%ratio == 0):
            for j in range(len(morph)):
                if (j%ratio == 0):
                    if (morph[i][j]>12):
                        dataMorph[m][n] = 255
                    else:
                        dataMorph[m][n] = 0
                    n += 1
                if (n == gridSize):
                    n = 0
            m += 1

    limit = 12 
    occupancyGrid = dataMorph.copy()
    occupancyGrid[dataMorph>limit] = 1
    occupancyGrid[dataMorph<=limit] = 0

    # Original Obstacle Data
    dataObstacles = np.empty((gridSize,gridSize))
    m = 0
    n = 0
    for i in range(len(obstacles)):
        if (i%ratio == 0):
            for j in range(len(obstacles)):
                if (j%ratio == 0):
                    if (obstacles[i][j]>12):
                        dataObstacles[m][n] = 255
                    else:
                        dataObstacles[m][n] = 0
                    n += 1
                if (n == gridSize):
                    n = 0
            m += 1

    limit = 12 
    obstacleGrid = dataObstacles.copy()
    obstacleGrid[dataObstacles>limit] = 1
    obstacleGrid[dataObstacles<=limit] = 0
    
    if show == 1:
        fig, ax = createEmptyPlot(gridSize)
    else:
        ax = 0
    cmap = colors.ListedColormap(['white', 'slategray'])
    
    return occupancyGrid, obstacleGrid, ax, cmap
  
    
    
def fastThimioLocator(n):
    mapImg, mapBw, eq = loadImages(takePicture(n))
    bilateral, thresh, opened = globalFilter(eq)
    tophat, circles = obstacleFilter(opened)
    mapWithShapes, thymioGPos, thymioRPos, smallCircles = thymioDetection(circles,mapImg)
    gridSize, ratio = getGridSize(mapBw)    
    thymioG = (int(thymioGPos[1]/ratio), int(thymioGPos[0]/ratio), int(thymioGPos[2]))
    thymioR = (int(thymioRPos[1]/ratio), int(thymioRPos[0]/ratio), int(thymioRPos[2]))
    return thymioG, thymioR