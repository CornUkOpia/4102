import numpy as np
import cv2
import copy
from matplotlib import pyplot as plt
import os

# The phone camera used is an iPhone 12, the default focal-length is 26mm
focal_length = -26

def runImageThroughCameraCalibration(inputImg):  
    Sy = imgL.shape[0]
    Sx = imgL.shape[1]
    K = np.array([[(focal_length/Sy),0,0],[0,(focal_length/Sx),0],[0,0,1]])
    return cv2.filter2D(inputImg,-1,K)


def featuringMatchingBetweenImages(img1,img2,index):
    detector = cv2.ORB_create()
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    finalTitle = "featureMatched/" + str(i) + "-" + str((i + 1) % len(os.listdir("outputImages"))) + ".jpg"
    cv2.imwrite(finalTitle,img3)


if not os.path.exists('outputImages'):
    os.makedirs('outputImages')
if not os.path.exists('featureMatched'):
    os.makedirs('featureMatched')
inputVideo = cv2.VideoCapture("20210331_222940.mp4")
currentFrame = 0
frameInterval = 5
while(True):
    ret,frame = inputVideo.read()
    if ret:
        if (currentFrame % frameInterval) == 0:
            name = "outputImages/" + str(int(currentFrame/frameInterval)) + ".jpg"
            cv2.imwrite(name, frame)
        currentFrame += 1
    else:
        break
inputVideo.release()

cv2.destroyAllWindows()


for i in range(0,len(os.listdir("outputImages"))):
    title1Str = "outputImages/" + str(i) + ".jpg"
    img1 = cv2.imread(title1Str,1)     
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    """scale_percent = 50
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)

    
    # resize image
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)"""
    title2Str = "outputImages/" + str((i + 1) % len(os.listdir("outputImages"))) + ".jpg"
    img2 = cv2.imread(title2Str,1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    featuringMatchingBetweenImages(img1,img2,i)
