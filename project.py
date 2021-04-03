import numpy as np
import cv2
import copy
from matplotlib import pyplot as plt
import os
import glob

# The phone camera used is an iPhone 12, the default focal-length is 26mm
focal_length = -26

def runImageThroughCameraCalibration(inputImg):  
   
    return cv2.filter2D(inputImg,0,K)


def featuringMatchingBetweenImages(img1,img2,index):
   
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    finalTitle = "featureMatched/" + str(i) + "-" + str((i + 1) % len(os.listdir("outputImages"))) + ".jpg"
    cv2.imwrite(finalTitle,img3)
    src_pts = np.float32([ kp1[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)

    return src_pts, dst_pts

def compute_camera_matrix(image1,image2):
    kp1, des1 = detector.detectAndCompute(image1,None)
    kp2, des2 = detector.detectAndCompute(image2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    src_pts = np.float32([ kp1[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_RANSAC)
    return  np.asanyarray(F,np.float32)

detector = cv2.ORB_create()
K = compute_camera_matrix(cv2.imread("outputImages/0.jpg",0),cv2.imread("outputImages/1.jpg",0))  
if not os.path.exists('outputImages'):
    os.makedirs('outputImages')
else:
    files = glob.glob("/outputImages")
    for f in files:
        os.remove(f)
if not os.path.exists('featureMatched'):
    os.makedirs('featureMatched')
else:
    files = glob.glob("/featureMatched")
    for f in files:
        os.remove(f)
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
for i in range(0,1):
#for i in range(0,len(os.listdir("outputImages"))):
    title1Str = "outputImages/" + str(i) + ".jpg"
    img1 = cv2.imread(title1Str,-1)     
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = runImageThroughCameraCalibration(img1)

    """scale_percent = 50
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    
    # resize image
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)"""
    title2Str = "outputImages/" + str((i + 1) % len(os.listdir("outputImages"))) + ".jpg"
    img2 = cv2.imread(title2Str,1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = runImageThroughCameraCalibration(img2)
    img1Points, img2Points = featuringMatchingBetweenImages(img1,img2,i)
    essentialMatrix, otherThingy = cv2.findEssentialMat(img1Points,img2Points,K)
    #print(essentialMatrix)
    print(np.linalg.svd(essentialMatrix))
