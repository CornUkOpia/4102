import numpy as np
import cv2
import copy
from matplotlib import pyplot as plt

prefix = "./stereotest_pics/"
step = 2
pics = 4
importFlags = cv2.IMREAD_GRAYSCALE

def featuringMatchingBetweenImages(img1,img2,index):
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    print(len(matches))
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    finalTitle = "./out/" + str(i) + "-" + str(i + 1) + ".jpg"
    cv2.imwrite(finalTitle,img3)

for i in range(0,pics-1,step):
    title1Str = prefix + str(i) + ".jpg"
    img1 = cv2.imread(title1Str,importFlags)
    scale_percent = 50
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)


    # resize image
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    if importFlags == cv2.IMREAD_GRAYSCALE:
        img1 = cv2.equalizeHist(img1)
    
    img1 = cv2.GaussianBlur(img1,(5,5),0)

    title2Str = prefix + str(i + 1) + ".jpg"
    img2 = cv2.imread(title2Str,importFlags)
    scale_percent = 50
    width = int(img2.shape[1] * scale_percent / 100)
    height = int(img2.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    if importFlags == cv2.IMREAD_GRAYSCALE:
        img2 = cv2.equalizeHist(img2)
    
    img2 = cv2.GaussianBlur(img2,(5,5),0)

    # Edginess
    """
    orig1 = img1.copy()
    orig2 = img2.copy()
    img1 = cv2.Canny(img1, 100, 220);
    img2 = cv2.Canny(img2, 100, 220);
    img1 = cv2.merge([img1, img1, img1])
    img2 = cv2.merge([img2, img2, img2])
    img1 = cv2.bitwise_and(img1, orig1);
    img2 = cv2.bitwise_and(img2, orig2);
    """
    
    featuringMatchingBetweenImages(img1,img2,i)
