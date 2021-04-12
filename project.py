from PIL import Image, ImageSequence
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def featuringMatchingBetweenImages(img1,img2,index, interval):
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
 
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    finalTitle = "featureMatched/" + str(index) + "-" + str((index + interval) % len(os.listdir("outputImages"))) + ".jpg"
  
    cv2.imwrite(finalTitle,img3)
    src_pts = np.float64([ kp1[matches[m].queryIdx].pt for m in range(0, len(matches)) ]).reshape(-1,1,2)
    dst_pts = np.float64([ kp2[matches[m].trainIdx].pt for m in range(0, len(matches)) ]).reshape(-1,1,2)

    return src_pts, dst_pts

detector = cv2.ORB_create()


videoSource = Image.open("./cubeSpin.gif")
i = 0
for frame in ImageSequence.Iterator(videoSource):
    frame.save("outputImages/"+str(i)+".png",format = "png", lossless = True)
    i += 1
numImages = i

"""
inputVideo = cv2.VideoCapture("20210331_222940.mp4")
currentFrame = 0
frameInterval = 2
while(True):
    ret,frame = inputVideo.read()
    if ret:
        if (currentFrame % frameInterval) == 0:
            name = "outputImages/" + str(currentFrame) + ".png"
            cv2.imwrite(name, frame)
        currentFrame += 1
    else:
        break
numImages = currentFrame
inputVideo.release()
"""


cameraMatrix = np.asarray([[1022.5,0,273.648],[0,1004.33,618.487],[0,0,1]])

X = []
Y = []
Z = []
interval = 2
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
M_r = np.empty((3,4))
fundamentalMat = []
fundamentalMatrices = []
essentialMat = []
for i in range(0, numImages-interval, interval):
    initial1 = cv2.imread("outputImages/" + str(i) + ".png",0)
    initial2 = cv2.imread("outputImages/" + str(i+interval) + ".png",0)
    print(str(i) + " " + str(i+interval))

    initialPoints1, initialPoints2 = featuringMatchingBetweenImages(initial1,initial2,i,interval)
    initialPoints1 = np.int32(initialPoints1)
    initialPoints2 = np.int32(initialPoints2)
    initialPoints1 = np.float32(initialPoints1)
    initialPoints2 = np.float32(initialPoints2)
    if i == 0:
        fundamentalMat, mask = cv2.findFundamentalMat(initialPoints1,initialPoints2,cv2.FM_RANSAC)
        fundamentalMat = fundamentalMat[:3,:3]  # Take only the first solution.

    initialPoints1 = initialPoints1[mask.ravel() == 1]
    initialPoints2 = initialPoints2[mask.ravel() == 1]
    essentialMat = np.linalg.multi_dot([np.transpose(cameraMatrix),fundamentalMat,cameraMatrix])
    print("Fundamental and essential matrices:")
    print(fundamentalMat)
    print(essentialMat)
    print("")

    points, R, t, mask = cv2.recoverPose(essentialMat,initialPoints1,initialPoints2)
    #M_r = np.hstack((R, t))
    M_r[:3,:3] = np.matmul(R, M_l[:3,:3])
    M_r[:3,3] = M_l[:3,3] + np.matmul(M_l[:3,:3], t.ravel())
    P_l = np.matmul(cameraMatrix,  M_l)
    P_r = np.matmul(cameraMatrix,  M_r)
    M_l = np.copy(M_r)
    print("Previous and current projections:")
    print(P_l)
    print(P_r)
    print("")

    # undistort points

    initialPoints1 = np.squeeze(initialPoints1)
    initialPoints2 = np.squeeze(initialPoints2)

    #triangulate points this requires points in normalized coordinate
    print("Initial points:")
    print(initialPoints1)
    print(initialPoints2)
    point_4d_hom = cv2.triangulatePoints(P_l, P_r, initialPoints1.T, initialPoints2.T)
    print("Points, in homogeneous coordinates:")
    print(point_4d_hom)
    print("")

    point_3d = point_4d_hom / point_4d_hom[3] #np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_3d[:3, :].T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    X_temp = []
    Y_temp = []
    Z_temp = []
    for x, y, z in point_3d:
        X_temp.append(x)
        Y_temp.append(y)
        Z_temp.append(z)
        X.append(x)
        Y.append(y)
        Z.append(z)

    ax.scatter3D(X_temp, Z_temp, Y_temp, c="r", marker="o")
    #plt.show()
    fig.savefig("graphs/" + str(i) + "-" + str(i+interval) + ".png")
    plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X Label")
ax.set_ylabel("Z Label")
ax.set_zlabel("Y Label")
ax.scatter3D(X, Z, Y, c="r", marker="o")
plt.show()

"""
for i in range(0,len(os.listdir("outputImages"))):
    title1Str = "outputImages/" + str(i) + ".png"
    title2Str = "outputImages/" + str((i + 1) % len(os.listdir("outputImages"))) + ".png"
    fundamental, mask = compute_camera_matrix(cv2.imread(title1Str,0),cv2.imread(title2Str,0))  
    print(title1Str,title2Str)
    
    img1 = cv2.imread(title1Str,-1)     

    img2 = cv2.imread(title2Str,-1)

    
    img1Points, img2Points = featuringMatchingBetweenImages(img1,img2,i)
"""
