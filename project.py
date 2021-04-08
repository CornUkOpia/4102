import numpy as np
import cv2
import copy
import matplotlib as mpl
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
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    finalTitle = "featureMatched/" + str(i) + "-" + str((i + 1) % len(os.listdir("outputImages"))) + ".jpg"
    cv2.imwrite(finalTitle,img3)
    src_pts = np.float32([ kp1[matches[m].queryIdx].pt for m in range(0, 100) ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[matches[m].trainIdx].pt for m in range(0, 100) ]).reshape(-1,1,2)

    return src_pts, dst_pts


def compute_camera_matrix(image1,image2):
    kp1, des1 = detector.detectAndCompute(image1,None)
    kp2, des2 = detector.detectAndCompute(image2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    src_pts = np.float32([ kp1[matches[m].queryIdx].pt for m in range(0, 100) ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[matches[m].trainIdx].pt for m in range(0, 100) ]).reshape(-1,1,2)
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_RANSAC)
    return  np.asanyarray(F,np.float32), mask

detector = cv2.ORB_create()
K = [[1022.5,0,273.648],[0,1004.33,618.487],[0,0,1]]
K = np.asanyarray(K, np.float32)
distCoeffs = [[0.291851, -2.51515, -0.0582267, 0.0035612, 6.9084575]]
distCoeffs = np.asanyarray(distCoeffs, np.float32)

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
#for i in range(0,1):
    #identityMatrix = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
   # emptyMatrix = np.empty((3,4))
allPointsX = []
allPointsY = []
allPointsZ = []
R_t_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])	# FIXME: Taken from existing code, just as a test. Recreate this.
R_t_1 = np.empty((3,4))					# FIXME: Taken from existing code, just as a test. Recreate this.
ProjectionMatrix1 = np.matmul(K, R_t_0)			# FIXME: Taken from existing code, just as a test. Recreate this.
ProjectionMatrix2 = np.empty((3,4))			# FIXME: Taken from existing code, just as a test. Recreate this.
for i in range(0,len(os.listdir("outputImages"))):
    title1Str = "outputImages/" + str(i) + ".jpg"
    title2Str = "outputImages/" + str((i + 1) % len(os.listdir("outputImages"))) + ".jpg"
    F, mask = compute_camera_matrix(cv2.imread(title1Str,0),cv2.imread(title2Str,0))  
    print(title1Str,title2Str)
    
    img1 = cv2.imread(title1Str,-1)     
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
  
    """scale_percent = 50
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    
    # resize image
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)"""
    
    img2 = cv2.imread(title2Str,1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    img1Points, img2Points = featuringMatchingBetweenImages(img1,img2,i)
    # Filter the outliers.
    for j in range(len(img1Points)):
        img1Points[j] = img1Points[j] if mask[j].ravel() == 1 else 0
    for j in range(len(img2Points)):
        img2Points[j] = img2Points[j] if mask[j].ravel() == 1 else 0
    essentialMatrix, otherThingy = cv2.findEssentialMat(img1Points,img2Points,K)
    if essentialMatrix is None:
        continue;
    else:
        retval, R, t, mask = cv2.recoverPose(essentialMatrix,img1Points,img2Points,K)

        undistorted1 = cv2.undistortPoints(img1Points,K,distCoeffs=distCoeffs)
        undistorted2 = cv2.undistortPoints(img2Points,K,distCoeffs=distCoeffs)

        #ProjectionMatrix1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        #ProjectionMatrix2 = np.concatenate((np.dot(K,R),np.dot(K,t)),axis = 1)
        # FIXME: Taken from existing code, just as a test. Recreate this.
        R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
        R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3], t.ravel())
        R_t_0 = np.copy(R_t_1)
        ProjectionMatrix2 = np.matmul(K, R_t_1)
       
        triangulatedPoints = cv2.triangulatePoints(ProjectionMatrix1,ProjectionMatrix2,undistorted1,undistorted2)
        #print(triangulatedPoints[3])
        #triangulatedPoints /= triangulatedPoints[3]

        ProjectionMatrix1 = np.copy(ProjectionMatrix2)

        print("___________________________________")
        allPointsX.append(triangulatedPoints[0])
        allPointsY.append(triangulatedPoints[1])
        allPointsZ.append(triangulatedPoints[2])
        #print(triangulatedPoints.shape)
        #print(len(triangulatedPoints))
        #print(triangulatedPoints)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        #ax.set_autoscale_on(False)
        #ax.set_xlim3d([0.0, 0.5])
        #ax.set_zlim3d([0.45, 0.5])
        #ax.set_ylim3d([-0.86, 0.00005])
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.scatter3D(triangulatedPoints[0], triangulatedPoints[2], triangulatedPoints[1])
        plt.savefig("graphs/temp"+ str(i) +".png")
        plt.close()
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlim3d([np.average(allPointsX) - 2*np.std(allPointsX), np.average(allPointsX) + 2*np.std(allPointsX)])
ax.set_ylim3d([np.average(allPointsY) - 2*np.std(allPointsY), np.average(allPointsY) + 2*np.std(allPointsY)])
ax.set_zlim3d([np.average(allPointsZ) - 2*np.std(allPointsZ), np.average(allPointsZ) + 2*np.std(allPointsZ)])
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
ax.scatter3D(allPointsX, allPointsZ, allPointsY)
plt.savefig("./pointcloud.png")
plt.close()
