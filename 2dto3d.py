#!/usr/lib/python2.7/
import cv2
import numpy as np



#cv2.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2[, points4D]) --> returns points4D
#Parameters:	
#projMatr1 – 3x4 projection matrix of the first camera.
#projMatr2 – 3x4 projection matrix of the second camera.
#projPoints1 – 2xN array of feature points in the first image. In case of c++ version it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
#projPoints2 – 2xN array of corresponding points in the second image. In case of c++ version it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
#points4D – 4xN array of reconstructed points in homogeneous coordinates.


# Input: matrices of 2 sets points (pts1 & pts2),
#       intrinsic(K1,K2) & extrinsic(Rt1,Rt2) camera matrices
#       assuming using same camera setting for everything 
#       so K is the same for both set of points here
# Output: 4*N array of points (3d world coords?)
def worldcoord(pts1,pts2,K1,K2,Rt1,Rt2):
    cam1 = np.matmul(K,Rt1)
    cam2 = np.matmul(K,Rt2)
    return cv2.triangulatePoints(cam1,cam2,pts1,pts2)

# Get Rt matrix from essential matrix
# Input: Essential matrix
# Output: Rt
def calculateRt():
    
