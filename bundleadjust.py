#!/usr/bin/env python
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals. 
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def reprojection_residuals(pts3d,proj,pts2d):
    est2d = np.dot(proj,np.transpose(pts3d))
    #normalize points since pts2d is normalized
    est2d = pts2d/pts2d[2,:]
    # squared error
    res = sum(sum(np.power(pts2d-est2d[0:2,:],2)))
    return res

def readPoints(file_name):
    with open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d

def runBA(P):

    camera_params, points_3d, camera_indices, point_indices, points_2d = readPoints('BApoints.txt')

    ### PRINTING SOME STATS ###
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0] #number_of_features

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    ### PRINTING SOME STATS ###
   
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    print(camera_params.ravel())
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
   
    plt.plot(f0)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)


#    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
#                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    #print(res)
#    result=least_squares(fun,x0, max_nfev=1000,method='trf',args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    result = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    X=result.x
    print(result)
    feat_2D = [n_cameras, n_points, camera_indices, point_indices, points_2d]
    error=np.power(sum(f0),2)
    print("errrrrrrrrrrrrrrrrrr")
    print(P)
    print(X)
    #get the refined projection matrices from the optimal vector
    for i in range(0,3):
            P[:,:,i]=np.reshape(X[0+i*11:12+i*11],(3,4));
            P[2,3,i]=P[2,2,i]
            P[2,2,i]=1
    
    #get the refined 3D coordinates from the optimal vector
    feat3D =  numpy.zeros((n_points,4))
    feat3D[:,0:3]=np.reshape(X[self._sequence_length*11:self._sequence_length*11+self._sequence_length*number_of_features*3],(number_of_features,3))

    Tp1= np.vstack([P[:,:,0],[0,0,0,1]]);
    for i in range(0,self._sequence_length):
            P[:,:,i]=P[:,:,i].dot(inv(Tp1))

    feat3D=Tp1.dot(np.transpose(feat3D))
    feat3D=np.transpose(feat3D/feat3D[3,:]);

    return P,feat3D,error

# Input: List of 2D points
#       List of corresponding camera 2D matrices
#       3d point model#
#
#def bundle_adjustment():
# 
#    '''
#    Method to refine structure and motion, i.e. refine the projection matrices and 3D points using the reprojection error
#    Args: 
#            feat_2D: 2D feature coordinates for all images
#            P: projection matrices
#            points3d: 3D point cloud
#    Returns: 
#            P: the refined projection matrices
#            feat3D: the refined 3D point cloud
#            error: the reprojection error
#    '''
#    camera_params, points_3d, camera_indices, point_indices, points_2d = readPoints('BApoints.txt')
#   
#
#    ### PRINTING SOME STATS ###
#    n_cameras = camera_params.shape[0]
#    n_points = points_3d.shape[0] #number_of_features
#
#    n = 9 * n_cameras + 3 * n_points
#    m = 2 * points_2d.shape[0]
#
#    print("n_cameras: {}".format(n_cameras))
#    print("n_points: {}".format(n_points))
#    print("Total number of parameters: {}".format(n))
#    print("Total number of residuals: {}".format(m))
#    ### PRINTING SOME STATS ###
#
#    ################TODO EDIT BELOW ############################
#    #The vector to be optimized 
#    X=np.reshape(P[:,:,0],(1,12));
#
#    # Append the projection matrices...
#    for i in range(1,self._sequence_length):
#        X=np.append(X,np.reshape(P[:,:,i],(1,12)))
#    X=np.delete(X,[10,22,(self._sequence_length-1)*12+10])
#
#    # ...and then append the 3D points
#    X=np.append(X,np.reshape(feat3D[:,0:3],number_of_features*self._sequence_length))
#
#    # Levenberg-Marquardt Bundle Adjustment (optimizer) from scipy

    #result=least_squares(self._eg_utils.overall_reprojection_error,X, max_nfev=1000,method='lm',args=([feat_2D]))
#    result=least_squares(residuals,X, max_nfev=1000,method='lm',args=([feat_2D]))
#    X=result.x
#
#    error=np.power(sum(reprojection_residuals(X,feat_2D)),2)
#
#    #get the refined projection matrices from the optimal vector
#    for i in range(0,self._sequence_length):
#            P[:,:,i]=np.reshape(X[0+i*11:12+i*11],(3,4));
#            P[2,3,i]=P[2,2,i]
#            P[2,2,i]=1
#
#    #get the refined 3D coordinates from the optimal vector
#    feat3D[:,0:3]=np.reshape(X[self._sequence_length*11:self._sequence_length*11+self._sequence_length*number_of_features*3],(number_of_features,3))
#
#    Tp1= np.vstack([P[:,:,0],[0,0,0,1]]);
#    for i in range(0,self._sequence_length):
#            P[:,:,i]=P[:,:,i].dot(inv(Tp1))
#
#    feat3D=Tp1.dot(np.transpose(feat3D))
#    feat3D=np.transpose(feat3D/feat3D[3,:]);
#
#    return P,feat3D,error
