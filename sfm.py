#!/usr/bin/env python3

import cv2
import sys
import glob
import logging
import argparse
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize


from bundleadjust import *
from calibrated import calibrated_sfm
from feature_matcher import FeatureMatcher
#from bundle import *
LOG_FORMAT = '[%(asctime)s %(levelname)s %(filename)s/%(funcName)s] %(message)s'


def normalise_keypoints(kp):
    """ Apply a normalising transformation (translation and scaling)

    According to HZ p282 - normalization and saling of each image 
    so that the centroid of the reference points is at the origin of the coordinates
    and the RMS distance of the points from the origin is equal to sqrt(2)

    (this is formulated in slide 13 of 
    https://www.cs.auckland.ac.nz/courses/compsci773s1t/lectures/773-GGpdfs/773GG-FundMatrix-A.pdf)

    :param kp: ndarray of homogeneous keypoints
    :return: normalised keypoints and the translation matrix (for denomalising later)
    """
    kp = kp / kp[2]
    centre = np.mean(kp[:2], axis=1)

    # scaling
    s = np.sqrt(2) / np.std(kp[:2])
    
    # translation
    t = np.array([[s, 0, -s*centre[0]],
                  [0, s, -s*centre[1]],
                  [0, 0, 1]])
    return np.dot(t, kp), t


def fundamental_error(f, kp1, kp2):
    """ based on py3rec

    :param F:
    :param kp1:
    :param kp2:
    :return:
    """
    error = np.zeros((kp1.shape[1], 1))
    F = np.asarray([np.transpose(f[0:3]),
                    np.transpose(f[3:6]),
                    [f[6], -(-f[0] * f[4] + f[6] * f[2] * f[4] + f[3] * f[1] - f[6] * f[1] * f[5]) /
                     (-f[3] * f[2] + f[0] * f[5]), 1]])
    for i in range(kp1.shape[1]):
        error[i] = np.dot(kp2[:, i], np.dot(F, np.transpose(kp2[:, i])))
    return error.flatten()


def keypoints_to_fundamental(kp1, kp2, normalise=True, optimise=True):
    ''' compute the fundamental matrix based on 8 point algorithm

    now with normalization on the keypoints
    (see HZ Algorithm 11.1
    and/or https://www.cs.unc.edu/~marc/tutorial/node54.html)

    :param kp1: keypoints in image 1
    :param kp2: keypoints in image 2
    :param normalise: Apply normalising transform
    :param optimise: 
    :return:
    '''
    assert (kp1.shape == kp2.shape)
    if normalise:
        kp1, T1 = normalise_keypoints(kp1)
        kp2, T2 = normalise_keypoints(kp2)
    
    # as per HZ eq 11.3
    A = np.zeros((kp1.shape[1], 9))
    A[:, 0] = np.transpose(kp2[0, :]) * (kp1[0, :])
    A[:, 1] = np.transpose(kp2[0, :]) * (kp1[1, :])
    A[:, 2] = np.transpose(kp2[0, :])
    A[:, 3] = np.transpose(kp1[0, :]) * (kp1[1, :])
    A[:, 4] = np.transpose(kp2[1, :]) * (kp1[1, :])
    A[:, 5] = np.transpose(kp2[1, :])
    A[:, 6] = np.transpose(kp1[0, :])
    A[:, 7] = np.transpose(kp1[1, :])
    A[:, 8] = np.ones(kp1.shape[1])

    # compute F from the smallest singular value of A (linear solution)
    U, S, Vt = linalg.svd(A)
    V = np.transpose(Vt)
    F = V[:,8].reshape(3, 3)

    # ensure F is of rank 2 by zeroing out last singular value (constraing enforcement)
    U, S, Vt = linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))

    if normalise:
        # denormalise
        F = np.dot(T1.T, np.dot(F, T2))
        F = F/F[2,2]

    logging.info("Initial estimate of F {}".format(F))

    if optimise:
        # Optimize initial estimate using the algebraic error
        f = np.append(np.concatenate((F[0, :], F[1, :]), axis=0), F[2, 0]) / F[2, 2]

        result = optimize.least_squares(fundamental_error, f, args=(kp1, kp2))
        f = result.x

        f = np.asarray([np.transpose(f[0:3]),
                        np.transpose(f[3:6]),
                        [f[6], -(-f[0] * f[4] + f[6] * f[2] * f[4] + f[3] * f[1] - f[6] * f[1] * f[5]) /
                         (-f[3] * f[2] + f[0] * f[5]), 1]])
        F = f / np.sum(np.sum(f))*9
        logging.info("Optimised F by algebraic error {}".format(F))

    return F


def fundamental_to_epipole(F):
    ''' Compute the epipole that satisfies Fe = 0
        (Use with F.T for left epipole.)
    :param F: fundamental matrix
    :return: epipole (null space of F)
    '''
    U, S, V = linalg.svd(F)
    e = V[-1]
    return e/e[2]


def skew(e):
    """ Find the skew matrix Se

    :param e: epiople
    :return: a 3x3 skew symmetric matrix from *e*
    """
    return np.array([[0, -e[2], e[1]],
                    [e[2], 0, -e[0]],
                    [-e[1], e[0], 0]])


def compute_homography(epipole, F):
    ''' Compute homography [epiople]x[F]

    :param epipole:
    :param F:
    :return:
    '''
    H = np.dot(skew(epipole), F)
    H = H * np.sign(np.trace(H))
    return H


def triangulate_point(kp1, kp2, P1, P2):
    """ triangulate keypoints using DLT (HZ Chapter 12.2)

    :param kp1: normalised 2d feature coordinates in view 1
    :param kp2: normalised 2d feature coordinates in view 2
    :param P1: projection matrix in view 1
    :param P2: projection matrix in view 2
    :return: triangulated 3d points
    """
    # M = np.zeros((6, 6))
    # M[:3, :4] = P1
    # M[3:, :4] = P2
    # M[:3, 4] = -kp1
    # M[3:, 5] = -kp2
    # U, S, V = linalg.svd(M)
    # X = V[-1, :4]
    # return X / X[3]

    A = np.zeros((4, 4))
    A[0, :] = P1[2, :] * kp1[0] - P1[0, :]
    A[1, :] = P1[2, :] * kp1[1] - P1[1, :]
    A[2, :] = P2[2, :] * kp2[0] - P2[0, :]
    A[3, :] = P2[2, :] * kp2[1] - P2[1, :]

    _, _, Vh = linalg.svd(A)
    V = np.transpose(Vh)
    feature_3d = V[:, V.shape[0] - 1]
    feature_3d = feature_3d / feature_3d[3]
    return feature_3d


def points_to_ply(points, ply_file):
    with open(ply_file, 'w') as fd:
        fd.write('ply\nformat ascii 1.0\nelement vertex {}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                 'end_header\n'.format(len(points)))
        for point in points:
            x, y, z, w = point[0]
            b, g, r = point[1]
            fd.write('{} {} {} {} {} {}\n'.format(x, y, z, r, g, b))
"""
def points_to_obj(points, obj_file):
    with open(obj_file, 'w') as fd:
        # write the points
        for point in points:
            x, y, z, w = point[0]
            b, g, r = point[1]
            fd.write('v {} {} {}\n'.format(x, y, z))

        # write the faces
        for i in range(len(points)-2):
            # find indices of 2 nearest points
            
            fd.write('f {} {} {}\n'.format(i, n1, n2))

            
        #fd.write('ply\nformat ascii 1.0\nelement vertex {}\n'
        #         'property float x\nproperty float y\nproperty float z\n'
        #         'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        #         'end_header\n'.format(len(points)))
        #for point in points:
        #    x, y, z, w = point[0]
        #    b, g, r = point[1]
        #    fd.write('{} {} {} {} {} {}\n'.format(x, y, z, r, g, b))

def nearestNeighbours(target, arr):
    distance = (arr-point)**2).sum(axis=1)
    ndx = distance.argsort()
"""



def projective_pose_estimation(feat_2D,P,points3D):
    '''
    Method to add views using an initial 3D structure, i.e. compute the projection matrices for all the additional views (the first two are already
    estimated in previous steps)
    Args: 
            feat_2D: 2D feature coordinates for all images
            P: projection matrices
            points3d: 3D point cloud
    Returns: 
            P: projection matrices for all views
    '''
    number_of_features=feat_2D.shape[2]

    AA=np.zeros(shape=[2*number_of_features,12]);

    for i in range(2,len(feat_2D)): 
            for j in range(0,number_of_features):
                    AA[2*j,0:4]=points3D[j];
                    AA[2*j,8:12]=-feat_2D[i,0,j]*points3D[j]
                    AA[2*j+1,4:8]=points3D[j];
                    AA[2*j+1,8:12]=-feat_2D[i,1,j]*points3D[j]

            U, s, Vh = svd(AA)
            V=np.transpose(Vh)

            VV=V[0:12,11]
            VV=VV/VV[10]
            VV=np.delete(VV,10)

            #refine the estimate for the i-th projection matrix
            result=least_squares(self._eg_utils.refine_projection_matrix,VV, args=(points3D,feat_2D[i,:,:]))
            VV=result.x

            Pr=np.zeros(shape=[3,4]);
            Pr[0,:]=VV[0:4]
            Pr[1,:]=VV[4:8]
            Pr[2,:]=np.append(np.append(VV[8:10],1),VV[10])
            P[:,:,i]=Pr

    return P


def estimate_initial_projection_matrices(F):
    """Estimate the projection matrices from the Fundamental Matrix
    
    A pair of camera matrices P1 and P2 corresponding to the fundamental matrix F are 
    easily computed using the direct formula in result HZ 9.14
    
    Arguments:
        F {[type]} -- [description]
    """
    e1 = fundamental_to_epipole(F)
    e2 = fundamental_to_epipole(F.T)
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    Te = skew(e2)

    P2 = np.vstack((np.dot(Te, F).T, e2)).T

    logging.info("Initial projection matrix: {}".format(P2))
    return P1, P2, e1, e2


def triangulate_points(kp1, kp2, P1, P2, image1_data, image2_data):
    point_cloud = []
    for i in range(kp1.shape[1]):
        pointA = (kp1[:, i][0], kp1[:, i][1])
        pointB = (kp2[:, i][0], kp2[:, i][1])

        # convert pointA back to image plane coordinates
        height1, width1, depth = image1_data.shape
        mm1 = (height1 + width1)/2

        pointRGB = [int(pointA[0] * mm1 + height1), int(pointA[1] * mm1 + width1)]

        color = image1_data[pointRGB[1]][pointRGB[0]]
        point = triangulate_point(pointA, pointB, P1, P2)
        point_cloud.append([point, color])
    return point_cloud


def compute_projection(kp_2d, kp_3d):
    """ 

    :param kp_2d:
    :param kp_3d:
    :return:
    """
    kp_2d = kp_2d.T
    kp_3d = kp_3d.T
    n = kp_2d.shape[1]
    M = np.zeros((3*n, 12 + n))
    for i in range(n):
        M[3 * i, 0:4] = kp_3d[:, i]
        M[3 * i + 1, 4:8] = kp_3d[:, i]
        M[3 * i + 2, 8:12] = kp_3d[:, i]
        M[3 * i:3 * i + 3, i + 12] = -kp_2d[:, i]
    U, S, V = linalg.svd(M)
    return V[-1, :12].reshape((3, 4))


def compute_error(kp_2d, kp_3d, P):
    projections = np.dot(P, kp_3d.T).T
    projections = projections / projections[:, -1, np.newaxis]
    err = np.sum((projections[:, :-1] - kp_2d[:, :-1]) ** 2, axis=1)
    return err, np.sqrt(err) 


def decompose_projection(P):
    """ Decompose P = K[R|t] using RQ decomposition
    :param P: the projection matrix
    :return: calibration matrix, rotation, translation, camera centre
    """
    K, R = linalg.rq(P[:, :-1])
    s = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, s)
    R = np.dot(R, s)
    T = np.dot(np.linalg.inv(K), P[:, -1])
    C = np.dot(R.T, -T)
    return K, R, T, C


def uncalibrated_sfm(frame_names, detector_type, matcher_type):

    num_frames = len(frame_names)
    frame_names.sort()

    # initialise the feature matcher
    fm = FeatureMatcher(detector_type=detector_type, matcher_type=matcher_type)

    # preprocess features and descriptors
    fm.process(frame_names)

    # find features that are can be matched across all images
    fm.find_complete_tracks(frame_names)

    # find overlap in features between image pairs for projective pose estimation
    fm.find_correspondences(frame_names)

    inv_point_cloud = []  # 2d points -> 3d point

    P = [] # list of camera matrices

    for i in range(0, num_frames - 1):
        frame1 = i
        frame2 = i+1

        image1_name = frame_names[frame1]
        image2_name = frame_names[frame2]
        
        image1_data = cv2.imread(image1_name)
        image2_data = cv2.imread(image2_name)

        #keypoints1, descriptors1, _ = fm.extract(image1_name, image_data=cv2.cvtColor(image1_data, cv2.COLOR_BGR2GRAY))
        #keypoints2, descriptors2, _ = fm.extract(image2_name, image_data=cv2.cvtColor(image2_data, cv2.COLOR_BGR2GRAY))

        kp1, kp2 = fm.cross_match(image1_name, image2_name)
        norm_kp1 = fm.normalise(kp1.T.copy()).T
        norm_kp2 = fm.normalise(kp2.T.copy()).T

        logging.info("Keypoints matched: {}".format(kp1.shape[0]))
        kp1_homo = cv2.convertPointsToHomogeneous(norm_kp1).reshape(kp1.shape[0], 3).T
        kp2_homo = cv2.convertPointsToHomogeneous(norm_kp2).reshape(kp2.shape[0], 3).T

        logging.info("Estimating Fundamental Matrix from correspondences")
        F = keypoints_to_fundamental(kp1_homo, kp2_homo, optimise=True)

        if i == 0:
            logging.info("Estimating Projection Matrices from Fundamental Matrix")
            P1, P2, _, _ = estimate_initial_projection_matrices(F)
            P.append(P2)
        else:
            logging.info("Estimating Projection Matrices from Point Correspondences")

            match_2d = []
            last_kpts = {(x, y): n for n, (x, y) in enumerate(fm.matches[(frame_names[i-1], image1_name)][1])}
            for (x, y) in kp1:
                if (x, y) in last_kpts:
                    match_2d.append((x, y))

            matches = []
            for (u, v) in match_2d:
                if (u, v) in inv_point_cloud[i - 1][1]:
                    x, y, z = inv_point_cloud[i - 1][1][(u, v)]
                    matches.append(((u, v, 1), (x, y, z, 1)))

            kpts_2d = np.array([pt2 for pt2, pt3 in matches], dtype='float32')
            kpts_3d = np.array([pt3 for pt2, pt3 in matches], dtype='float32')

            P2 = compute_projection(fm.normalise(kpts_2d.T).T, kpts_3d)
            print(P2)
            P1 = P[0]
            P.append(P2)

        logging.info("Triangulating")
        points = triangulate_points(kp1_homo, kp2_homo, P1, P2, image1_data, image2_data)
        inv_point_cloud.append((dict(), dict()))
        for ((x, y, z, _), _), (u1, v1), (u2, v2) in zip(points, kp1, kp2):
            inv_point_cloud[i][0][(u1, v1)] = (x, y, z)
            inv_point_cloud[i][1][(u2, v2)] = (x, y, z)

        # points = np.asarray(points)
        # print(points)
        # points_2D = [kp1_homo, kp2_homo]
        # points_2D = np.asarray(points_2D)
        #
        # P = projective_pose_estimation(points_2D, P2, points)
        # print(P)

        points_to_ply(points, 'uncal_{:04d}_{:04d}.ply'.format(frame1, frame2))
    #runBA(P,points,points_2D) 
    #logging.info("Saving to PLY")    
    #points_to_ply(points, 'uncal_{:04d}_{:04d}.ply'.format(frame1, frame2))
    #points_to_obj(points, 'uncal_{:04d}_{:04d}.obj'.format(frame1, frame2))

    logging.info("Done")

def get_args():
    parser = argparse.ArgumentParser(description='Compute fundamental matrix from image file(s)')
    parser.add_argument('--mode', type=str, help='calibrated or uncalibrated', default='uncalibrated')
    parser.add_argument('--source', type=str, help='source files', default='./fountain_int/[0-9]*.png')
    #parser.add_argument('--source', type=str, help='source files', default='./bird_data/images/[0-9]*.ppm')
    #parser.add_argument('--source', type=str, help='source files', default='./zeno/*.jpg')
    parser.add_argument('--detector', type=str, default='SURF', help='Feature detector type')
    parser.add_argument('--matcher', type=str, default='flann', help='Matching type')
    parser.add_argument('--log_level', type=int, default=10, help='logging level (0-50)')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT) #, filename='logging.txt')

    return args


if __name__ == '__main__':
    args = get_args()

    image_files = glob.glob(args.source)

    if len(image_files) == 0:
        logging.error("No image files found")
        sys.exit(-1)

    if args.mode == 'calibrated':
        calibrated_sfm(image_files)
    else:
        uncalibrated_sfm(image_files, args.detector, args.matcher)
