#!/usr/bin/env python3

import cv2
import sys
import glob
import logging
import argparse
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize


from calibrated import calibrated_sfm

from bundleadjust import *
from calibrated import calibrated_sfm
from feature_matcher import FeatureMatcher
from bundle import *
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
                 'end_header\n'.format(len(points)))
        for x, y, z, w in points:
            fd.write('{} {} {}\n'.format(x, y, z))


def estimate_initial_projection_matrices(F):
    """Estimate the projection matrices from the Fundamental Matrix
    
    A pair of camera matrices P1 and P2 corresponding to the fundamental matrix F are 
    easily computed using the direct formula in result HZ 9.14
    
    Arguments:
        F {[type]} -- [description]
    """
    e1 = fundamental_to_epipole(F)
    e2 = fundamental_to_epipole(F.T)
    P1 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0]])
    Te = skew(e2)

    P2 = np.vstack((np.dot(Te, F).T, e2)).T

    logging.info("Initial projection matrix: {}".format(P2))
    return P1, P2, e1, e2


def triangulate_points(kp1, kp2, P1, P2):
    point_cloud = []    # TODO: convert to numpy
    for i in range(kp1.shape[1]):
        point = triangulate_point(kp1[:, i], kp2[:, i], P1, P2)
        point_cloud.append(point)
    return point_cloud


def compute_projection(kp_2d, kp_3d, P):
    """ 

    :param kp_2d:
    :param kp_3d:
    :return:
    """
    A = np.zeros((2 * kp_2d.shape[0], 12))
    A[::2, :4] = kp_3d
    A[1::2, 4:8] = kp_3d
    A[::2, 8:] = -kp_2d[:, 0, np.newaxis]*kp_3d
    A[1::2, 8:] = -kp_2d[:, 1, np.newaxis]*kp_3d

    _, _, Vh = linalg.svd(A)

    P = Vh[-1, :].T.reshape((3, 4))
    P = P//P[10]
    # print(P)
    return P


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


def uncalibrated_sfm(frame_names):

    fm = FeatureMatcher()

    for frame_name in frame_names:
        fm.extract(frame_name)

    frame1 = 0
    frame2 = 1
    image1_name = frame_names[frame1]
    image2_name = frame_names[frame2]

    keypoints1, descriptors1, image1_data = fm.extract(image1_name) #, image1_data=None, draw=False)
    keypoints2, descriptors2, image2_data = fm.extract(image2_name) #, image2_data=None, draw=False)

    kp1, kp2 = fm.cross_match(image1_name, image2_name)
    kp1 = fm.normalise(kp1.T).T
    kp2 = fm.normalise(kp2.T).T

    logging.info("Keypoints matched: {}".format(kp1.shape[0]))
    kp1_homo = cv2.convertPointsToHomogeneous(kp1).reshape(kp1.shape[0], 3).T
    kp2_homo = cv2.convertPointsToHomogeneous(kp2).reshape(kp2.shape[0], 3).T

    logging.info("Estimating Fundamental Matrix from correspondences")
    F = keypoints_to_fundamental(kp1_homo, kp2_homo, optimise=True)
    
    logging.info("Estimating Projection Matrices from Fundamental Matrix")
    P1, P2, _, _ = estimate_initial_projection_matrices(F)

    logging.info("Triangulating")
    points = triangulate_points(kp1_homo, kp2_homo, P1, P2)

    # now add image 2
    image3_name = frame_names[2]
    _, descriptors3, _ = fm.extract(image3_name)
    descriptors2 = fm._descriptors[image2_name]

    # find good matches in image 2 from image3
    matches2_3 = fm.matcher.knnMatch(descriptors2, descriptors3, 2)
    good_matches2_3 = [x for x, y in matches2_3 if x.distance < 0.8*y.distance]
    # find good matches in image 3 from image2
    matches3_2 = fm.matcher.knnMatch(descriptors3, descriptors2, 2)
    good_matches3_2 = [x for x, y in matches3_2 if x.distance < 0.8*y.distance]

    # find
    k1, k2 = fm.intersect_matches(image2_name, image3_name, good_matches2_3, good_matches3_2)

    # TODO: estimate
    P = [P1,P2] 
    #runBA(P) 
    logging.info("Saving to PLY")    
    points_to_ply(points, 'uncal_{:04d}_{:04d}.ply'.format(frame1, frame2))

    logging.info("Done")

def get_args():
    parser = argparse.ArgumentParser(description='Compute fundamental matrix from image file(s)')
    parser.add_argument('--mode', type=str, help='calibrated or uncalibrated', default='uncalibrated')
    parser.add_argument('--source', type=str, help='source files', default='./fountain_int/[0-9]*.png')
    parser.add_argument('--detector', type=str, default='SIFT', help='Feature detector type')
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
        uncalibrated_sfm(image_files)
