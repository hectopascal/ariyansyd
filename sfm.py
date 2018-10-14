#!/usr/bin/env python3

import cv2
import sys
import glob
import logging
import argparse
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize

#from calibrated_sfm import calibrated_sfm
from feature_matcher import FeatureMatcher

LOG_FORMAT = '%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s'


def normalise_keypoints(kp):
    """

    :param kp:
    :return:
    """
    kp = kp / kp[2]
    kp_mean = np.mean(kp[:2], axis=1)
    s = np.sqrt(2) / np.std(kp[:2])
    t = np.array([[s, 0, -s*kp_mean[0]], [0, s, -s * kp_mean[1]], [0, 0, 1]])
    return np.dot(t, kp)


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


def keypoints_to_fundamental(kp1, kp2, normalise=False, optimise=True):
    ''' compute the fundamental matrix based on 8 point algorithm

    :param kp1:
    :param kp2:
    :param normalise:
    :param optimise:
    :return:
    '''
    assert (kp1.shape == kp2.shape)
    if normalise:
        kp1 = normalise_keypoints(kp1)
        kp2 = normalise_keypoints(kp2)
    n = kp1.shape[1]
    A = np.zeros((n, 9))

    A[:, 0] = np.transpose(kp2[0, :]) * (kp1[0, :])
    A[:, 1] = np.transpose(kp2[0, :]) * (kp1[1, :])
    A[:, 2] = np.transpose(kp2[0, :])
    A[:, 3] = np.transpose(kp1[0, :]) * (kp1[1, :])
    A[:, 4] = np.transpose(kp2[1, :]) * (kp1[1, :])
    A[:, 5] = np.transpose(kp2[1, :])
    A[:, 6] = np.transpose(kp1[0, :])
    A[:, 7] = np.transpose(kp1[1, :])
    A[:, 8] = np.ones(n)

    # compute the linear least square solution
    U, S, Vh = linalg.svd(A)
    V = np.transpose(Vh)
    F = V[-1].reshape(3, 3)

    # ensure F is of rank 2 by zeroing out last singular value
    U, S, V = linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

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
    return F


def fundamental_to_epipole(F):
    ''' Compute the epipole that satisfies Fe = 0

    :param F: fundamental matrix
    :return: epipole (null space of F)
    '''
    U, S, V = linalg.svd(F)
    e = V[-1]
    return e/e[2]


def skew(e):
    """ Find the skew matrix Se

    :param e: epiople
    :return: the skew matrix
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


def reference_plane_error(ref_plane, H, e):
    """ based on pyrec3d

    :param ref_plane:
    :param H:
    :param e:
    :return:
    """
    epi = np.reshape(e, (3, 1))
    ref_plane = np.reshape(ref_plane, (1, 4))
    t = np.reshape(ref_plane[0,0:3], (1, 3))
    return np.sum(np.sum(np.abs(H + epi.dot(t) - ref_plane[0,3]*np.eye(3))))


def estimate_projection(H, e2, ref_plane):
    """ Estimate projection matrices for two views

    based on pyrec3d

    P1=[I | 0], P2=[H+epi1|e])

    :param H: homography [e]x[F]
    :param e2: epipole in 2nd view
    :param ref_plane: the reference plane at infinity
    :return: P1, P2
    """
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    P2 = np.zeros((3, 4))
    P2[:, :3] = H + np.dot(np.reshape(e2, (3, 1)), np.reshape(ref_plane, (1, 3)))
    P2[:, 3] = e2
    P2 = P2/P2[2, 2]

    return P1, P2


def triangulate_point(kp1, kp2, P1, P2):
    """ from py3drec

    :param kp1: normalised 2d feature coordinates in view 1
    :param kp2: normalised 2d feature coordinates in view 2
    :param P1: projection matrix in view 1
    :param P2: projection matrix in view 2
    :return: triangulated 3d points
    """
    A = np.zeros((4, 4))
    A[0, :] = P1[2, :] * kp1[0] - P1[0, :]
    A[1, :] = P1[2, :] * kp1[1] - P1[1, :]
    A[2, :] = P2[2, :] * kp2[0] - P2[0, :]
    A[3, :] = P2[2, :] * kp2[1] - P2[1, :]

    U, s, Vh = linalg.svd(A)
    V = np.transpose(Vh)
    feature_3d = V[:, V.shape[0] - 1]
    feature_3d = feature_3d / feature_3d[3]
    return feature_3d


def uncalibrated_sfm(frame_names):

    fm = FeatureMatcher()

    image1_name = frame_names[0]
    image2_name = frame_names[1]

    kp1, kp2 = fm.cross_match(image1_name, image2_name, normalise=True)
    kp1_homo = cv2.convertPointsToHomogeneous(kp1).reshape(kp1.shape[0], 3).T
    kp2_homo = cv2.convertPointsToHomogeneous(kp2).reshape(kp2.shape[0], 3).T

    num_keypoints = kp1.shape[0]
    logging.info("Keypoints matched: {}".format(num_keypoints))

    logging.info("Computing Fundamental Matrix")
    F = keypoints_to_fundamental(kp1_homo, kp2_homo)
    e1 = fundamental_to_epipole(F)
    e2 = fundamental_to_epipole(F.T)

    logging.info("Initialising projective reference plane")
    H = compute_homography(e2, F)
    reference_plane = np.sum(np.divide(np.eye(3) - H, np.transpose(np.asarray([e2, e2, e2]))), axis=0)/3

    logging.info("Optimising reference plane")
    reference_plane = optimize.fmin(reference_plane_error,
                                    np.append(reference_plane, 1),
                                    xtol=1e-25,
                                    ftol=1e-25,
                                    args=(H.real, e2.real))[0:3]

    # now estimate P2
    P1, P2 = estimate_projection(H, e2, reference_plane)
    logging.info("Initial projection matrix: {}".format(P2))

    logging.info("Generating initial point cloud")
    point_cloud = []
    for i in range(num_keypoints):
        point = triangulate_point(kp1_homo[:, i], kp2_homo[:, i], P1, P2)
        point_cloud.append(point)


def get_args():
    parser = argparse.ArgumentParser(description='Compute fundamental matrix from image file(s)')
    parser.add_argument('--mode', type=str, help='calibrated or uncalibrated', default='uncalibrated')
    parser.add_argument('--source', type=str, help='source files', default='./data/fountain_int/[0-9]*.png')
    parser.add_argument('--detector', type=str, default='SIFT', help='Feature detector type')
    parser.add_argument('--matcher', type=str, default='flann', help='Matching type')
    parser.add_argument('--log_level', type=int, default=10, help='logging level (0-50)')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, filename='logging.txt')

    return args


if __name__ == '__main__':
    args = get_args()

    image_files = glob.glob(args.source)

    if len(image_files) == 0:
        logging.error("No image files found")
        sys.exit(-1)

    if args.mode == 'calibrated':
        raise NotImplementedError()
        #calibrated_sfm(image_files)
    else:
        uncalibrated_sfm(image_files)
