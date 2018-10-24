import cv2
import logging
import numpy as np
import scipy.linalg as linalg
from collections import namedtuple
# from collections import defaultdict

from feature_matcher import FeatureMatcher

Point3d = namedtuple('Point3d', ['coordinates', 'origin'])


class SingleCameraReconstruction(object):

    def __init__(self, K):
        self.intrinsic = K
        self.projections = [np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0]], dtype=float)]
        self.fm = FeatureMatcher()
        self.point_cloud = dict()               # 3d point -> 2d points
        # self.point_tracks = defaultdict(set)    # 2d points -> image tracks
        self.inv_point_cloud = dict()        # 2d points -> 3d point
        # self.image_tracks = defaultdict(set)    # image tracks -> 2d points
        self.keypoints = []                     # pairs of keypoints that were matched in an image pair

    def _tohomogeneous(self, keypoints):
        return np.vstack((keypoints.T, np.ones(keypoints.shape[0])))

    def _fromhomogeneous(self, homo_keypoints):
        return homo_keypoints[:-1].T

    def _normalise(self, keypoints):
        """ Normalise keypoints

        Normalise by multiplying with K.inv

        :param keypoints: keypoints
        :return: normalised keypoints
        """
        # convert to homogeneous so its just a dot product
        homo_kpts = self._tohomogeneous(keypoints)
        norm_kpts = np.dot(np.linalg.inv(self.intrinsic), homo_kpts)
        return self._fromhomogeneous(norm_kpts)

    def _compute_essential(self, keypoints1, keypoints2):
        """ Compute F and E using K

        :param keypoints1: 2d points (2 x n)
        :param keypoints2: 2d points (2 x n)
        :return: F, F_mask, E, kp1_inlier, kp2_inlier
        """
        # normalise and find F using OpenCV
        fundamental, fundamental_mask = cv2.findFundamentalMat(keypoints1, keypoints2, cv2.RANSAC, 0.1, 0.99)

        # E = K.T*F*K and ensure rank 2
        essential = np.dot(self.intrinsic.T, np.dot(fundamental, self.intrinsic))
        U, S, V = np.linalg.svd(essential)
        if linalg.det(np.dot(U, V)) < 0:
            V = -V
        essential = np.dot(U, np.dot(np.diag([1, 1, 0]), V))

        return fundamental, fundamental_mask, essential


    @staticmethod
    def in_front_of_both_cameras(keypoints1, keypoints2, rotation, translation):
        """ Check whether point correspondences are in front of both images

        """
        for kpt1, kpt2 in zip(keypoints1, keypoints2):
            first_z = np.dot(rotation[0, :] - kpt2[0] * rotation[2, :], translation) / \
                      np.dot(rotation[0, :] - kpt2[0] * rotation[2, :], kpt2)
            first_3d_point = np.array([kpt1[0] * first_z, kpt2[0] * first_z, first_z])
            second_3d_point = np.dot(rotation.T, first_3d_point) - np.dot(rotation.T, translation)
            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

    def _compute_projections(self, essential):
        """ Estimate the projection matrices P1 and P2 from a set of keypoints and E

        :param essential: essential matrix
        :return:
        """
        P1 = self.projections[-1]

        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        U, _, V = linalg.svd(essential)
        if linalg.det(np.dot(U, V)) < 0:
            V = -V

        # there are 4 possible solutions - find the one that predicts all imaged keypoints in front of both cameras
        P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
               np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
               np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
               np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

        # P2 = 0
        # max_results = 0
        # for P2_ in P2s:
        #     # triangulate inliers and compute depth for each camera
        #     homog_3D = cv2.triangulatePoints(P1, P2_, keypoints1, keypoints2)
        #
        #     # the sign of the depth is the 3rd value of the image point after projecting back to the image
        #     d1 = np.dot(P1, homog_3D)[2]
        #     d2 = np.dot(P2, homog_3D)[2]
        #
        #     if sum(d1 > 0) + sum(d2 < 0) > max_results:
        #         max_results = sum(d1 > 0) + sum(d2 < 0)
        #         P2 = P2_
        #         #homo_infront = homog_3D[:, (d1 > 0) & (d2 < 0)]

        # keypoints1 = self._tohomogeneous(keypoints1).T
        # keypoints2 = self._tohomogeneous(keypoints2).T
        # R, t = np.dot(U, np.dot(W, V)).T, U[:, 2]
        # if not SingleCameraReconstruction.in_front_of_both_cameras(keypoints1, keypoints2, R, t):
        #     R, t = np.dot(U, np.dot(W, V)).T, -U[:, 2]
        #     if not SingleCameraReconstruction.in_front_of_both_cameras(keypoints1, keypoints2, R, t):
        #         R, t = np.dot(U, np.dot(W.T, V)).T, U[:, 2]
        #         if not SingleCameraReconstruction.in_front_of_both_cameras(keypoints1, keypoints2, R, t):
        #             R, t = np.dot(U, np.dot(W.T, V)).T, -U[:, 2]

        return P1, P2s

    def _compute_next_projections(self, E, keypoints1, seq_num):
        """ Use PnP RANSAC to find projection matrices P1 and P2

        :param E: essential matrix
        :param keypoints1: keypoints in view 1
        :param keypoints2: keypoits in view 2
        :param seq_num: the image number in the image sequence
        :return: P1, P2
        """
        # find 2d and 3d correspondences
        logging.debug("Finding 2D and 3D correspondences")

        # firstly find keypoints that are the intersection of keypoints1 and 'keypoints2' from the previous triangulation
        match_2d = []
        last_kpts = {(x,y): n for n, (x,y) in enumerate(self.keypoints[seq_num-1][1])}
        for num, (x,y) in enumerate(keypoints1):
            if (x,y) in last_kpts:
                match_2d.append(np.array((x,y)))
        
        # then find what 3d point that 2d point corresponds to
        matches = []
        for (u, v) in match_2d:
            if (u, v) in self.inv_point_cloud[seq_num-1][1]:
                matches.append(((u, v), self.inv_point_cloud[seq_num-1][1][(u,v)]))

        kpts_2d = np.array([pt2 for pt2, pt3 in matches], dtype = 'float32')

        kpts_3d = np.array([pt3 for pt2, pt3 in matches], dtype='float32')

        logging.debug("Solving by PnP RANSAC")
        ret, R, t, inliers = cv2.solvePnPRansac(kpts_3d, kpts_2d, self.intrinsic, None)

        logging.debug("Computing pose")
        Rmat, _ = cv2.Rodrigues(R)
        pose = np.hstack((Rmat, t))
        self.projections.append(pose)
        return self._compute_projections(E)

    def _triangulate_points(self, kp1, kp2, P1, P2s):
        kp1 = kp1.T
        kp2 = kp2.T

        P2 = None
        max_results = 0
        for P2_ in P2s:
            # triangulate inliers and compute depth for each camera
            homog_3D = cv2.triangulatePoints(P1, P2_, kp1, kp2)

            # the sign of the depth is the 3rd value of the image point after projecting back to the image
            d1 = np.dot(P1, homog_3D)[2]
            d2 = np.dot(P2_, homog_3D)[2]

            if sum(d1 > 0) + sum(d2 < 0) > max_results:
                max_results = sum(d1 > 0) + sum(d2 < 0)
                P2 = P2_
                infront = (d1 > 0) & (d2 < 0)

        homo_pts3d = cv2.triangulatePoints(P1, P2, kp1, kp2)
        #homo_pts3d = homo_pts3d[:, infront]
        homo_pts3d = homo_pts3d/homo_pts3d[3]
        return homo_pts3d, np.array(homo_pts3d[:3]).T, infront, P2

    def _track_points(self, kp_3d, kp1, kp2, seq_num, image1, image2):
        """ Add new 2d/3d correspondences

        :param kp_3d:
        :param kp1:
        :param kp2:
        :param seq_num:
        :param image1:
        :param image2:
        :return:
        """
        new_point_cloud = {(x, y, z): image1[int(v), int(u), ] for (u, v), (x, y, z) in zip(kp1, kp_3d) }
        self.point_cloud[seq_num] = new_point_cloud

        self.inv_point_cloud[seq_num] = (dict(), dict())
        for (x, y, z), (u1, v1), (u2, v2) in zip(kp_3d, kp1, kp2):
            self.inv_point_cloud[seq_num][0][(u1, v1)] = (x, y, z)
            self.inv_point_cloud[seq_num][1][(u2, v2)] = (x, y, z)

    def save_point_cloud(self, i, file_name='./calibrated.ply'):
        points = self.point_cloud[i].items()
        with open(file_name, 'w') as fd:
            fd.write('ply\nformat ascii 1.0\nelement vertex {}\n'
                     'property float x\nproperty float y\nproperty float z\n'
                     'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                     'end_header\n'.format(len(points)))
            for point in points:
                x, y, z = point[0]
                b, g, r = point[1]
                fd.write('{} {} {} {} {} {}\n'.format(x, y, z, r, g, b))
        pass

    def refine_points(self, norm_pts1, norm_pts2, E):
        '''Refine the coordinates of the corresponding points using the Optimal Triangulation Method.'''
        # convert to 1xNx2 arrays for cv2.correctMatches
        refined_pts1 = np.array([[pt for pt in norm_pts1]])
        refined_pts2 = np.array([[pt for pt in norm_pts2]])
        refined_pts1, refined_pts2 = cv2.correctMatches(E, refined_pts1, refined_pts2)

        # refined_pts are 1xNx2 arrays
        return refined_pts1, refined_pts2

    def triangulate_points(self, P1, P2, refined_pts1, refined_pts2):
        '''Reconstructs 3D points by triangulation using Direct Linear Transformation.'''
        # convert to 2xN arrays
        refined_pts1 = refined_pts1[0].T
        refined_pts2 = refined_pts2[0].T

        # pick the P2 matrix with the most scene points in front of the cameras after triangulation
        ind = 0
        maxres = 0

        for i in range(4):
            # triangulate inliers and compute depth for each camera
            homog_3D = cv2.triangulatePoints(P1, P2[i], refined_pts1, refined_pts2)
            # the sign of the depth is the 3rd value of the image point after projecting back to the image
            d1 = np.dot(P1, homog_3D)[2]
            d2 = np.dot(P2[i], homog_3D)[2]

            if sum(d1 > 0) + sum(d2 < 0) > maxres:
                maxres = sum(d1 > 0) + sum(d2 < 0)
                ind = i
                infront = (d1 > 0) & (d2 < 0)

        # triangulate inliers and keep only points that are in front of both cameras
        # homog_3D is a 4xN array of reconstructed points in homogeneous coordinates, pts_3D is a Nx3 array
        homog_3D = cv2.triangulatePoints(P1, P2[ind], refined_pts1, refined_pts2)
        #homog_3D = homog_3D[:, infront]
        homog_3D = homog_3D / homog_3D[3]
        pts_3D = np.array(homog_3D[:3]).T

        return homog_3D, pts_3D, infront, P2[ind]

    def init_structure(self, image1_name, image2_name):
        logging.info("Loading images {} and {}".format(image1_name, image2_name))
        image1 = cv2.cvtColor(cv2.imread(image1_name), cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(cv2.imread(image2_name), cv2.COLOR_BGR2RGB)

        image1 = cv2.undistort(image1, self.intrinsic, None)
        image2 = cv2.undistort(image2, self.intrinsic, None)

        logging.info("Matching keypoints")
        keypoints1, keypoints2 = self.fm.cross_match(image1_name, image2_name,
                                                     image1_data=cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY),
                                                     image2_data=cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY))
        logging.info("Found {} matches".format(len(keypoints1)))
        self.keypoints.append((keypoints1, keypoints2))
        norm_keypoints1, norm_keypoints2 = self._normalise(keypoints1), self._normalise(keypoints2)

        logging.info("Computing Fundamental and Essential matrices from matched keypoints")
        F, F_mask, E, = self._compute_essential(norm_keypoints1, norm_keypoints2)
        logging.info("Fundamental Matrix: {}".format(list(F)))
        logging.info("Essential Matrix: {}".format(list(E)))

        logging.info("Computing Camera Projections")
        P1, P2s = self._compute_projections(E)
        logging.info("P1: {}".format(list(P1)))
        logging.info("P2 1: {}".format(list(P2s[0])))
        logging.info("P2 2: {}".format(list(P2s[1])))
        logging.info("P2 3: {}".format(list(P2s[2])))
        logging.info("P2 4: {}".format(list(P2s[3])))

        logging.info("Refining keypoints based on Fundamental/Essential Matrices")
        keypoints1 = keypoints1[F_mask.ravel() == 1]
        keypoints2 = keypoints2[F_mask.ravel() == 1]
        norm_keypoints1 = norm_keypoints1[F_mask.ravel() == 1]
        norm_keypoints2 = norm_keypoints2[F_mask.ravel() == 1]
        ref_keypoints1, ref_keypoints2 = self.refine_points(norm_keypoints1, norm_keypoints2, E) # refine points

        logging.info("Triangulating 3D sparse points")
        _, keypoints3d, _, P2 = self.triangulate_points(P1, P2s, ref_keypoints1, ref_keypoints2)
        logging.debug("{} 3D points".format(len(keypoints3d)))
        self.projections.append(P2)

        self._track_points(keypoints3d, keypoints1, keypoints2, 0, image1, image2)

    def add_image_pair(self, i, image1_name, image2_name):
        logging.info("Loading images {} and {}".format(image1_name, image2_name))
        image1 = cv2.cvtColor(cv2.imread(image1_name), cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(cv2.imread(image2_name), cv2.COLOR_BGR2RGB)

        logging.info("Matching keypoints")
        keypoints1, keypoints2 = self.fm.cross_match(image1_name, image2_name,
                                                     image1_data=cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY),
                                                     image2_data=cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY))
        logging.info("Found {} matches".format(len(keypoints1)))
        self.keypoints.append((keypoints1, keypoints2))
        norm_keypoints1, norm_keypoints2 = self._normalise(keypoints1), self._normalise(keypoints2)

        logging.info("Computing Fundamental and Essential matrices from matched keypoints")
        F, F_mask, E, = self._compute_essential(norm_keypoints1, norm_keypoints2)
        logging.info("Fundamental Matrix: {}".format(list(F)))
        logging.info("Essential Matrix: {}".format(list(E)))

        logging.info("Finding correspondences to get projection matrices")
        P1, P2s = self._compute_next_projections(E, keypoints1, i)
        logging.debug("P1: {}".format(list(P1)))
        logging.debug("P2 1: {}".format(list(P2s[0])))
        logging.debug("P2 2: {}".format(list(P2s[1])))
        logging.debug("P2 3: {}".format(list(P2s[2])))
        logging.debug("P2 4: {}".format(list(P2s[3])))

        keypoints1 = keypoints1[F_mask.ravel() == 1]  # filter_keypoints
        keypoints2 = keypoints2[F_mask.ravel() == 1]
        norm_keypoints1 = norm_keypoints1[F_mask.ravel() == 1]  # apply_mask
        norm_keypoints2 = norm_keypoints2[F_mask.ravel() == 1]
        ref_keypoints1, ref_keypoints2 = self.refine_points(norm_keypoints1, norm_keypoints2, E)  # refine points

        logging.info("Triangulating 3D sparse points")
        _, keypoints3d, _, P2 = self.triangulate_points(P1, P2s, ref_keypoints1, ref_keypoints2)
        logging.debug("{} 3D points".format(len(keypoints3d)))
        logging.info("P2: {}".format(list(P2)))
        self.projections.append(P2)

        self._track_points(keypoints3d, keypoints1, keypoints2, i, image1, image2)


def calibrated_sfm(image_files):

    K = np.ndfromtxt('{}.K'.format(image_files[0]))
    recon = SingleCameraReconstruction(K)

    recon.init_structure(image_files[0], image_files[1])
    recon.save_point_cloud(0, file_name='./calibrated_{:04d}_{:04d}.ply'.format(0, 1))
    for i in range(1, len(image_files) - 1):
        recon.add_image_pair(i, image_files[i], image_files[i + 1])
        recon.save_point_cloud(i, file_name='./calibrated_{:04d}_{:04d}.ply'.format(i, i+1))
    #recon.save_point_cloud(None, file_name='./calibrated.ply')