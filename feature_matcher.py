import cv2
import logging
import glob
import os.path
import pickle
import copyreg
import argparse
import numpy as np

try:
    import progressbar
    use_progress = True
except ImportError:
    use_progress = False

LOG_FORMAT = '%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s'


def get_args():
    parser = argparse.ArgumentParser(description='Extract features and descriptors from image file(s)')
    parser.add_argument('source', type=str, help='source folder')
    parser.add_argument('--detector', type=str, default='SURF', help='Feature detector type')
    parser.add_argument('--matcher', type=str, default='flann', help='Matching type')
    parser.add_argument('--log_level', type=int, default=10, help='logging level (0-50)')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, filename='logging.txt')

    return args

def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)


class FeatureMatcher(object):

    def __init__(self, detector_type='SURF', matcher_type='flann'):
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        self.detector = self._get_detector(detector_type)
        self.matcher  = self._get_matcher(matcher_type)
        self._features = dict()
        self._descriptors = dict()

    def _get_detector(self, detector='SURF'):
        logging.debug("detector {}".format(detector))
        if detector == 'SIFT':
            return cv2.xfeatures2d.SIFT_create()
        elif detector == 'SURF':
            return cv2.xfeatures2d.SURF_create()
        # elif detector =='FAST':
        #     return cv2.FastFeatureDetector_create()
        # elif detector == 'ORB':
        #     return cv2.ORB_create()
        # elif detector == 'BRISK':
        #     return cv2.BRISK_create()
        # elif detector == 'KAZE':
        #     return cv2.KAZE_create()
        # elif detector == 'AKAZE':
        #     return cv2.AKAZE_create()
        else:
            raise ValueError("Invalid detector type {}".format(detector))

    def _get_matcher(self, matcher='flann'):
        logging.debug("matcher {}".format(matcher))
        if matcher == 'brute':
            bf_params = cv2.NORM_L1
            # update the default parameters
            if self.detector_type in {'ORB'}:
                bf_params = cv2.NORM_HAMMING
            return cv2.BFMatcher(bf_params)
        elif matcher == 'flann':
            index_params = {'algorithm': 1, 'trees': 5}
            # index_params = dict(algorithm=6,
            #                 table_number=6,  # 12
            #                 key_size=12,  # 20
            #                 multi_probe_level=1)
            return cv2.FlannBasedMatcher(index_params, {'checks': 50})
        else:
            raise ValueError("Invalid matching type {}".format(matcher))

    def extract(self, image_name, image_data=None, draw=False):
        if not image_data:
                image_data = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        if image_name not in self._features:
            feature_file = '{}.{}.feat'.format(image_name, self.detector_type)
            descriptor_file = '{}.{}.desc'.format(image_name, self.detector_type)

            if os.path.exists(feature_file):
                logging.info("Loading features from {}".format(feature_file))
                try:
                    with open(feature_file, 'rb') as fd:
                        self._features[image_name] = pickle.load(fd)
                except:
                    logging.error("Error loading features from {}".format(feature_file))

            if self._features.get(image_name) is None:
                self._features[image_name] = self.detector.detect(image_data)
                logging.info("Extracted {} features from {} ".format(len(self._features[image_name]), image_name))
                with open(feature_file, 'wb') as fd:
                    pickle.dump(self._features[image_name], fd)
                logging.info("Saved features to {} ".format(feature_file))

            if os.path.exists(descriptor_file):
                logging.info("Loading descriptors from {}".format(descriptor_file))
                try:
                    with open(descriptor_file, 'rb') as fd:
                        self._descriptors[image_name] = pickle.load(fd)
                except:
                    logging.error("Error loading descriptors from {}".format(descriptor_file))

            if self._descriptors.get(image_name) is None:
                logging.info("Computing descriptors from {}".format(image_name))
                self._descriptors[image_name] = self.detector.compute(image_data, self._features[image_name])
                if isinstance(self._descriptors[image_name], tuple):
                    self._descriptors[image_name] = self._descriptors[image_name][1]    # SURF compute() returns a tuple of (keypoints, descriptors)
                with open(descriptor_file, 'wb') as fd:
                    pickle.dump(self._descriptors[image_name], fd)
                logging.info("Saved descriptors to {}".format(descriptor_file))

        if draw:
            draw_file = '{}.{}.keypoints.jpg'.format(image_name, self.detector_type)
            if not os.path.exists(draw_file):
                draw_image = cv2.drawKeypoints(image_data, self._features[image_name], None)
                cv2.imwrite(draw_file, draw_image)

        return self._features[image_name], self._descriptors[image_name], image_data
        
    def cross_match(self, image1_name, image2_name, image1_data=None, image2_data=None, good_ratio=0.8, normalise=False, draw=False):
        keypoints1, descriptors1, image1_data = self.extract(image1_name, image1_data, draw=draw)
        keypoints2, descriptors2, image2_data = self.extract(image2_name, image2_data, draw=draw)

        # find good matches in image 1
        matches1 = self.matcher.knnMatch(descriptors1, descriptors2, 2)
        logging.info("Matching {} > {}: {}".format(image1_name, image2_name, len(matches1)))
        good_matches1 = [x for x, y in matches1 if x.distance < good_ratio*y.distance]

        # find good matches in image 2
        matches2 = self.matcher.knnMatch(descriptors2, descriptors1, 2)
        logging.info("Matching {} < {}: {}".format(image1_name, image2_name, len(matches2)))
        good_matches2 = [x for x, y in matches2 if x.distance < good_ratio*y.distance]

        # cross match to ensure they are very good
        kp1 = set([(keypoints1[p.queryIdx].pt, keypoints2[p.trainIdx].pt) for p in good_matches1])
        kp2 = set([(keypoints1[p.trainIdx].pt, keypoints2[p.queryIdx].pt) for p in good_matches2])

        very_good_matches = kp1.intersection(kp2)

        very_good_keypoints_1 = np.zeros((len(very_good_matches), 2), dtype=np.float)
        very_good_keypoints_2 = np.zeros((len(very_good_matches), 2), dtype=np.float)

        height1, width1 = image1_data.shape
        mm1 = (height1 + width1)/2
        height2, width2 = image2_data.shape
        mm2 = (height2 + width2)/2

        for idx, (pt1, pt2) in enumerate(very_good_matches):
            if normalise:
                p1_w = (pt1[0] - width1) / mm1
                p1_h = (pt1[1] - height1) / mm1
                p2_w = (pt2[0] - width2) / mm2
                p2_h = (pt2[1] - height2) / mm2
                very_good_keypoints_1[idx] = np.array((p1_w, p1_h))
                very_good_keypoints_2[idx] = np.array((p2_w, p2_h))
            else:
                very_good_keypoints_1[idx] = np.array(pt1)
                very_good_keypoints_2[idx] = np.array(pt2)

        # Use RANSAC to find inliers
        retval, mask = cv2.findHomography(very_good_keypoints_1, very_good_keypoints_2, cv2.RANSAC, 100.0)
        mask = mask.ravel()
        r1, r2 = very_good_keypoints_1[mask == 1], very_good_keypoints_2[mask == 1]

        if draw:
            draw_file = 'matches.{}-{}.{}.jpg'.format(os.path.split(image1_name)[1], os.path.split(image2_name)[1], self.detector_type)
            draw_file = os.path.join(os.path.split(image1_name)[0], draw_file)
            if not os.path.exists(draw_file):
                draw_image = cv2.drawMatches(image1_data, [cv2.KeyPoint(x=r[0], y=r[1], _size=1) for r in r1], 
                                             image2_data, [cv2.KeyPoint(x=r[0], y=r[1], _size=1) for r in r2], 
                                             [cv2.DMatch(x, x, 1) for x in range(len(r1))], None)
                cv2.imwrite(draw_file, draw_image)
        return r1, r2

    def process(self, files):
        image1 = None
        image2 = None

        if use_progress:
            files = progressbar.progressbar(files, redirect_stdout=True)
        for file in files:
            if image1 is None:
                image1 = file
                continue
            image2 = image1
            image1 = file
            print("Processing {} and {}".format(image2, image1))
            kp1, kp2 = self.cross_match(image2, image1, draw=True)


def main():
    args = get_args()
    source_files = glob.glob(args.source)
    feature_matcher = FeatureMatcher(detector_type=args.detector, matcher_type=args.matcher)
    feature_matcher.process(source_files)


if __name__ == '__main__':
    main()
