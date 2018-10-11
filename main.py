import cv2
import numpy as np
import time
import sys

# GLOBAL VARIABLES
MIN_MATCH_COUNT = 8
DISTANCE_RATIO = 0.7

# TODO General process:
#
#  Loop through images
#for every frame i and i+1, do the following:
#    ##sfm
#    kpmatches = Sift/surf matching
#    get Fundamental Matrix for matches
#    get Essential matrix from fundatmental matrix
#    get Rt from Essential matrix
#
#    Triangulate points and add to 3d model
#    bundle adjustment (?) >>> combining multiple sets of 3d points?????///?/?/?
#    sparse cloud at this point
#
#    #mvs
#    optimization /remove outliers
#    >>> sparse points at this point
#    make point cloud dense (?) with mvs black magic
#        --> maybe can use library for this bit
#
#    generate ply file for visualization


def main():
    # Get video input
    if (len(sys.argv) != 2):
        print("Provide video file path as argument. e.g './test_data/sample1.mp4'")
        exit(1)
    fileName = sys.argv[1]
    if fileName == '0':
        # Use webcam if argument specified is '0'
        cap = cv2.VideoCapture(0)
        time.sleep(1) # warm up time
    else:
        cap = cv2.VideoCapture(fileName)

    # Create a resizeable window for selecting objects
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    # Initialise feature detector
    surf = cv2.xfeatures2d.SURF_create()

    # Begin main loop
    _, curr = cap.read()
    original = curr
    while (cap.isOpened()):
        # Update [prev,curr] frame pair
        prev = curr
        moreFrames, curr = cap.read()
        if moreFrames == False:
            break

        # Find keypoints and their descriptors
        kp1, des1 = surf.detectAndCompute(prev, None)
        kp2, des2 = surf.detectAndCompute(curr, None)

        # Find 'good' matches based on distance ratio
        good = []
        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1,des2,k=2)
        for m,n in matches:
            if m.distance/n.distance < DISTANCE_RATIO:
                good.append(m)

        # Check if enough good matches are found
        if len(good) > MIN_MATCH_COUNT:
            # store good keypoints in start/end arrays
            starts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            ends = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            # Find fundamental matrix, use the 8-point algorithm
            # NOTE: Using 8-point will always find <= 1 matrix (while 7-point
            #       sometimes finds multiple.
            fundM, mask = cv2.findFundamentalMat(starts, ends, cv2.FM_8POINT)

            # Do something with the fundamental matrix
            if fundM is not None:
                # Print the fundamental matrix
                print(np.matrix(fundM))
                print("")

                """
                FOLLOWING CODE DIDNT WORK OUT, but might be useful later
                # create mask to reveal only good matches
                matchesMask = mask.ravel().tolist()

                # Apply perspective transformation
                h, w, _ = curr.shape
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, fundM)

                # Draw 
                curr = cv2.polylines(curr, [np.int32(dst)], True,(255,0,0),1,cv2.LINE_AA)
                """
            else:
                print("Fundamental matrix not found.")

        else:
            print("Not enough matches found! Need at least %d" % (MIN_MATCH_COUNT))

        cv2.imshow('Video', curr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

main()
