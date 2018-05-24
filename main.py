#!/usr/bin/python3

import cv2
import numpy as np
import glob
import os
import tempfile

def calibrateCamera(imagePaths, shape):
    """Calculates intrinsic camera matrix from a chessboard pattern.

    imagePaths = List of input images
    shape = (columns, rows) of the chessboard

    Assumes the images are already corrected for lens distortion.

    Each square of the chessboard is treated as a unit square in the resulting
    world coordinates. e.g., If you use 1x1cm squares, then the units of the
    world coordinates are in cm.
    """

    corners = (shape[0] - 1, shape[1] - 1)
    imageShape = None

    # Define the chessboard's world points
    chessboardWPoints = np.zeros((corners[0] * corners[1], 3), np.float32)
    chessboardWPoints[:, :2] = np.mgrid[0:corners[0], 0:corners[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    worldPoints = [] # 3D points in world space
    imagePoints = [] # 2D points in image plane
    for path in imagePaths:
        # Read the image
        image = cv2.imread(path)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not imageShape:
            imageShape = grey.shape
        else:
            assert(imageShape == grey.shape)

        # Find the chess board corners
        found, foundCorners = cv2.findChessboardCorners(
            grey, corners,
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # If found, add object points, image points (after refining them)
        if found:
            worldPoints.append(chessboardWPoints)
            imagePoints.append(foundCorners)

            # Draw and display the corners
            """cv2.drawChessboardCorners(image, corners, foundCorners, found)
            cv2.imshow("Detected Chessboards", image)
            cv2.waitKey(500)
    cv2.destroyAllWindows()"""

    reprojError, cameraMatrix, _, _, _ = cv2.calibrateCamera(
        worldPoints, imagePoints, imageShape, None, None,
        flags = cv2.CALIB_FIX_ASPECT_RATIO |
            # Assume no distortion
            cv2.CALIB_ZERO_TANGENT_DIST |
            cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    )
    assert(reprojError < 0.5)

    return cameraMatrix

def detectFeatures(image):
    sift = cv2.xfeatures2d.SIFT_create()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(grey, None)
    return keypoints, descriptors

def matchFeatures(descriptorsA, descriptorsB):
    matcher = cv2.BFMatcher()
    #matcher = cv2.FlannBasedMatcher()
    matchesA = matcher.knnMatch(descriptorsA, descriptorsB, 2)
    matchesB = matcher.knnMatch(descriptorsB, descriptorsA, 2)

    # Use Lowe's nearest neighbour to next nearest ratio test (as presented in
    # the original SIFT paper section 7.1) to filter out likely-"bad" matches
    goodMatchesA = []
    goodMatchesB = []
    for a, b in matchesA:
        if a.distance < 0.8 * b.distance:
            goodMatchesA.append(a)
    for a, b in matchesB:
        if a.distance < 0.8 * b.distance:
            goodMatchesB.append(a)

    # Cross check the matches
    goodMatchesB = set((m.trainIdx, m.queryIdx) for m in goodMatchesB)
    goodMatches = []
    for match in goodMatchesA:
        if (match.queryIdx, match.trainIdx) in goodMatchesB:
            goodMatches.append(match)

    return goodMatches

def trackFeatures(images):
    assert(len(images) >= 2)

    # Detect the features
    keypoints, descriptors = list(zip(*(detectFeatures(image) for image in images)))

    # Match the features along every adjacent pair of images
    matches = []
    for i in range(len(images) - 1):
        descriptorsA = descriptors[i]
        descriptorsB = descriptors[i + 1]
        matches.append(matchFeatures(descriptorsA, descriptorsB))

    return keypoints, matches

def findProjectionMatrices(keypoints, matches, intrinsicMatrix):
    """Finds the projection matrix for each camera relative to the first camera.

    keypoints = [keypoints of each image]
    matches = matches between keypoints[i] and keypoints[i + 1]
    intrinsicMatrix = Shared intrinsic camera matrix of all the cameras

    Updates the matches to only include inliers, and returns the 4x3 projection
    matrices.
    """

    assert(len(keypoints) == len(matches) + 1)

    rotations = []
    translations = []
    matrices = []
    prevRotation = np.identity(3)
    prevTranslation = np.zeros((3, 1))
    rotations.append(prevRotation)
    translations.append(prevTranslation)
    matrices.append(intrinsicMatrix @ np.hstack((prevRotation, prevTranslation)))
    for i, match in enumerate(matches):
        # Find the essential matrix
        pointsA = np.array([keypoints[i][m.queryIdx].pt for m in match])
        pointsB = np.array([keypoints[i + 1][m.trainIdx].pt for m in match])
        essentialMatrix, mask = cv2.findEssentialMat(pointsA, pointsB, intrinsicMatrix)

        # If #inliers < 50% of matching points, probably wasn't a good matrix
        #print(mask)
        #assert(np.count_nonzero(mask) >= 0.5 * len(match))

        # Recover the relative rotation and translation of the cameras.
        # recoverPose() gives the relative transformation of camera B from A.
        _, rotation, translation, _ = cv2.recoverPose(
            essentialMatrix, pointsA, pointsB,
            intrinsicMatrix, mask = mask
        )

        # Invert the transforms so we get the transformation from camera A to B
        rotation = np.linalg.inv(rotation)
        translation = -translation

        # Update the matches to only include inliers
        inlierMatches = []
        for idx, m in enumerate(match):
            if mask[idx][0]:
                inlierMatches.append(m)
        matches[i] = inlierMatches

        # Compute the projection matrix
        prevRotation = rotation @ prevRotation
        prevTranslation = translation + prevTranslation
        rotations.append(prevRotation)
        translations.append(prevTranslation)
        matrices.append(intrinsicMatrix @ np.hstack((prevRotation, prevTranslation)))

    return matrices, rotations, translations

def computePointCloud(images, projectionMatrices):
    assert(len(images) == len(projectionMatrices))
    with tempfile.TemporaryDirectory() as tempdir:
        os.mkdir(os.path.join(tempdir, "visualize"))
        os.mkdir(os.path.join(tempdir, "txt"))
        os.mkdir(os.path.join(tempdir, "models"))
        for i, (image, matrix) in enumerate(zip(images, projectionMatrices)):
            if type(image) == str:
                imageExt = os.path.splitext(image)[1]
                assert(imageExt == ".jpg" or imageExt == ".ppm")
                destImage = "{:08d}{}".format(i, imageExt)
                os.symlink(os.path.abspath(image), os.path.join(tempdir, "visualize", destImage))
            else:
                destImage = os.path.join(tempdir, "visualize", "{:08d}.jpg".format(i))
                cv2.imwrite(destImage, image)

            destMatrix = "{:08d}.txt".format(i)
            with open(os.path.join(tempdir, "txt", destMatrix), "w") as destMatrixFile:
                destMatrixFile.write(
                    "CONTOUR\n"
                    "{P[0][0]} {P[0][1]} {P[0][2]} {P[0][3]}\n"
                    "{P[1][0]} {P[1][1]} {P[1][2]} {P[1][3]}\n"
                    "{P[2][0]} {P[2][1]} {P[2][2]} {P[2][3]}\n"
                    .format(P = matrix)
                )

        with open(os.path.join(tempdir, "options.txt"), "w") as optionsFile:
            optionsFile.write(
                "timages -1 0 {}\n".format(len(images)) +
                "oimages 0\n" +
                "minImageNum 2\n"
            )

        retval = os.spawnve(
            os.P_WAIT,
            os.path.join("pmvs2", "pmvs2"),
            ["pmvs2", os.path.join(tempdir, ""), "options.txt"],
            {"LD_LIBRARY_PATH": "pmvs2"}
        )
        #import sys
        #sys.stdin.readline()
        assert(retval == 0)

        with open(os.path.join(tempdir, "models", "options.txt.ply"), "r") as plyFile:
            return plyFile.read()

"""intrinsicMatrix = calibrateCamera(glob.glob("pixel-calibration-downscaled/*.jpg"), (10, 7))
print("Found intrinsic camera matrix:")
print(intrinsicMatrix)"""

"""imageNames = ["monitor-downscaled/IMG_20180503_144813.jpg", "monitor-downscaled/IMG_20180503_144817.jpg"]
images = [cv2.imread(i) for i in imageNames]
keypoints, matches = trackFeatures(images)
matrices, rotations, translations = findProjectionMatrices(keypoints, matches, intrinsicMatrix)
print("Found projection matrices:")
print(matrices)"""

"""imageNames = sorted(glob.glob("temple-test/*.png"))
images = [cv2.imread(i) for i in imageNames]
params = np.loadtxt("temple-test/templeSR_par.txt", skiprows = 1, usecols = range(1, 22))
intrinsicMatrices = params[:, :9].reshape(-1, 3, 3)
rotations = params[:, 9:18].reshape(-1, 3, 3)
translations = params[:, 18:].reshape(-1, 3, 1)
matrices = [K @ np.hstack((R, t)) for K, R, t in zip(intrinsicMatrices, rotations, translations)]"""

params = np.loadtxt("temple-test/templeSR_par.txt", skiprows = 1, usecols = range(1, 22))
intrinsicMatrix = params[:, :9].reshape(-1, 3, 3)[0]

imageNames = sorted(glob.glob("temple-test/*.png"))
images = [cv2.imread(i) for i in imageNames]
keypoints, matches = trackFeatures(images)
matrices, rotations, translations = findProjectionMatrices(keypoints, matches, intrinsicMatrix)
print("Found translations:")
print(repr(np.array(translations).reshape(-1, 3)))

pointCloud = computePointCloud(images, matrices)
with open("output.ply", "w") as outputFile:
    outputFile.write(pointCloud)
