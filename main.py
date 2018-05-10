#!/usr/bin/python3

import cv2
import numpy as np
import glob

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
    
    matrices = []
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
        matrices.append(intrinsicMatrix @ np.hstack((rotation, translation)))

    return matrices

intrinsicMatrix = calibrateCamera(glob.glob("pixel-calibration-downscaled/*.jpg"), (10, 7))
print("Found intrinsic camera matrix:")
print(intrinsicMatrix)

images = [cv2.imread("monitor-downscaled/IMG_20180503_144813.jpg"), cv2.imread("monitor-downscaled/IMG_20180503_144817.jpg")]
keypoints, matches = trackFeatures(images)
matrices = findProjectionMatrices(keypoints, matches, intrinsicMatrix)
print("Found projection matrices:")
print(matrices)

cameraA = np.hstack((intrinsicMatrix, np.mat([0, 0, 0]).T))
pointsA = np.array([keypoints[0][m.queryIdx].pt for m in matches[0]])
pointsB = np.array([keypoints[1][m.trainIdx].pt for m in matches[0]])
pointCloud = cv2.triangulatePoints(cameraA, matrices[0], pointsA.transpose(), pointsB.transpose()).transpose()
pointCloud = cv2.convertPointsFromHomogeneous(pointCloud).reshape(-1, 3)
print(pointCloud)
