import cv2
import numpy as np
import calib.corner_kernels as ck
from math import pi
from skimage.feature import peak_local_max


def compute_corner_kernel_whole(size, offset=0.0):
    kern = np.zeros((size, size), dtype=np.uint8)
    half = size // 2
    factor = 255 / ((half - offset) * (half - offset))
    for row in range(0, half):
        for col in range(0, half):
            kern[row, col] = round((row + 1 - offset) * (col + 1 - offset) * factor)
        for col in range(half, size):
            kern[row, col] = round((row + 1 - offset) * (size - col - offset) * factor)
    for row in range(half, size):
        for col in range(0, half):
            kern[row, col] = round((size - row - offset) * (col + 1 - offset) * factor)
        for col in range(half, size):
            kern[row, col] = round((size - row - offset) * (size - col - offset) * factor)
    return kern


def compute_inital_corner_likelihood(image):
    likelihoods = []
    for prototype in ck.CORNER_KERNEL_PROTOTYPES:
        filter_responses = [cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=kernel) for kernel in prototype]
        fA, fB, fC, fD = filter_responses
        mean_response = (fA + fB + fC + fD) / 4.
        minAB = np.minimum(fA, fB)
        minCD = np.minimum(fC, fD)
        diff1 = minAB - mean_response
        diff2 = minCD - mean_response
        # For an ideal corner, the response of {A,B} should be greater than the mean response of {A,B,C,D},
        # while the response of {C,D} should be smaller, and vice versa for flipped corners.
        likelihood1 = np.minimum(diff1, -diff2)
        likelihood2 = np.minimum(-diff1, diff2)  # flipped case
        likelihoods.append(likelihood1)
        likelihoods.append(likelihood2)
    corner_likelihood = np.max(likelihoods, axis=0)
    return corner_likelihood


def find_chessboard_corners(greyscale_image, neighborhood_size = 10, max_theshold = .5):
    corner_likelihood = compute_inital_corner_likelihood(greyscale_image)
    # TODO: the absolute threshold should be statistically determined based on actual checkerboard images
    candidates = peak_local_max(corner_likelihood, neighborhood_size, corner_likelihood.max()*max_theshold)
    # TODO: this should be done on local neighborhoods!

    grad_x = cv2.Sobel(greyscale_image, cv2.CV_64FC1, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(greyscale_image, cv2.CV_64FC1, dx=0, dy=1, ksize=3)
    # grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5)
    orientations = cv2.phase(grad_x, grad_y).astype(np.float32)  # accuracy: about 0.3 degrees
    hist = cv2.calcHist([orientations], [0], None, [32], [0.0, 2 * pi])
    return hist
    # back_proj = cv2.calcBackProject([orientations], [0], hist, [0.0, 2*pi], 1.0)


