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


def __filter_candidate(greyscale_image, coord, neighborhood_size):
    window = greyscale_image[coord[0] - neighborhood_size:coord[0] + neighborhood_size + 1,
             coord[1] - neighborhood_size:coord[1] + neighborhood_size + 1]
    grad_x = cv2.Sobel(window, cv2.CV_32FC1, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(window, cv2.CV_32FC1, dx=0, dy=1, ksize=3)
    grad_mag_flat = (np.abs(grad_x) + np.abs(grad_y)).flatten()
    orientations_flat = (cv2.phase(grad_x, grad_y)).flatten()  # phase accuracy: about 0.3 degrees
    hist = np.histogram(orientations_flat, bins=64, range=(0, 2 * pi), weights=grad_mag_flat)[0]

    return hist


def find_chessboard_corners(greyscale_image, neighborhood_size=10, candidate_threshold=.5):
    corner_likelihood = compute_inital_corner_likelihood(greyscale_image)
    # TODO: the absolute threshold should be statistically determined based on actual checkerboard images
    candidates = peak_local_max(corner_likelihood, neighborhood_size, corner_likelihood.max() * candidate_threshold)
    bordered_image = cv2.copyMakeBorder(greyscale_image, neighborhood_size, neighborhood_size, neighborhood_size,
                                        neighborhood_size, cv2.BORDER_CONSTANT, value=0)
    detected_corners = []
    # for candidate in candidates:
    #     if __filter_candidate(bordered_image, candidate, neighborhood_size):
    #         detected_corners.append(candidate)

    detected_corners = np.array(detected_corners)
    # return detected_corners

    return candidates
