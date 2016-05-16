import cv2
import numpy as np
import calib.corner_kernels as ck
from math import pi, tan
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


def __find_dominant_directions(hist64, cutoff=10):
    # find clusters
    clusters = []
    cur_cluster = []
    val_angle = 0
    angle_increment = pi / 64
    for val in hist64:
        if val > cutoff:
            cur_cluster.append((val, val_angle))
        else:
            if len(cur_cluster) > 0:
                clusters.append(cur_cluster)
                cur_cluster = []
        val_angle += angle_increment
    if len(cur_cluster) > 0:
        clusters.append(cur_cluster)
    # if the fist and last values are above threshold, join the first and last clusters
    if hist64[0] > cutoff and hist64[63] > cutoff:
        clusters[0] = clusters[len(clusters) - 1] + clusters[0]
        clusters = [np.array(cluster) for cluster in clusters[:-1]]
    else:
        clusters = [np.array(cluster) for cluster in clusters]

    if len(clusters) < 2:
        return None

    # find the two dominant clusters
    cluster_areas = [cluster[:, 0].sum() for cluster in clusters]
    biggest_at = np.argmax(cluster_areas)
    biggest_cluster_area = cluster_areas[biggest_at]
    cluster_areas[biggest_at] = -1.0
    second_biggest_at = np.argmax(cluster_areas)
    cluster_areas = [biggest_cluster_area, cluster_areas[second_biggest_at]]
    clusters = [clusters[biggest_at], clusters[second_biggest_at]]
    angles = []
    for i_cluster in range(0, 2):
        cluster = clusters[i_cluster]
        area = cluster_areas[i_cluster]
        mode = area / 2.0
        running_total = 0.
        for i_bin in range(0, len(cluster)):
            hist_bin = cluster[i_bin]
            new_total = running_total + hist_bin[0]
            if new_total > mode:
                # linear interpolation between bin angles
                if i_bin > 0:
                    angle_1 = cluster[i_bin - 1][1]
                    angle_2 = cluster[i_bin - 1][1]
                    frac = (mode - running_total) / hist_bin[0]
                else:
                    angle_1 = cluster[0][1]
                    angle_2 = cluster[1][1]
                    frac = mode / new_total
                if angle_1 > angle_2:
                    angle_2 += pi
                angle = angle_1 + frac * (angle_2 - angle_1)
                break
            running_total = new_total
        angles.append((-angle + (pi / 2)) % pi)
    angles.sort()
    return tuple(angles)


def __build_corner_template(size, directions):
    template = np.zeros((size, size), dtype=np.float32)
    a45 = pi / 4
    a90 = pi / 2
    a135 = pi / 2 + pi / 4
    s = size // 2
    for direction in directions:
        on_vertical_border = True
        sign = 1.0
        if 0. <= direction < a45:
            beta = direction
        elif a45 <= direction < a90:
            beta = a90 - direction
            on_vertical_border = False
        elif a90 <= direction < a135:
            beta = direction - a90
            on_vertical_border = False
            sign = -1.0
        elif a135 <= direction < pi:
            beta = pi - direction
            sign = -1.0
        else:
            raise ValueError("Illegal direction value: {:.3f}. Direction must be within [0, pi)".format(direction))

        s_tan_beta = s * tan(beta)
        p0c0 = 0
        p0c1 = int(0 + s + sign * s_tan_beta)
        p1c0 = 2 * s
        p1c1 = int(0 + s - sign * s_tan_beta)
        if on_vertical_border:
            p0 = (p0c0, p0c1)
            p1 = (p1c0, p1c1)
        else:
            p0 = (p0c1, p0c0)
            p1 = (p1c1, p1c0)
        cv2.line(template, p0, p1, 1, 3, cv2.LINE_AA)
    return template


def __filter_candidate(greyscale_image, coord, neighborhood_size):
    window = greyscale_image[coord[0] - neighborhood_size:coord[0] + neighborhood_size + 1,
             coord[1] - neighborhood_size:coord[1] + neighborhood_size + 1]
    grad_x = cv2.Sobel(window, cv2.CV_32FC1, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(window, cv2.CV_32FC1, dx=0, dy=1, ksize=3)
    grad_mag = np.abs(grad_x) + np.abs(grad_y)
    grad_mag_flat = grad_mag.flatten()
    orientations_flat = (cv2.phase(grad_x, grad_y) % pi).flatten()  # phase accuracy: about 0.3 degrees
    hist = (np.histogram(orientations_flat, bins=64, range=(0, pi), weights=grad_mag_flat)[0] /
            (neighborhood_size * neighborhood_size))

    return hist, grad_mag


def find_candidates(greyscale_image, neighborhood_size=20, candidate_threshold=.5):
    corner_likelihood = compute_inital_corner_likelihood(greyscale_image)
    # TODO: the absolute threshold should be statistically determined based on actual checkerboard images
    candidates = peak_local_max(corner_likelihood, neighborhood_size, corner_likelihood.max() * candidate_threshold)
    return candidates


def prep_img_save(img, b=5):
    return cv2.normalize(cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT, value=0), 0, 255,
                  cv2.NORM_MINMAX).astype(np.uint8)

def find_chessboard_corners(greyscale_image, neighborhood_size=10, candidate_threshold=.5):
    candidates = find_candidates(greyscale_image, neighborhood_size, candidate_threshold)
    bordered_image = cv2.copyMakeBorder(greyscale_image, neighborhood_size, neighborhood_size, neighborhood_size,
                                        neighborhood_size, cv2.BORDER_CONSTANT, value=0)
    detected_corners = []
    windows = []
    grad_mags = []
    templates = []
    ix_candidate = 0
    for candidate in candidates:
        print(ix_candidate)
        coord = candidate
        window = greyscale_image[coord[0] - neighborhood_size:coord[0] + neighborhood_size + 1,
                 coord[1] - neighborhood_size:coord[1] + neighborhood_size + 1]
        hist, grad_mag = __filter_candidate(bordered_image, candidate, neighborhood_size)
        win_b = cv2.copyMakeBorder(window, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
        windows.append(win_b)
        grad_mags.append(prep_img_save(grad_mag))
        angles = __find_dominant_directions(hist)
        if angles is not None:
            template = __build_corner_template(neighborhood_size * 2 + 1, angles)
            templates.append(prep_img_save(template))
        else:
            templates.append(np.zeros_like(win_b))
        ix_candidate += 1
        # if __filter_candidate(bordered_image, candidate, neighborhood_size):
        #      detected_corners.append(candidate)

    ch_test = np.vstack((np.hstack(windows), np.hstack(grad_mags), np.hstack(templates)))
    cv2.imwrite("~/Desktop/TMP/ch_test01.png", ch_test)

    detected_corners = np.array(detected_corners)
    # return detected_corners

    return candidates
