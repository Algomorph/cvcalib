"""
Created on Jan 1, 2016

@author: Gregory Kramida
"""

import cv2
import numpy as np


def generate_board_object_points(board_height, board_width, board_square_size):
    board_dims = (board_width, board_height)
    object_points = np.zeros((board_height * board_width, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    # convert square sizes to meters
    object_points *= board_square_size
    return object_points


def homogenize_4vec(vec):
    return np.array([vec[0] / vec[3], vec[1] / vec[3], vec[2] / vec[3], 1.0]).T


class Pose(object):
    def __init__(self, transform=None, inverse_transform=None, rotation=None, translation_vector=None):
        if translation_vector is not None:
            if type(translation_vector) != np.ndarray:
                translation_vector = np.array(translation_vector)
            if translation_vector.shape != (3, 1):
                translation_vector = translation_vector.reshape(3, 1)
        if rotation is not None:
            if type(rotation) != np.ndarray:
                rotation = np.array(rotation)
            if rotation.size == 9:
                rotation_vector = cv2.Rodrigues(rotation)[0]
                rotation_matrix = rotation
            elif rotation.size == 3:
                rotation_matrix = cv2.Rodrigues(rotation)[0]
                rotation_vector = rotation
            else:
                raise ValueError(
                    "Wrong rotation size: {:d}. Expecting a 3-length vector or 3x3 matrix.".format(rotation.size))
        if transform is None:
            if translation_vector is None or rotation is None:
                raise (ValueError("Expecting either the transform matrix or both the rotation & translation vector"))
            self.T = np.vstack((np.append(rotation_matrix, translation_vector, axis=1), [0, 0, 0, 1]))
        else:
            self.T = transform
            if translation_vector is None:
                translation_vector = transform[0:3, 3].reshape(3, 1)
            if rotation is None:
                rotation_matrix = transform[0:3, 0:3]
                rotation_vector = cv2.Rodrigues(rotation_matrix)[0]
        if inverse_transform is None:
            rot_mat_inv = rotation_matrix.T
            inverse_translation = -rot_mat_inv.dot(translation_vector)
            inverse_transform = np.vstack((np.append(rot_mat_inv, inverse_translation, 1), [0, 0, 0, 1]))

        self.rmat = rotation_matrix
        self.tvec = translation_vector
        self.rvec = rotation_vector
        self.T_inv = inverse_transform

    def dot(self, other_pose):
        return Pose(self.T.dot(other_pose.T))

    def diff(self, other_pose):
        """
        Find difference between two poses.
        I.e. find the euclidean distance between unit vectors after being transformed by the poses.
        """
        unit_vector = np.array([1., 1., 1., 1.]).T
        p1 = self.T.dot(unit_vector)
        p2 = other_pose.T.dot(unit_vector)
        # no need to homogenize, since the last entry will end up being one anyway
        return np.linalg.norm(p1 - p2)  # it will also not contribute to the norm, i.e. 1 - 1 = 0

    @staticmethod
    def invert_pose_matrix(transform_matrix):
        translation_vector = transform_matrix[0:3, 3].reshape(3, 1)
        rotation_matrix = transform_matrix[0:3, 0:3]
        rotation_matrix_inverse = rotation_matrix.T
        translation_vector_inverse = -rotation_matrix_inverse.dot(translation_vector)
        return np.vstack((np.append(rotation_matrix_inverse, translation_vector_inverse, 1), [0, 0, 0, 1]))

    def __str__(self):
        return "================\nPose rotation: \n" + str(self.rmat) + "\nTranslation:\n" + str(
            self.tvec) + "\n===============\n"
