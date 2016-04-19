"""
Created on Jan 1, 2016

@author: Gregory Kramida
"""

import cv2
import numpy as np


def generate_board_object_points(board_height, board_width, board_square_size):
    board_dims = (board_width,board_height)
    object_points = np.zeros((board_height*board_width,1,3), np.float32)
    object_points[:, :, :2] = np.indices(board_dims).T.reshape(-1, 1, 2)
    # convert square sizes to meters
    object_points *= board_square_size
    return object_points


def homogenize_4vec(vec):
    return np.array([vec[0] / vec[3], vec[1] / vec[3], vec[2] / vec[3], 1.0]).T


class Pose(object):
    def __init__(self, transform=None, inverse_transform=None, rotation_vector=None, translation_vector=None):
        if transform is None:
            if translation_vector is None or rotation_vector is None:
                raise (ValueError("Expecting either the transform matrix or both the rotation & translation vector"))
            rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
            self.T = np.vstack((np.append(rotation_matrix, translation_vector, 1), [0, 0, 0, 1]))
        else:
            self.T = transform
            if translation_vector is None:
                translation_vector = transform[0:3, 3].reshape(3, 1)
            if rotation_vector is None:
                rot_mat = transform[0:3, 0:3]
                rotation_vector = cv2.Rodrigues(rot_mat)[0]
        if inverse_transform is None:
            rot_mat = cv2.Rodrigues(rotation_vector)[0]
            rot_mat_inv = rot_mat.T
            inverse_translation = -rot_mat_inv.dot(translation_vector)
            inverse_transform = np.vstack((np.append(rot_mat_inv, inverse_translation, 1), [0, 0, 0, 1]))

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
