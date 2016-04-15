"""
Created on Nov 23, 2015
@author: Gregory Kramida
@licence: Apache v2

Copyright 2015-2016 Gregory Kramida

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import time
import cv2


def undistort_stereo(stereo_rig, test_im_left, test_im_right, size_factor):
    im_size = test_im_left.shape
    new_size = (int(im_size[1] * size_factor), int(im_size[0] * size_factor))
    rotation1, rotation2, pose1, pose2 = \
        cv2.stereoRectify(cameraMatrix1=stereo_rig.cameras[0].intrinsic_mat,
                          distCoeffs1=stereo_rig.cameras[0].distortion_coeffs,
                          cameraMatrix2=stereo_rig.cameras[1].intrinsic_mat,
                          distCoeffs2=stereo_rig.cameras[1].distortion_coeffs,
                          imageSize=im_size,
                          R=stereo_rig.rotation,
                          T=stereo_rig.translation,
                          flags=cv2.CALIB_ZERO_DISPARITY,
                          newImageSize=new_size)[0:4]
    map1x, map1y = cv2.initUndistortRectifyMap(stereo_rig.cameras[0].intrinsic_mat,
                                               stereo_rig.cameras[0].distortion_coeffs,
                                               rotation1, pose1, new_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(stereo_rig.cameras[1].intrinsic_mat,
                                               stereo_rig.cameras[1].distortion_coeffs,
                                               rotation2, pose2, new_size, cv2.CV_32FC1)
    rect_left = cv2.remap(test_im_left, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(test_im_right, map2x, map2y, cv2.INTER_LINEAR)
    return rect_left, rect_right


def calibrate(camera, object_points, flags, criteria):
    # OpenCV prefers [width x height] as "Size" to [height x width]
    frame_dims = (camera.intrinsics.resolution[1], camera.intrinsics.resolution[0])
    start = time.time()
    camera.intrinsics.error, camera.intrinsics.intrinsic_mat, camera.intrinsics.distortion_coeffs, \
        rotation_vectors, translation_vectors = \
        cv2.calibrateCamera(object_points, camera.imgpoints, frame_dims,
                            camera.intrinsics.intrinsic_mat,
                            camera.intrinsics.distortion_coeffs,
                            flags=flags, criteria=criteria)
    end = time.time()
    camera.intrinsics.time = end - start
    return rotation_vectors, translation_vectors


def calibrate_wrapper(camera, object_points,
                      use_rational_model=True,
                      use_tangential=False,
                      use_thin_prism=False,
                      max_iterations=30,
                      use_existing_guess=False,
                      test=False):
    flags = 0
    if test:
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
        # fix everything
        flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
        flags = flags | cv2.CALIB_FIX_ASPECT_RATIO
        flags = flags | cv2.CALIB_FIX_FOCAL_LENGTH
        # apparently, we can't fix the tangential distance. What the hell? Zero it out.
        flags = flags | cv2.CALIB_ZERO_TANGENT_DIST
        flags = flags | cv2.CALIB_FIX_K1
        flags = flags | cv2.CALIB_FIX_K2
        flags = flags | cv2.CALIB_FIX_K3
        flags = flags | cv2.CALIB_FIX_K4
        flags = flags | cv2.CALIB_FIX_K5
        flags = flags | cv2.CALIB_FIX_K6
        flags = flags | cv2.CALIB_FIX_S1_S2_S3_S4
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 1)
    else:
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iterations,
                    2.2204460492503131e-16)
    if use_existing_guess:
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
    if not use_tangential:
        flags = flags | cv2.CALIB_ZERO_TANGENT_DIST
    if use_rational_model:
        flags = flags | cv2.CALIB_RATIONAL_MODEL
    if use_thin_prism:
        flags = flags | cv2.CALIB_THIN_PRISM_MODEL
    return calibrate(camera, object_points, flags, criteria)


def stereo_calibrate(rig,
                     object_points,
                     use_fisheye=False,
                     use_rational_model=True,
                     use_tangential=False,
                     use_thin_prism=False,
                     precalibrate_solo=True,
                     stereo_only=False,
                     max_iterations=30,
                     fix_intrinsics=False):
    # timestamp of calibration for record
    signature = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    rig.id = signature

    # shorten notation later in the code
    cam0 = rig.cameras[0]
    cam1 = rig.cameras[1]
    intrinsics0 = rig.cameras[0].intrinsics
    intrinsics1 = rig.cameras[1].intrinsics

    # OpenCV prefers the Width x Height as "Size" to Height x Width
    frame_dims = (intrinsics0.resolution[1], intrinsics0.resolution[0])

    # Global flags
    flags = 0

    if fix_intrinsics:
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iterations, 2.2204460492503131e-16)

    if stereo_only:
        flags = flags | cv2.CALIB_FIX_INTRINSIC
        precalibrate_solo = False

    if use_fisheye:
        d0 = intrinsics0.distortion_coeffs
        d1 = intrinsics1.distortion_coeffs
        # this hack is necessary to get around incorrect argument handling for fisheye.stereoCalibrate
        # function in OpenCV
        object_points_transposed = [np.transpose(point_set, (1, 0, 2)).astype(np.float64) for point_set in object_points]
        left_image_points_transposed = [np.transpose(point_set, (1, 0, 2)).astype(np.float64) for point_set in
                                        intrinsics0.imgpoints]
        right_image_points_transposed = [np.transpose(point_set, (1, 0, 2)).astype(np.float64) for point_set in
                                         intrinsics1.imgpoints]
        rig.error, intrinsics0.intrinsic_mat, intrinsics0.distortion_coeffs, intrinsics1.intrinsic_mat, \
            intrinsics1.distortion_coeffs, rig.rotation, rig.translation \
            = cv2.fisheye.stereoCalibrate(object_points_transposed,
                                          left_image_points_transposed,
                                          right_image_points_transposed,
                                          intrinsics0.intrinsic_mat, d0,
                                          intrinsics1.intrinsic_mat, d1,
                                          imageSize=frame_dims,
                                          flags=flags,
                                          criteria=criteria)
    else:
        if not use_tangential:
            flags = flags | cv2.CALIB_ZERO_TANGENT_DIST
        if use_rational_model:
            flags = flags | cv2.CALIB_RATIONAL_MODEL
        if use_thin_prism:
            flags = flags | cv2.CALIB_RATIONAL_MODEL
        if precalibrate_solo:
            calibrate(cam0, object_points, flags, criteria)
            calibrate(cam1, object_points, flags, criteria)
            flags = flags | cv2.CALIB_FIX_INTRINSIC

        start = time.time()
        rig.extrinsics.error, \
            intrinsics0.intrinsic_mat, intrinsics0.distortion_coeffs, \
            intrinsics1.intrinsic_mat, intrinsics1.distortion_coeffs, \
            rig.extrinsics.rotation, rig.extrinsics.translation, \
            rig.extrinsics.essential_mat, rig.extrinsics.fundamental_mat \
            = cv2.stereoCalibrate(object_points,
                                  cam0.imgpoints,
                                  cam1.imgpoints,
                                  cameraMatrix1=intrinsics0.intrinsic_mat,
                                  distCoeffs1=intrinsics0.distortion_coeffs,
                                  cameraMatrix2=intrinsics1.intrinsic_mat,
                                  distCoeffs2=intrinsics1.distortion_coeffs,
                                  imageSize=frame_dims,
                                  flags=flags,
                                  criteria=criteria)
        end = time.time()
        rig.extrinsics.time = end - start
