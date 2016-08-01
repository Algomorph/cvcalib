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


def compute_stereo_rectification_maps(stereo_rig, im_size, size_factor):
    new_size = (int(im_size[1] * size_factor), int(im_size[0] * size_factor))
    rotation1, rotation2, pose1, pose2 = \
        cv2.stereoRectify(cameraMatrix1=stereo_rig.cameras[0].intrinsics.intrinsic_mat,
                          distCoeffs1=stereo_rig.cameras[0].intrinsics.distortion_coeffs,
                          cameraMatrix2=stereo_rig.cameras[1].intrinsics.intrinsic_mat,
                          distCoeffs2=stereo_rig.cameras[1].intrinsics.distortion_coeffs,
                          imageSize=(im_size[1], im_size[0]),
                          R=stereo_rig.cameras[1].extrinsics.rotation,
                          T=stereo_rig.cameras[1].extrinsics.translation,
                          flags=cv2.CALIB_ZERO_DISPARITY,
                          newImageSize=new_size
                          )[0:4]
    map1x, map1y = cv2.initUndistortRectifyMap(stereo_rig.cameras[0].intrinsics.intrinsic_mat,
                                               stereo_rig.cameras[0].intrinsics.distortion_coeffs,
                                               rotation1, pose1, new_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(stereo_rig.cameras[1].intrinsics.intrinsic_mat,
                                               stereo_rig.cameras[1].intrinsics.distortion_coeffs,
                                               rotation2, pose2, new_size, cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y


def undistort_stereo(stereo_rig, test_im_left, test_im_right, size_factor):
    im_size = test_im_left.shape
    map1x, map1y, map2x, map2y = compute_stereo_rectification_maps(stereo_rig, im_size, size_factor)
    rect_left = cv2.remap(test_im_left, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(test_im_right, map2x, map2y, cv2.INTER_LINEAR)
    return rect_left, rect_right


def __calibrate_intrinsics(camera, image_points, object_points, flags, criteria):
    """
    Calibrate intrinsics of the provided camera using provided image & object points & calibration flags & criteria.
    @param camera: camera to calibrate
    @param image_points: points in images taken with the camera that correspond to the 3d object_points.
    @param object_points: 3d points on the object that appears in *each* of the images.
    Usually, inner corners of a calibration board. Note: assumes *the same* object appears in all of the images.
    @param flags: OpenCV camera calibration flags. For details, see OpenCV calib3d documentation, calibrate function.
    @param criteria: OpenCV criteria.
    @return: estimated object-space rotation & translation vectors of the camera (assuming object is static)
    """
    # OpenCV prefers [width x height] as "Size" to [height x width]
    frame_dims = (camera.intrinsics.resolution[1], camera.intrinsics.resolution[0])
    start = time.time()
    camera.intrinsics.error, camera.intrinsics.intrinsic_mat, camera.intrinsics.distortion_coeffs, \
    rotation_vectors, translation_vectors = \
        cv2.calibrateCamera(objectPoints=np.array([object_points]*len(image_points)), imagePoints=image_points,
                            imageSize=frame_dims, cameraMatrix=camera.intrinsics.intrinsic_mat,
                            distCoeffs=camera.intrinsics.distortion_coeffs,
                            flags=flags, criteria=criteria)
    end = time.time()
    camera.intrinsics.time = end - start
    camera.intrinsics.timestamp = end
    camera.intrinsics.calibration_image_count = len(image_points)
    return rotation_vectors, translation_vectors


def fix_radial_flags(flags):
    flags = flags | cv2.CALIB_FIX_K1
    flags = flags | cv2.CALIB_FIX_K2
    flags = flags | cv2.CALIB_FIX_K3
    flags = flags | cv2.CALIB_FIX_K4
    flags = flags | cv2.CALIB_FIX_K5
    flags = flags | cv2.CALIB_FIX_K6
    return flags


def calibrate_intrinsics(camera, image_points,
                         object_points,
                         use_rational_model=True,
                         use_tangential=False,
                         use_thin_prism=False,
                         fix_radial=False,
                         fix_thin_prism=False,
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
        flags = fix_radial_flags(flags)
        flags = flags | cv2.CALIB_FIX_S1_S2_S3_S4
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 1, 0)
    else:
        if fix_radial:
            flags = fix_radial_flags(flags)
        if fix_thin_prism:
            flags = flags | cv2.CALIB_FIX_S1_S2_S3_S4
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iterations,
                    2.2204460492503131e-16)
    if use_existing_guess:
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
    if not use_tangential:
        flags = flags | cv2.CALIB_ZERO_TANGENT_DIST
    if use_rational_model:
        flags = flags | cv2.CALIB_RATIONAL_MODEL
        if len(camera.intrinsics.distortion_coeffs) < 8:
            camera.intrinsics.distortion_coeffs.resize((8,))
    if use_thin_prism:
        flags = flags | cv2.CALIB_THIN_PRISM_MODEL
        if len(camera.intrinsics.distortion_coeffs) != 12:
            camera.intrinsics.distortion_coeffs = np.resize(camera.intrinsics.distortion_coeffs, (12,))
    return __calibrate_intrinsics(camera, image_points, object_points, flags, criteria)


def calibrate_stereo(rig, image_point_sets,
                     object_points,
                     use_fisheye=False,
                     use_rational_model=True,
                     use_tangential=False,
                     use_thin_prism=False,
                     fix_radial=False,
                     fix_thin_prism=False,
                     precalibrate_solo=True,
                     stereo_only=False,
                     max_iterations=30,
                     use_intrinsic_guess=False):
    # timestamp of calibration for record
    signature = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    rig.id = signature

    # compile image point sets in case of disjunct sets
    if len(image_point_sets) != 2:
        raise ValueError(
            "Expecting two total image point sets, each can be list or hash. Got object of length {:d} instead.".format(
                len(image_point_sets)))

    if type(image_point_sets[0]) == list or type(image_point_sets[0]) == np.ndarray:
        solo_image_points0 = image_point_sets[0]
        solo_image_points1 = image_point_sets[1]
        stereo_image_points0 = image_point_sets[0]
        stereo_image_points1 = image_point_sets[1]
        if len(stereo_image_points0) != len(stereo_image_points1):
            raise ValueError("When passed in as lists, image point sets should have the same length." +
                             " Got lengths {:d} and {:d} instead.".format(
                                 len(stereo_image_points0), len(stereo_image_points1)))
    else:
        # assume dict
        solo_image_points0 = np.array(list(image_point_sets[0].values()))
        solo_image_points1 = np.array(list(image_point_sets[1].values()))
        frame_number_overlap = list(set(image_point_sets[0].keys()).intersection(set(image_point_sets[1].keys())))
        stereo_image_points0 = np.array([image_point_sets[0][frame_number] for frame_number in frame_number_overlap])
        stereo_image_points1 = np.array([image_point_sets[1][frame_number] for frame_number in frame_number_overlap])

    cam0 = rig.cameras[0]
    cam1 = rig.cameras[1]
    intrinsics0 = rig.cameras[0].intrinsics
    intrinsics1 = rig.cameras[1].intrinsics

    if not precalibrate_solo and not stereo_only and intrinsics0.resolution != intrinsics1.resolution:
        raise ValueError("calibrate_stereo: to ensure proper intrisic matrix intialization," +
                         " stereo calibration for different-resolution cameras must either use precalibrate_solo flag" +
                         " or stereo_only flag. In the latter case, cameras in rig should already have " +
                         " proper intrinsics.")

    # OpenCV prefers the Width x Height as "Size" to Height x Width
    frame_dims = (intrinsics0.resolution[1], intrinsics0.resolution[0])

    # Global flags
    flags = 0

    if use_intrinsic_guess:
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iterations, 2.2204460492503131e-16)

    if stereo_only:
        flags = flags | cv2.CALIB_FIX_INTRINSIC
        precalibrate_solo = False

    if use_fisheye:
        # this hack is necessary to get around incorrect argument handling for fisheye.stereoCalibrate
        # function in OpenCV
        rig.error, intrinsics0.intrinsic_mat, intrinsics0.distortion_coeffs, intrinsics1.intrinsic_mat, \
        intrinsics1.distortion_coeffs, rig.rotation, rig.translation \
            = cv2.fisheye.stereoCalibrate([object_points] * len(stereo_image_points0),
                                          stereo_image_points0,
                                          stereo_image_points1,
                                          intrinsics0.intrinsic_mat,
                                          intrinsics0.distortion_coeffs,
                                          intrinsics1.intrinsic_mat,
                                          intrinsics1.distortion_coeffs,
                                          imageSize=frame_dims,
                                          flags=flags,
                                          criteria=criteria)
    else:
        if not use_tangential:
            flags = flags | cv2.CALIB_ZERO_TANGENT_DIST
        if use_rational_model:
            flags = flags | cv2.CALIB_RATIONAL_MODEL
            # resize to accommodate sought number of coefficients
            if len(intrinsics0.distortion_coeffs) < 8:
                intrinsics0.distortion_coeffs.resize((8,))
            if len(intrinsics1.distortion_coeffs) < 8:
                intrinsics1.distortion_coeffs.resize((8,))
        if use_thin_prism:
            flags = flags | cv2.CALIB_THIN_PRISM_MODEL
            # the thin prism model currently demands exactly 12 coefficients
            if len(intrinsics0.distortion_coeffs) != 12:
                intrinsics0.distortion_coeffs = np.resize(intrinsics0.distortion_coeffs, (12,))
            if len(intrinsics1.distortion_coeffs) != 12:
                intrinsics1.distortion_coeffs = np.resize(intrinsics1.distortion_coeffs, (12,))
        if fix_radial:
            if fix_radial:
                flags = fix_radial_flags(flags)
            if fix_thin_prism:
                flags = flags | cv2.CALIB_FIX_S1_S2_S3_S4

        if precalibrate_solo:
            __calibrate_intrinsics(cam0, solo_image_points0, object_points, flags, criteria)
            __calibrate_intrinsics(cam1, solo_image_points1, object_points, flags, criteria)
            flags = flags | cv2.CALIB_FIX_INTRINSIC
        start = time.time()
        cam1.extrinsics.error, \
        intrinsics0.intrinsic_mat, intrinsics0.distortion_coeffs, \
        intrinsics1.intrinsic_mat, intrinsics1.distortion_coeffs, \
        cam1.extrinsics.rotation, cam1.extrinsics.translation, \
        cam1.extrinsics.essential_mat, cam1.extrinsics.fundamental_mat \
            = cv2.stereoCalibrate(np.array([object_points] * len(stereo_image_points0)),
                                  stereo_image_points0,
                                  stereo_image_points1,
                                  cameraMatrix1=intrinsics0.intrinsic_mat,
                                  distCoeffs1=intrinsics0.distortion_coeffs,
                                  cameraMatrix2=intrinsics1.intrinsic_mat,
                                  distCoeffs2=intrinsics1.distortion_coeffs,
                                  imageSize=frame_dims,
                                  flags=flags,
                                  criteria=criteria)
        end = time.time()
        cam1.extrinsics.time = end - start
        cam1.extrinsics.timestamp = end
        cam1.extrinsics.calibration_image_count = len(stereo_image_points0)


def calibrate(rig, image_point_sets,
              object_points,
              use_fisheye=False,
              use_rational_model=True,
              use_tangential=False,
              use_thin_prism=False,
              fix_radial=False,
              fix_thin_prism=False,
              precalibrate_solo=True,
              extrinsics_only=False,
              max_iterations=30,
              use_intrinsic_guess=False,
              test=False):
    if len(rig.cameras) != len(image_point_sets):
        raise ValueError("The number of cameras in the rig must be equal to the number of videos.")
    if len(rig.cameras) == 1:
        camera = rig.cameras[0]
        calibrate_intrinsics(camera, image_point_sets[0],
                             object_points,
                             use_rational_model,
                             use_tangential,
                             use_thin_prism,
                             fix_radial,
                             fix_thin_prism,
                             max_iterations,
                             use_intrinsic_guess,
                             test)
    elif len(rig.cameras) == 2:
        calibrate_stereo(rig, image_point_sets,
                         object_points,
                         use_fisheye,
                         use_rational_model,
                         use_tangential,
                         use_thin_prism,
                         fix_radial,
                         fix_thin_prism,
                         precalibrate_solo,
                         extrinsics_only,
                         max_iterations,
                         use_intrinsic_guess)
    else:
        raise NotImplementedError(
            "Support for rigs with arbitrary number of cameras is not yet implemented. Please use 1 or two cameras.")
