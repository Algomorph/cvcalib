'''
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
'''

import numpy as np
import time
import calib.data as data
import cv2#@UnresolvedImport


def generate_preview(stereo_calib_info, test_im_left, test_im_right, size_factor):
    im_size = test_im_left.shape
    new_size = (int(im_size[1]*size_factor),int(im_size[0]*size_factor))
    R1, R2, P1, P2 = \
    cv2.stereoRectify(cameraMatrix1=stereo_calib_info.videos[0].intrinsic_mat, 
                      distCoeffs1  =stereo_calib_info.videos[0].distortion_coeffs, 
                      cameraMatrix2=stereo_calib_info.videos[1].intrinsic_mat,
                      distCoeffs2  =stereo_calib_info.videos[1].distortion_coeffs, 
                      imageSize=im_size, 
                      R=stereo_calib_info.rotation,
                      T=stereo_calib_info.translation, 
                      flags=cv2.CALIB_ZERO_DISPARITY, 
                      newImageSize=new_size)[0:4]
    map1x, map1y = cv2.initUndistortRectifyMap(stereo_calib_info.videos[0].intrinsic_mat, 
                                               stereo_calib_info.videos[0].distortion_coeffs, 
                                               R1, P1, new_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(stereo_calib_info.videos[1].intrinsic_mat, 
                                               stereo_calib_info.videos[1].distortion_coeffs, 
                                               R2, P2, new_size, cv2.CV_32FC1)
    rect_left = cv2.remap(test_im_left, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(test_im_right, map2x, map2y, cv2.INTER_LINEAR)
    return rect_left, rect_right

def calibrate(objpoints, imgpoints, flags, criteria, calibration_info, verbose = False):
    #OpenCV prefers the width x height as "Size" to height x width
    frame_dims = (calibration_info.resolution[1],calibration_info.resolution[0])
    start = time.time()
    calibration_info.error, calibration_info.intrinsic_mat, calibration_info.distortion_coeffs =\
    cv2.calibrateCamera(objpoints, imgpoints, frame_dims, 
                        calibration_info.intrinsic_mat, 
                        calibration_info.distortion_coeffs, 
                        flags=flags,criteria = criteria)[0:3]
    end = time.time()
    calibration_info.time = end - start
    return calibration_info

def calibrate_wrapper(objpoints, video,
                      use_rational_model = True,
                      use_tangential = False,
                      max_iters = 30,
                      initial_calibration = None):
    flags = 0
    if initial_calibration != None:
        video.calib = initial_calibration
        flags += cv2.CALIB_USE_INTRINSIC_GUESS
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iters, 
                2.2204460492503131e-16)
    if not use_tangential:
        flags += cv2.CALIB_ZERO_TANGENT_DIST
    if use_rational_model:
        flags += cv2.CALIB_RATIONAL_MODEL
    calibration_result = calibrate(objpoints, video.imgpoints, flags, 
                                    criteria, video.calib)
    return calibration_result



def stereo_calibrate(limgpoints,rimgpoints,objpoints,
                     resolution,
                     use_fisheye = False,
                     use_rational_model = True,
                     use_tangential = False,
                     precalibrate_solo = True,
                     stereo_only = False, 
                     max_iters = 30,
                     initial_calibration = None):
    flags = 0
    
    signature = time.strftime("%Y%m%d-%H%M%S",time.localtime())
    if initial_calibration != None:
        result = initial_calibration
        flags += cv2.CALIB_USE_INTRINSIC_GUESS
        result.id = signature
    else:
        result = data.StereoRig((data.CameraIntrinsics(resolution, index=0),
                                             data.CameraIntrinsics(resolution, index=1)), 
                                             _id=signature)
    #shorten notation later in the code
    cam0 = result.intrinsics[0]
    cam1 = result.intrinsics[1]
    
    #OpenCV prefers the Width x Height as "Size" to Height x Width
    frame_dims = (resolution[1],resolution[0])
    
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iters, 2.2204460492503131e-16)
    
    if(stereo_only):
        if(initial_calibration == None):
            raise ValueError("Initial calibration required when calibrating only stereo parameters.")
        flags = flags | cv2.CALIB_FIX_INTRINSIC
    
    if use_fisheye:
        if initial_calibration == None:  
            d0 = np.zeros(4,np.float64)
            d1 = np.zeros(4,np.float64)
        else:
            d0 = cam0.distortion_coeffs
            d1 = cam1.distortion_coeffs
        #this hack is necessary to get around incorrect argument handling for fisheye.stereoCalibrate
        #function in OpenCV
        obp2 = [np.transpose(pointset,(1,0,2)).astype(np.float64) for pointset in objpoints]
        lpts2 = [np.transpose(pointset,(1,0,2)).astype(np.float64) for pointset in limgpoints]
        rpts2 = [np.transpose(pointset,(1,0,2)).astype(np.float64) for pointset in rimgpoints]
        result.error, cam0.intrinsic_mat, cam0.distortion_coeffs, cam1.intrinsic_mat,\
        cam1.distortion_coeffs, result.rotation, result.translation \
        = cv2.fisheye.stereoCalibrate(obp2, lpts2, rpts2, 
                                      cam0.intrinsic_mat,d0, 
                                      cam1.intrinsic_mat,d1, 
                                      imageSize=frame_dims,
                                      flags=flags,
                                      criteria=criteria)
    else:
        if not use_tangential:
            flags += cv2.CALIB_ZERO_TANGENT_DIST
        if use_rational_model:
            flags += cv2.CALIB_RATIONAL_MODEL
        if(precalibrate_solo):
            cam0 = calibrate(objpoints, limgpoints, flags, criteria, cam0)
            cam1 = calibrate(objpoints, rimgpoints, flags, criteria, cam1)
            flags = flags | cv2.CALIB_FIX_INTRINSIC
        
        start = time.time()
        result.error,\
        cam0.intrinsic_mat, cam0.distortion_coeffs,\
        cam1.intrinsic_mat, cam1.distortion_coeffs,\
        result.rotation, result.translation, result.essential_mat, result.fundamental_mat\
        = cv2.stereoCalibrate(objpoints,limgpoints,rimgpoints, 
                              cameraMatrix1 = cam0.intrinsic_mat, 
                              distCoeffs1   = cam0.distortion_coeffs, 
                              cameraMatrix2 = cam1.intrinsic_mat, 
                              distCoeffs2   = cam1.distortion_coeffs,
                              imageSize = frame_dims,
                              flags = flags,
                              criteria = criteria)
        end = time.time()
        result.time = end - start 
    return result