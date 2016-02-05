'''
Created on Nov 23, 2015

@author: algomorph
'''

import numpy as np

import cv2#@UnresolvedImport
import time
import calib.io as io
import calib.data as data

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

def generate_preview(stereo_calib_info, test_im_left, test_im_right):
    im_size = test_im_left.shape
    new_size = (int(im_size[1]*1.5),int(im_size[0]*1.5))
    R1, R2, P1, P2 = \
    cv2.stereoRectify(cameraMatrix1=stereo_calib_info.cameras[0].intrinsic_mat, 
                      distCoeffs1  =stereo_calib_info.cameras[0].distortion_coeffs, 
                      cameraMatrix2=stereo_calib_info.cameras[1].intrinsic_mat,
                      distCoeffs2  =stereo_calib_info.cameras[1].distortion_coeffs, 
                      imageSize=im_size, 
                      R=stereo_calib_info.rotation,
                      T=stereo_calib_info.translation, 
                      flags=cv2.CALIB_ZERO_DISPARITY, 
                      newImageSize=new_size)[0:4]
    map1x, map1y = cv2.initUndistortRectifyMap(stereo_calib_info.cameras[0].intrinsic_mat, 
                                               stereo_calib_info.cameras[0].distortion_coeffs, 
                                               R1, P1, new_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(stereo_calib_info.cameras[1].intrinsic_mat, 
                                               stereo_calib_info.cameras[1].distortion_coeffs, 
                                               R2, P2, new_size, cv2.CV_32FC1)
    rect_left = cv2.remap(test_im_left, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(test_im_right, map2x, map2y, cv2.INTER_LINEAR)
    return rect_left, rect_right

def stereo_calibrate(limgpoints,rimgpoints,objpoints,
                     resolution,
                     use_fisheye = False,
                     use_8 = True,
                     use_tangential = False,
                     precalibrate_solo = True, 
                     max_iters = 30,
                     path_to_calib_file = None):
    flags = 0
       
    if path_to_calib_file != None:
        result = io.load_opencv_stereo_calibration(path_to_calib_file)
        if(resolution != result.resolution):
            raise ValueError("Resolution in specified calibration file (" + 
                             path_to_calib_file + ") does not correspond to given resolution.")
        flags += cv2.CALIB_USE_INTRINSIC_GUESS
    else:
        signature = time.strftime("%Y%m%d-%H%M%S",time.localtime())
        result = data.StereoCalibrationInfo((data.CameraCalibrationInfo(resolution, index=0),
                                             data.CameraCalibrationInfo(resolution, index=1)), 
                                             _id=signature)
    #shorten notation later in the code
    cam0 = result.cameras[0]
    cam1 = result.cameras[1]
    
    #OpenCV prefers the Width x Height as "Size" to Height x Width
    frame_dims = (resolution[1],resolution[0])
    
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iters, 2.2204460492503131e-16)
    
    if use_fisheye:
        if path_to_calib_file == None:  
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
        if use_8:
            flags += cv2.CALIB_RATIONAL_MODEL
        if(precalibrate_solo):
            cam0 = calibrate(objpoints, limgpoints, flags, criteria, cam0)
            cam1 = calibrate(objpoints, rimgpoints, flags, criteria, cam1)  # @UnusedVariable
            flags += cv2.CALIB_FIX_INTRINSIC
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