'''
Created on Feb 4, 2016

@author: algomorph
'''
import os
import os.path as osp
import cv2#@UnresolvedImport
import numpy as np
from calib import utils as cutils
from calib import io as cio
from calib.data import CameraInfo
import datetime
import sys
import re


class CalibrateVideoApplication:
    min_frames_to_calibrate = 4
    def __init__(self,args):
        self.camera = CameraInfo(args.folder,args.videos[0], 0)
        
        if(len(args.videos) > 1):
            self.cameras = [self.camera, CameraInfo(args.folder, args.videos[1], 1)]
            self.__automatic_filter_basic = self.__automatic_filter_basic_stereo
            self.__automatic_filter = self.__automatic_filter_stereo
            if(len(args.preview_files) != len(args.videos)):
                raise ValueError("There must be two preview file arguments passed in for stereo calibration.")
            self.total_frames = min(self.cameras[0].frame_count,self.cameras[1].frame_count)
            if(self.cameras[0].frame_dims != self.cameras[1].frame_dims):
                raise ValueError("The videos must have the same resolution.")
        else:
            self.cameras = [self.camera]
            self.__automatic_filter_basic = self.__automatic_filter_basic_mono
            self.__automatic_filter = self.__automatic_filter_mono
            self.total_frames = self.camera.frame_count
        
        self.full_frame_folder_path = osp.join(args.folder,args.filtered_image_folder)
        #if image folder (for captured frames) doesn't yet exist, create it
        if args.save_images and not os.path.exists(self.full_frame_folder_path):
            os.makedirs(self.full_frame_folder_path)
        self.full_corners_path = osp.join(args.folder,args.corners_file)
        
        #set up board (3D object points of checkerboard used for calibration)
        self.objpoints = []
        self.board_dims = (args.board_width,args.board_height)
        self.board_object_corner_set = np.zeros((args.board_height*args.board_width,1,3), np.float32)
        self.board_object_corner_set[:,:,:2] = np.indices(self.board_dims).T.reshape(-1, 1, 2)
        self.board_object_corner_set *= args.board_square_size
        
        if args.frame_numbers:
            path = osp.join(args.folder, args.frame_numbers)
            print("Loading frame numbers from \"{0:s}\"".format(path))
            npzfile = np.load(path)
            self.frame_numbers = set(npzfile["frame_numbers"])
            
        self.criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        self.frame_dims = self.camera.frame_dims
        
        
        self.pixel_difference_factor = 1.0 / (self.board_dims[0] * self.board_dims[1] * 3 * 256.0)
        if(args.use_existing):
            self.path_to_calib_file = osp.join(self.args.folder, self.args.output)
        else:
            self.path_to_calib_file = None
        if(args.output is None):
            args.output = "calib{0:s}.xml".format(re.sub(r"-|:","",
                                                         str(datetime.datetime.now())[:-7])
                                                  .replace(" ","-"))
        self.args = args
    
    def __automatic_filter_stereo(self, verbose = False):
        l_frame = self.cameras[0].frame
        lframe_prev = self.cameras[0].previous_frame
        r_frame = self.cameras[1].frame

        sharpness = min(cv2.Laplacian(l_frame, cv2.CV_64F).var(), 
                        cv2.Laplacian(r_frame, cv2.CV_64F).var())
        if(verbose):
            print("Minimum frame pair sharpness: " + sharpness)
        
        #compare left frame to the previous left **filtered** one
        ldiff = np.sum(abs(lframe_prev - l_frame)) * self.pixel_difference_factor
        
        if sharpness < self.args.sharpness_threshold or ldiff < self.args.difference_threshold:
            return False
        
        lfound,lcorners = cv2.findChessboardCorners(l_frame,self.board_dims)
        rfound,rcorners = cv2.findChessboardCorners(r_frame,self.board_dims)
        if not (lfound and rfound):
            return False
        
        self.cameras[0].current_corners = lcorners
        self.cameras[1].current_corners = rcorners
        
        return True
    
    def __automatic_filter_mono(self):
        frame = self.camera.frame
        frame_prev = self.camera.previous_frame
        sharpness = cv2.Laplacian(frame,cv2.CV_64F).var()
        if sharpness < self.args.sharpness_threshold:
            return False
        #compare left frame to the previous left **filtered** one
        ldiff = np.sum(abs(frame_prev - frame)) * self.pixel_difference_factor
        if ldiff < self.args.difference_threshold:
            return False
        
        found,corners = cv2.findChessboardCorners(frame,self.board_dims)
        
        if not found:
            return False
        
        self.camera.current_corners = corners
        
        return True
        
    
    def __automatic_filter_basic_stereo(self):
        l_frame = self.cameras[0].frame
        r_frame = self.cameras[1].frame
        
        lfound,lcorners = cv2.findChessboardCorners(l_frame,self.board_dims)
        rfound,rcorners = cv2.findChessboardCorners(r_frame,self.board_dims)
        if not (lfound and rfound):
            return False
        
        self.cameras[0].current_corners = lcorners
        self.cameras[1].current_corners = rcorners
        
        return True
    
    def __automatic_filter_basic_mono(self):
        frame = self.camera.frame
        found,corners = cv2.findChessboardCorners(frame,self.board_dims)  
        self.camera.current_corners = corners
        return found

    def load_frame_images(self):
        print("Loading frames from '{0:s}'".format(self.full_frame_folder_path))
        all_files = [f for f in os.listdir(self.full_frame_folder_path) 
                 if osp.isfile(osp.join(self.full_frame_folder_path,f)) and f.endswith(".png")]
        all_files.sort()
        
        usable_frame_ct = sys.maxsize
        
        for camera in self.cameras:
            #assume matching numbers in corresponding left & right files
            files = [f for f in all_files if f.startswith(camera.name)]
            cam_frame_ct = 0
            for ix_pair in range(len(files)):
                #TODO: assumes there is the same number of frames for all videos, and all frame
                #indexes match
                frame = cv2.imread(osp.join(self.full_frame_folder_path,files[ix_pair]))
                found,lcorners = cv2.findChessboardCorners(frame,self.board_dims)
                if not found:
                    raise ValueError("Could not find corners in image '{0:s}'".format(files[ix_pair]))
                grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(grey, lcorners, (11,11),(-1,-1),self.criteria_subpix)
                camera.imgpoints.append(lcorners)
                cam_frame_ct+=1
            usable_frame_ct = min(usable_frame_ct,cam_frame_ct)
        for i_frame in range(usable_frame_ct):#@UnusedVariable
            self.objpoints.append(self.board_object_corner_set)
        return usable_frame_ct
    
    def add_corners(self, usable_frame_ct, report_interval, i_frame):
        if(usable_frame_ct % report_interval == 0):
            print ("Usable frames: {0:d} ({1:.3%})"
                   .format(usable_frame_ct, float(usable_frame_ct)/(i_frame+1)))
            
        for camera in self.cameras:
            grey_frame = cv2.cvtColor(camera.frame,cv2.COLOR_BGR2GRAY)
            cv2.cornerSubPix(grey_frame, camera.current_corners, (11,11),(-1,-1),self.criteria_subpix)
            if(self.args.save_images):
                fname = (osp.join(self.full_frame_folder_path,
                               "{0:s}{1:04d}{2:s}".format(camera.name,i_frame,".png")))
                cv2.imwrite(fname, camera.frame)
            camera.imgpoints.append(camera.current_corners)
        self.objpoints.append(self.board_object_corner_set)
    
    def filter_frame_manually(self):
        if len(self.cameras) == 2:
            display_image = np.hstack((self.cameras[0].frame, self.cameras[1].frame))
        else:
            display_image = self.cameras[0].frame
        cv2.imshow("frame", display_image)
        key = cv2.waitKey(0) & 0xFF
        add_corners = (key == ord('a'))
        cv2.destroyWindow("frame")
        return add_corners, key
    
    def run_capture_deterministic_count(self):
        skip_interval = int(self.total_frames / self.args.frame_count_target)

        continue_capture = 1
        for camera in self.cameras:
            #init capture
            camera.read_next_frame()
            continue_capture &= camera.more_frames_remain
            
        usable_frame_ct = 0
        i_start_frame = 0
        i_frame = 0
         
        report_interval = 10
        
        while continue_capture:
            add_corners = False
            i_frame = i_start_frame
            i_end_frame = i_start_frame + skip_interval
            for camera in self.cameras:
                camera.scroll_to_frame(i_frame)
            while not add_corners and i_frame < i_end_frame and continue_capture:
                add_corners = self.__automatic_filter()
                if self.args.manual_filter:
                    add_corners, key = self.filter_frame_manually()
                      
                if add_corners:
                    usable_frame_ct += 1
                    self.add_corners(usable_frame_ct, report_interval, i_frame)
                    #log last usable **filtered** frame
                    for camera in self.cameras:
                        camera.set_previous_to_current()
  
                i_frame += 1
                continue_capture = 1
                for camera in self.cameras:
                    camera.read_next_frame()
                    continue_capture &= camera.more_frames_remain
                continue_capture &= (not (self.args.manual_filter and key == 27))
            i_start_frame = i_end_frame
            
        if self.args.manual_filter and key == 27:
            continue_capture = False
        if self.args.manual_filter:
            cv2.destroyAllWindows()
        return usable_frame_ct
            
    def run_capture(self):
        continue_capture = 1
        for camera in self.cameras:
            #just in case we're running capture again
            camera.scroll_to_beginning()
            #init capture
            camera.read_next_frame()
            continue_capture &= camera.more_frames_remain
        
        report_interval = 10
        i_frame = 0
        usable_frame_ct = 0

        while(continue_capture):
            if not self.args.frame_numbers or i_frame in self.frame_numbers:
                add_corners = self.__automatic_filter()
                
                if self.args.manual_filter:
                    add_corners, key = self.filter_frame_manually()
                      
                if add_corners:
                    usable_frame_ct += 1
                    self.add_corners(usable_frame_ct, report_interval, i_frame)
                    
                    #log last usable **filtered** frame
                    for camera in self.cameras:
                        camera.set_previous_to_current()
      
            i_frame += 1
            continue_capture = 1
            for camera in self.cameras:
                camera.read_next_frame()
                continue_capture &= camera.more_frames_remain
            continue_capture &= (not (self.args.manual_filter and key == 27))
            
        if self.args.manual_filter:
            cv2.destroyAllWindows()
        return usable_frame_ct
        
    def gather_frame_data(self):
        self.objpoints = []
        print("Gathering frame data...")
        usable_frame_ct = 0
        if(self.args.load_corners):
            print("Loading corners from {0:s}".format(self.full_corners_path))    
            imgpoints, self.objpoints, usable_frame_ct =\
            cio.load_corners(self.full_corners_path)
            usable_frame_ct = len(self.objpoints)
            for camera in self.cameras:
                camera.imgpoints = imgpoints[camera.index]
        else:
            if(self.args.load_images):
                usable_frame_ct = self.load_frame_images()
            elif(self.args.frame_count_target != -1):
                usable_frame_ct = self.run_capture_deterministic_count()
            else:
                usable_frame_ct = self.run_capture()            
            if self.args.save_corners:
                print("Saving corners to {0:s}".format(self.full_corners_path))
                file_dict = {}
                for camera in self.cameras:
                    file_dict["imgpoints"+str(camera.index)] = camera.imgpoints
                file_dict["object_point_set"]=self.board_object_corner_set
                np.savez_compressed(self.full_corners_path,**file_dict)
                
        print ("Total usable frames: {0:d} ({1:.3%})"
               .format(usable_frame_ct, float(usable_frame_ct)/self.total_frames))
        self.usable_frame_count = usable_frame_ct
               
    def run_calibration(self):
        min_frames = CalibrateVideoApplication.min_frames_to_calibrate
        if self.usable_frame_count < min_frames:
            print("Not enough usable frames to calibrate."+
                  " Need at least {0:d}, got {1:d}".format(min_frames,self.usable_frame_count))
            return
        print ("Calibrating for max. {0:d} iterations...".format(self.args.max_iterations))
        
        if len(self.cameras) > 1:
            calibration_result = cutils.stereo_calibrate(self.cameras[0].imgpoints, 
                                                         self.cameras[1].imgpoints, 
                                                         self.objpoints, self.frame_dims, 
                                                         self.args.use_fisheye_model, 
                                                         self.args.use_rational_model, 
                                                         self.args.use_tangential_coeffs, 
                                                         self.args.precalibrate_solo, 
                                                         self.args.max_iterations, 
                                                         self.path_to_calib_file )
            if self.args.preview:
                l_im = cv2.imread(osp.join(self.args.folder,self.args.preview_files[0]))
                r_im = cv2.imread(osp.join(self.args.folder,self.args.preview_files[1]))
                l_im, r_im = cutils.generate_preview(calibration_result, l_im, r_im)
                path_l = osp.join(self.args.folder,self.args.preview_files[0][:-4] + "_rect.png")
                path_r = osp.join(self.args.folder,self.args.preview_files[1][:-4] + "_rect.png")
                cv2.imwrite(path_l, l_im)
                cv2.imwrite(path_r, r_im)
        else:
            flags = 0
            if self.path_to_calib_file != None:
                result = cio.load_opencv_stereo_calibration(self.path_to_calib_file)
                if(self.frame_dims != result.resolution):
                    raise ValueError("Resolution in specified calibration file (" + 
                                     self.path_to_calib_file + ") does not correspond to given resolution.")
                flags += cv2.CALIB_USE_INTRINSIC_GUESS
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, self.args.max_iterations, 
                        2.2204460492503131e-16)
            if not self.args.use_tangential_coeffs:
                flags += cv2.CALIB_ZERO_TANGENT_DIST
            if self.args.use_rational_model:
                flags += cv2.CALIB_RATIONAL_MODEL
            calibration_result = cutils.calibrate(self.objpoints, self.camera.imgpoints, flags, 
                                                  criteria, self.camera.calib)
        if not self.args.skip_printing_output:
            print(calibration_result)
        if not self.args.skip_saving_output:
            cio.save_opencv_stereo_calibration(osp.join(self.args.folder,self.args.output), calibration_result)