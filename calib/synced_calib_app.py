'''
Created on Feb 4, 2016
@author: Gregory Kramida
@licence: Apache v2

Copyright 2016 Gregory Kramida

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
import os
import os.path as osp
import cv2#@UnresolvedImport
import numpy as np
import calib.utils as cutils
from calib import io as cio
from calib.camera import Camera
from calib.data import StereoRig
import sys
import re
import common.filter as cf 
from calib.calib_app import CalibApplication


class SyncedCalibApplication(CalibApplication):
    '''
    Application class for calibration of single cameras or genlocked stereo cameras
    '''
    min_frames_to_calibrate = 4
    def __init__(self,args):
        CalibApplication.__init__(self, args)
        
        self.frame_numbers = []
        if args.frame_numbers:
            path = osp.join(args.folder, args.frame_numbers)
            print("Loading frame numbers from \"{0:s}\"".format(path))
            npzfile = np.load(path)
            self.frame_numbers = set(npzfile["frame_numbers"])
            
        #load videos
        if(len(args.videos) > 1):
            self.videos = [self.video, Camera(args.folder, args.videos[1], 1)]
            self.__automatic_filter_basic = self.__automatic_filter_basic_stereo
            self.__automatic_filter = self.__automatic_filter_stereo
            if(len(args.preview_files) != len(args.videos)):
                raise ValueError("There must be two preview file arguments passed in for stereo calibration.")
            self.total_frames = min(self.videos[0].frame_count,self.videos[1].frame_count)
            if(self.videos[0].frame_dims != self.videos[1].frame_dims):
                raise ValueError("The videos must have the same resolution.")
        else:
            self.videos = [self.video]
            self.__automatic_filter_basic = self.__automatic_filter_basic_mono
            self.__automatic_filter = self.__automatic_filter_mono
            self.total_frames = self.video.frame_count 
        
        #load existing calibration file if need-be
        if(args.input_calibration != None):
            full_path = osp.join(args.folder, args.input_calibration[0])
            self.initial_calibration = cio.load_opencv_calibration(full_path)
            if(len(args.videos) == 1 and type(self.initial_calibration) == StereoRig):
                raise ValueError("Got only one video input, \'{0:s}\', but a stereo calibration "+
                                 "input file '{0:s}'. Please provide a single camera's intrinsics."
                                 .format(self.video.name, args.input_calibration))
        else:
            self.initial_calibration = None


        
    
    def __automatic_filter_stereo(self):
        l_frame = self.videos[0].frame
        lframe_prev = self.videos[0].previous_frame
        r_frame = self.videos[1].frame

        sharpness = min(cv2.Laplacian(l_frame, cv2.CV_64F).var(), 
                        cv2.Laplacian(r_frame, cv2.CV_64F).var())
        verbose = False#set to True for sharpness analysis
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
        
        self.videos[0].current_image_points = lcorners
        self.videos[1].current_image_points = rcorners
        
        return True
    
    def __automatic_filter_mono(self):
        frame = self.video.frame
        frame_prev = self.video.previous_frame
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
        
        self.video.current_image_points = corners
        
        return True
        
    
    def __automatic_filter_basic_stereo(self):
        return cf.filter_basic_stereo(self.videos, self.board_dims)
    
    def __automatic_filter_basic_mono(self):
        return self.video.approximate_corners(self.board_dims)

    def load_frame_images(self):
        '''
        Load images (or image pairs) from self.full_frame_folder_path
        '''
        print("Loading frames from '{0:s}'".format(self.full_frame_folder_path))
        all_files = [f for f in os.listdir(self.full_frame_folder_path) 
                 if osp.isfile(osp.join(self.full_frame_folder_path,f)) and f.endswith(".png")]
        all_files.sort()
        
        usable_frame_ct = sys.maxsize
        
        frame_number_sets = []
        
        for video in self.videos:
            #assume matching numbers in corresponding left & right files
            files = [f for f in all_files if f.startswith(video.name)]
            files.sort()#added to be explicit
            
            cam_frame_ct = 0
            frame_numbers = []
            for ix_pair in range(len(files)):
                #TODO: assumes there is the same number of frames for all videos, and all frame
                #indexes match
                frame = cv2.imread(osp.join(self.full_frame_folder_path,files[ix_pair]))
                frame_number = int(re.search(r'\d\d\d\d',files[ix_pair]).group(0))
                frame_numbers.append(frame_number)
                found,lcorners = cv2.findChessboardCorners(frame,self.board_dims)
                if not found:
                    raise ValueError("Could not find corners in image '{0:s}'".format(files[ix_pair]))
                grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(grey, lcorners, (11,11),(-1,-1),self.criteria_subpix)
                video.imgpoints.append(lcorners)
                cam_frame_ct+=1
            usable_frame_ct = min(usable_frame_ct,cam_frame_ct)
            frame_number_sets.append(frame_numbers)
        
        if(len(self.videos) > 1):
            #check that all videos have the same frame number sets
            if(len(frame_number_sets[0]) != len(frame_number_sets[1])):
                raise ValueError("There are some non-paired frames in folder '{0:s}'".format(self.full_frame_folder_path))
            for i_fn in range(len(frame_number_sets[0])):
                fn0 = frame_number_sets[0][i_fn]
                fn1 = frame_number_sets[1][i_fn]
                if(fn0 != fn1):
                    raise ValueError("There are some non-paired frames in folder '{0:s}'."+
                                     " Check frame {1:d} for video {2:s} and frame {3:d} for video {4:s}."
                                     .format(self.full_frame_folder_path, 
                                             fn0, self.videos[0].name,
                                             fn1, self.videos[1].name))
            
        self.frame_numbers = frame_number_sets[0]
        
        for i_frame in range(usable_frame_ct):#@UnusedVariable
            self.objpoints.append(self.board_object_corner_set)
        return usable_frame_ct
    
    def add_corners_for_all(self, usable_frame_ct, report_interval, i_frame):
        if(usable_frame_ct % report_interval == 0):
            print ("Usable frames: {0:d} ({1:.3%})"
                   .format(usable_frame_ct, float(usable_frame_ct)/(i_frame+1)))
            
        for video in self.videos:
            video.add_corners(i_frame, self.criteria_subpix, 
                              self.full_frame_folder_path, self.args.save_images)
        self.frame_numbers.append(i_frame)
        self.objpoints.append(self.board_object_corner_set)
    
    def filter_frame_manually(self):
        if len(self.videos) == 2:
            display_image = np.hstack((self.videos[0].frame, self.videos[1].frame))
        else:
            display_image = self.videos[0].frame
        cv2.imshow("frame", display_image)
        key = cv2.waitKey(0) & 0xFF
        add_corners = (key == ord('a'))
        cv2.destroyWindow("frame")
        return add_corners, key
    
    def run_capture_deterministic_count(self):
        skip_interval = int(self.total_frames / self.args.frame_count_target)

        continue_capture = 1
        for video in self.videos:
            #just in case we're running capture again
            video.clear_results()
            video.scroll_to_beginning()
            #init capture
            video.read_next_frame()
            continue_capture &= video.more_frames_remain
            
        usable_frame_ct = 0
        i_start_frame = 0
        i_frame = 0
         
        report_interval = 10
        
        while continue_capture:
            add_corners = False
            i_frame = i_start_frame
            i_end_frame = i_start_frame + skip_interval
            for video in self.videos:
                video.scroll_to_frame(i_frame)
            while not add_corners and i_frame < i_end_frame and continue_capture:
                add_corners = self.__automatic_filter()
                if self.args.manual_filter:
                    add_corners, key = self.filter_frame_manually()
                      
                if add_corners:
                    usable_frame_ct += 1
                    self.add_corners_for_all(usable_frame_ct, report_interval, i_frame)
                    #log last usable **filtered** frame
                    for video in self.videos:
                        video.set_previous_to_current()
  
                i_frame += 1
                continue_capture = 1
                for video in self.videos:
                    video.read_next_frame()
                    continue_capture &= video.more_frames_remain
                continue_capture &= (not (self.args.manual_filter and key == 27))
            i_start_frame = i_end_frame
            
        if self.args.manual_filter and key == 27:
            continue_capture = False
        if self.args.manual_filter:
            cv2.destroyAllWindows()
        return usable_frame_ct
            
    def run_capture(self):
        continue_capture = 1
        for video in self.videos:
            #just in case we're running capture again
            video.clear_results()
            video.scroll_to_beginning()
            #init capture
            video.read_next_frame()
            continue_capture &= video.more_frames_remain
        
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
                    self.add_corners_for_all(usable_frame_ct, report_interval, i_frame)
                    
                    #log last usable **filtered** frame
                    for video in self.videos:
                        video.set_previous_to_current()
      
            i_frame += 1
            continue_capture = 1
            for video in self.videos:
                video.read_next_frame()
                continue_capture &= video.more_frames_remain
            continue_capture &= (not (self.args.manual_filter and key == 27))
            
        if self.args.manual_filter:
            cv2.destroyAllWindows()
        return usable_frame_ct
        
    def gather_frame_data(self):
        self.objpoints = []
        print("Gathering frame data...")
        usable_frame_ct = 0
        if(self.args.load_corners):
            self.board_object_corner_set, frame_numbers =\
            cio.load_corners(self.full_corners_path, self.videos)
            
            if(type(frame_numbers) == type(None)):
                self.frame_numbers = list(self.videos[0].usable_frames.keys())
            else:
                #use legacy frame numbers
                self.frame_numbers = frame_numbers
            usable_frame_ct = len(self.video.imgpoints)
            
            for i_frame in range(usable_frame_ct): # @UnusedVariable
                self.objpoints.append(self.board_object_corner_set)
            
        else:
            if(self.args.load_images):
                usable_frame_ct = self.load_frame_images()
            elif(self.args.frame_count_target != -1):
                usable_frame_ct = self.run_capture_deterministic_count()
            else:
                usable_frame_ct = self.run_capture()            
            if self.args.save_corners:
                cio.save_corners(self.full_corners_path, self.videos, self.board_object_corner_set)
                
        print ("Total usable frames: {0:d} ({1:.3%})"
               .format(usable_frame_ct, float(usable_frame_ct)/self.total_frames))
        self.usable_frame_count = usable_frame_ct
               
    def run_calibration(self):
        min_frames = SyncedCalibApplication.min_frames_to_calibrate
        if self.usable_frame_count < min_frames:
            print("Not enough usable frames to calibrate."+
                  " Need at least {0:d}, got {1:d}".format(min_frames,self.usable_frame_count))
            return
        print ("Calibrating for max. {0:d} iterations...".format(self.args.max_iterations))
        
        if len(self.videos) > 1:
            calibration_result = cutils.stereo_calibrate(self.videos[0].imgpoints, 
                                                         self.videos[1].imgpoints, 
                                                         self.objpoints, self.frame_dims, 
                                                         self.args.use_fisheye_model, 
                                                         self.args.use_rational_model, 
                                                         self.args.use_tangential_coeffs, 
                                                         self.args.precalibrate_solo,
                                                         self.args.stereo_only, 
                                                         self.args.max_iterations, 
                                                         self.initial_calibration )
            if self.args.preview:
                l_im = cv2.imread(osp.join(self.args.folder,self.args.preview_files[0]))
                r_im = cv2.imread(osp.join(self.args.folder,self.args.preview_files[1]))
                l_im, r_im = cutils.generate_preview(calibration_result, l_im, r_im)
                path_l = osp.join(self.args.folder,self.args.preview_files[0][:-4] + "_rect.png")
                path_r = osp.join(self.args.folder,self.args.preview_files[1][:-4] + "_rect.png")
                cv2.imwrite(path_l, l_im)
                cv2.imwrite(path_r, r_im)
                
            self.calibration = calibration_result
            self.videos[0].intrinsics = calibration_result.intrinsics[0]
            self.videos[1].intrinsics = calibration_result.intrinsics[1]
        else:
            calibration_result = cutils.calibrate_wrapper(self.objpoints, self.video,
                                                          self.args.use_rational_model,
                                                          self.args.use_tangential_coeffs,
                                                          self.args.max_iterations,
                                                          self.initial_calibration)
        if not self.args.skip_printing_output:
            print(calibration_result)
        if not self.args.skip_saving_output:
            cio.save_opencv_calibration(osp.join(self.args.folder,self.args.output), 
                                               calibration_result)