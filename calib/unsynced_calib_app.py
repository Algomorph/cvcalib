'''
 unsynced_calib_app

       Authors: Gregory Kramida
   Copyright: (c) Gregory Kramida 2016 

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
import cv2#@UnresolvedImport
from calib.data import StereoRig, CameraIntrinsics
from calib.video import Video
from calib.calib_app import CalibApplication
import calib.io as cio
import os.path
import time
import numpy as np
from enum import Enum


class Mode(Enum):
    multiview_separate = 0
    multiview_stereo = 1
    

class UnsyncedCalibApplication(CalibApplication):
    '''
    UnsynchedCalibApplication
    '''

    def __init__(self, args):
        '''
        Constructor
        '''
        CalibApplication.__init__(self, args)
        
        self.frame_numbers = {}
        
        intrinsic_arr = [] 
        
        if(args.input_calibration == None or len(args.input_calibration) == 0):
            raise ValueError("Unsynched calibration requires input calibration parameters for all "+
                             "cameras used to take the videos.")
        
        #load calibration files
        for calib_file in args.input_calibration:
            self.initial_calibration =\
            cio.load_opencv_calibration(os.path.join(args.folder, calib_file))
        
        #sanity checks & mode
        if(type(self.initial_calibration[0]) == StereoRig):
            self.mode = Mode.multiview_stereo
            for calib_info in self.initial_calibration:
                if (type(calib_info) != StereoRig):
                    raise TypeError("All calibration files should be of the same type. Expecting: " 
                                    + str(StereoRig) + ". Got: " + str(type(calib_info)))
                intrinsic_arr += calib_info.intrinsics#aggregate intrinsics into a single array
                
            if(len(self.initial_calibration) != int(len(args.videos)/2)):
                raise ValueError("Number of stereo calibration files is not half the number of videos.")
        else:
            self.mode = Mode.multiview_separate
            for calib_info in self.initial_calibration:
                if (type(calib_info) != CameraIntrinsics):
                    raise TypeError("All calibration files should be of the same type. Expecting: " 
                                    + str(CameraIntrinsics) + ". Got: " + str(type(calib_info)))
            if(len(self.initial_calibration) != len(args.videos)):
                raise ValueError("Number of intrinsics files is does not equal the number of videos.")
            intrinsic_arr = self.initial_calibration 
        
        if args.frame_numbers:
            path = os.path.join(args.folder, args.frame_numbers)
            print("Loading frame numbers from \"{0:s}\"".format(path))
            npzfile = np.load(path)
            for video_filename in args.videos:
                self.frame_numbers.append(set(npzfile["video_filename"]))
        
        ix_video = 0
        self.videos = []
        #load videos
        for video_filename in args.videos: 
            self.videos.append(Video(args.folder, video_filename, ix_video))
            ix_video +=1
    
    def run_capture(self,verbose = True):
        '''
        Run capture for each video separately
        '''
        report_interval = 5.0 #seconds
        i_frame = 0
        
        for video in self.videos:
            if(verbose):
                print("Capturing calibration board points for video {0:s}".format(video.name))
            
            #just in case we're running capture again
            video.clear_results()
            video.scroll_to_beginning()
            #init capture
            video.read_next_frame()
            key = 0
            check_time = time.time()
            
            while(video.more_frames_remain and (not (self.args.manual_filter and key == 27))):
                if not self.args.frame_numbers or i_frame in self.frame_numbers:
                    #TODO: add blur filter support to video class
                    add_corners = video.approximate_corners(video, self.board_dims)
                    
                    if self.args.manual_filter:
                        add_corners, key = video.filter_frame_manually()
                          
                    if add_corners:
                        video.add_corners(video, i_frame, self.criteria_subpix, 
                                          self.full_frame_folder_path, self.args.save_images)
                        video.find_current_pose(self.board_object_corner_set)
                        
                        #log last usable **filtered** frame
                        for video in self.videos:
                            video.set_previous_to_current()
                #fixed time interval reporting
                if(verbose and time.time() - check_time > report_interval):
                    print("Processed {:.0%}".format(video.frame_count / i_frame))
                    check_time += report_interval
                i_frame += 1
            if verbose: print(" Done.")
            
        if self.args.manual_filter:
            cv2.destroyAllWindows()
            
    def calibrate_time_brute_force(self):
        #for only two cameras in this first version
        #assume frames are contiguous for this version
        max_offset = self.args.max_frame_offset
        vid0 = self.videos[0]
        vid1 = self.videos[1]
        
        frame_range_max = min((vid0.frame_count, vid1.frame_count-max_offset))
        frame_range_min = max_offset
        
        if(vid1.frame_count < 2 * max_offset + 1):
            raise ValueError("Not enough frames for offset finding")
         
        distance_data = []
        med_poses = []
        
        for offset in range(-max_offset,max_offset):
            #distance between cameras:
            distance_set = []
            frame_numbers = []
            for i_frame_cam0 in range(frame_range_min,frame_range_max):
                i_frame_cam1 = i_frame_cam0 + offset
                if(i_frame_cam0 in vid0.usable_frames and i_frame_cam1 in vid1.usable_frames):
                    t0 = vid0.poses[vid0.usable_frames[i_frame_cam0]].tvec
                    t1 = vid1.poses[vid1.usable_frames[i_frame_cam1]].tvec
                    distance_set.append(cv2.norm(t0, t1, cv2.NORM_L1))
                    frame_numbers.append(i_frame_cam0)
            distance_set = np.array(distance_set)    
            dist_var = distance_set.var()
            ixs = np.argsort(distance_set)
            mid = int(len(distance_set)/2)
            med_fn = frame_numbers[ixs][mid]
            t0 = vid0.poses[vid0.usable_frames[med_fn]].tvec
            t1 = vid1.poses[vid1.usable_frames[med_fn+offset]].tvec
            med_poses.append((t0,t1)) 
            distance_data.append([dist_var, len(distance_set)])
            
        distance_data = np.array(distance_data)
        #sort by number of usable frames
        ixs = np.argsort(distance_data[:,1])
        distance_data = distance_data[ixs]
        med_poses = med_poses[ixs]
        #take top 90% by number of usable frames
        start_from = int(0.1 * distance_data.shape[0])
        distance_data = distance_data[start_from,:,:]
        med_poses = med_poses[start_from,:,:]
        #take one with smallest variance
        ix = np.argmax(distance_data[:,0])
        
        return distance_data[ix][0], distance_data[ix][1], med_poses[ix]
        
    def gather_frame_data(self, verbose = True):
        if(self.args.load_corners):
            self.board_object_corner_set = cio.load_corners(self.full_corners_path, self.videos, 
                                                            verbose=verbose)[0]
        else:
            self.run_capture(verbose)
            if(self.args.save_corners):
                cio.save_corners(self.full_corners_path, self.videos, self.board_object_corner_set)
                    
             
        
        