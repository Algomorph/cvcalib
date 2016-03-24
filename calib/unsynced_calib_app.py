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
from calib.utils import stereo_calibrate
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
        self.initial_calibration = []
        for calib_file in args.input_calibration:
            self.initial_calibration.append(cio.load_opencv_calibration(os.path.join(args.folder, calib_file)))
        
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
            self.videos.append(Video(args.folder, video_filename, ix_video, intrinsics=intrinsic_arr[ix_video]))
            ix_video +=1
            
    def __terminate_still_streak(self, streak, longest_streak, still_streaks, verbose = True):
        min_still_streak = self.args.max_frame_offset*2+1
        #min_still_streak = 30 
        
        if(len(streak) >= min_still_streak):
            still_streaks.append(streak)
        else:
            return [0,0], longest_streak
        if(len(streak) > len(longest_streak)):
            longest_streak = streak
            if(verbose):
                print("Longest consecutive streak with calibration board "
                      +"staying still relative to camera: {:d}".format(streak[1]-streak[0]))
        return [0,0], longest_streak

    def run_capture(self,verbose = True):
        '''
        Run capture for each video separately
        '''
        report_interval = 5.0 #seconds
        
        for video in self.videos:
            i_frame = 0
            if(verbose):
                print("Capturing calibration board points for video {0:s}".format(video.name))
            
            #just in case we're running capture again
            video.clear_results()
            video.scroll_to_beginning()
            #init capture
            video.read_next_frame()
            key = 0
            check_time = time.time()
            longest_still_streak = []
            still_streak = [0,0]
            still_streaks = []
            while(video.more_frames_remain and (not (self.args.manual_filter and key == 27))):
                add_corners = False
                if not self.args.frame_numbers or i_frame in self.frame_numbers:
                    #TODO: add blur filter support to video class
                    add_corners = video.approximate_corners(self.board_dims)
                    
                    if self.args.manual_filter:
                        add_corners, key = video.filter_frame_manually()
                          
                    if add_corners:
                        video.add_corners(i_frame, self.criteria_subpix, 
                                          self.full_frame_folder_path, self.args.save_images)
                        video.find_current_pose(self.board_object_corner_set)
                        
                        #log last usable **filtered** frame
                        video.set_previous_to_current()
                        
                        cur_corners = video.imgpoints[len(video.imgpoints) - 1]
                        if(len(still_streak)>0):
                            prev_corners = video.imgpoints[len(video.imgpoints) - 2]
                            mean_px_dist_to_prev = (((cur_corners - prev_corners)**2).sum(axis=2)**.5).mean()
                            if(mean_px_dist_to_prev < 0.5):
                                still_streak[1] = i_frame
                            else:
                                still_streak, longest_still_streak =\
                                self.__terminate_still_streak(still_streak, longest_still_streak, still_streaks, verbose)
                        else:
                            still_streak[0] = i_frame
                            
                if(not add_corners):
                    still_streak, longest_still_streak =\
                    self.__terminate_still_streak(still_streak, longest_still_streak, still_streaks, verbose)
                    
                video.read_next_frame()
                #fixed time interval reporting
                if(verbose and time.time() - check_time > report_interval):
                    print("Processed {:.2%}".format(i_frame / video.frame_count))
                    check_time += report_interval
                i_frame += 1
            
            #in case the last frame was also added
            still_streak, longest_still_streak =\
            self.__terminate_still_streak(still_streak, longest_still_streak, still_streaks, verbose)
            video.still_streaks = [tuple(streak) for streak in still_streaks]    
            if verbose: 
                print(" Done. Found {:d} usable frames.".format(len(video.usable_frames)))
            
        if self.args.manual_filter:
            cv2.destroyAllWindows()
            
    def __aux_streak_within(self, source_streak, target_streak):
        source_middle = source_streak[0] + (source_streak[1] - source_streak[0])//2 
        return (((target_streak[0] <= source_streak[0] <= target_streak[1]) or
                 (target_streak[0] <= source_streak[1] <= target_streak[1])) and
                (target_streak[0] <= source_middle <= target_streak[1]))
            
    def match_still_streaks(self, verbose = True):
        '''
        look for streaks in target streak set that overlap at or within the bounds of each
        streak within the source streak set
        '''
        if(verbose):
            print("For each pair of videos, looking for timewise-overlapping"
                  +"streaks of frames where calibration board is not moving.")
        for vid in self.videos:
            vid.still_streak_overlaps = {} 
        #TODO: potenitally reduce to processing only one source video
        for i_vid in range(len(self.videos)):
            source_streaks = self.videos[i_vid].still_streaks
            for j_vid in range(i_vid+1, len(self.videos)):
                target_streaks = self.videos[j_vid].still_streaks
                overlaps = []
                for source_streak in source_streaks:
                    found = False
                    ix_target_streak = 0
                    while(not found and ix_target_streak < len(target_streaks)):
                        target_streak = target_streaks[ix_target_streak]
                        if(self.__aux_streak_within(source_streak, target_streak)):
                            overlaps.append((source_streak),(target_streak))
                        ix_target_streak += 1
                self.videos[i_vid].still_streak_overlaps[j_vid] = overlaps

    def stereo_calibrate_stills(self, verbose = True):
        source_vid = self.videos[0]
        #cut at least this number of frames off the range bounds, because
        #some frames in the beginning or end of the ranges will have some minor board motion
        cutoff = 1
        for j_vid in range(len(self.videos)):
            target_vid = self.videos[j_vid]
            overlaps = source_vid.still_streak_overlaps[j_vid]
            
            imgpts_src = []
            imgpts_tgt = []
            for overlap in overlaps:
                source_range = overlap[0]
                target_range = overlap[1]
                
                src_range_len = source_range[1]-source_range[0]
                tgt_range_len = target_range[1]-target_range[0]
                tgt_half_len = tgt_range_len//2
                src_half_len = src_range_len//2
                if(src_range_len > tgt_range_len):    
                    increment = 0 if tgt_range_len % 2 == 0 else 1
                    src_mid = source_range[0] + src_half_len
                    source_range = (src_mid - tgt_half_len, src_mid + tgt_half_len + increment)
                else:#target range is smaller than or equal to the source range
                    increment = 0 if src_range_len % 2 == 0 else 1
                    tgt_mid = target_range[0] + src_half_len
                    target_range = (tgt_mid - src_half_len, src_mid + src_half_len + increment)
                    
                for ix_frame in range(source_range[0]+cutoff,source_range[1]-cutoff):
                    imgpts_src.append(source_vid.imgpoints[source_vid.usable_frames[ix_frame]])
                for ix_frame in range(target_range[0]+cutoff,target_range[1]-cutoff):
                    imgpts_tgt.append(target_vid.imgpoints[target_vid.usable_frames[ix_frame]])
                    
            initial_calibration = StereoRig((source_vid.intrinsics, target_vid.intrinsics))

            calibration_result = stereo_calibrate(imgpts_src, 
                                                  imgpts_tgt, 
                                                  self.objpoints, self.frame_dims, 
                                                  self.args.use_fisheye_model, 
                                                  self.args.use_rational_model, 
                                                  self.args.use_tangential_coeffs,
                                                  precalibrate_solo=False,
                                                  stereo_only=False, 
                                                  max_iters=self.args.max_iterations, 
                                                  initial_calibration=initial_calibration)
            
            
    
    def calibrate_time_reprojection(self, verbose = True):
        #TODO: this function is currently all old code borrowed from a function that doesn't really work
        
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
        
        #find distances between cameras at each frame, assuming a specific frame offset
        for offset in range(-max_offset,max_offset):
            if(verbose):
                print("Examining possible offset of {:d} frames.".format(offset))
            distance_set = []
            frame_numbers = []
            for i_frame_cam0 in range(frame_range_min,frame_range_max):
                i_frame_cam1 = i_frame_cam0 + offset
                if(i_frame_cam0 in vid0.usable_frames and i_frame_cam1 in vid1.usable_frames):
                    T0 = vid0.poses[vid0.usable_frames[i_frame_cam0]].T
                    T1 = vid1.poses[vid1.usable_frames[i_frame_cam1]].T
                    transform_10 = T0.dot(np.linalg.inv(T1))
                    transform_01 = T1.dot(np.linalg.inv(T0))
                    #cv.projectPoints(self.board_object_corner_set, )
                    frame_numbers.append(i_frame_cam0)
            distance_set = np.array(distance_set)
            frame_numbers = np.array(frame_numbers)    
            dist_var = distance_set.var()
            dist_mean = distance_set.mean()
            ixs = np.argsort(distance_set)
            mid = int(len(distance_set)/2)
            med_fn = frame_numbers[ixs][mid]
            t0 = vid0.poses[vid0.usable_frames[med_fn]].tvec
            t1 = vid1.poses[vid1.usable_frames[med_fn+offset]].tvec
            med_poses.append((t0,t1)) 
            distance_data.append([dist_var, len(distance_set), offset, dist_mean])
            
        distance_data = np.array(distance_data)
        np.savez_compressed(os.path.join(self.args.folder, "distance_data.npz"), distance_data=distance_data)
        med_poses = np.array(med_poses)
        #sort by number of usable frames
        ixs = np.argsort(distance_data[:,1])
        distance_data = distance_data[ixs]
        med_poses = med_poses[ixs]
        #take top 90% by number of usable frames
        start_from = int(0.1 * distance_data.shape[0])
        distance_data = distance_data[start_from:,:]
        med_poses = med_poses[start_from:]
        #take one with smallest variance
        ix = np.argmin(distance_data[:,0])
        
        return distance_data[ix][0], int(distance_data[ix][1]), int(distance_data[ix][2]), distance_data[ix][3], med_poses[ix]
        
    def gather_frame_data(self, verbose = True):
        if(self.args.load_corners):
            self.board_object_corner_set = cio.load_corners(self.full_corners_path, self.videos, 
                                                            verbose=verbose)[0]
        else:
            self.run_capture(verbose)
            if(self.args.save_corners):
                cio.save_corners(self.full_corners_path, self.videos, self.board_object_corner_set)
                    
             
        
        