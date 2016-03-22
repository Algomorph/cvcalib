'''
/home/algomorph/Factory/calib_video_opencv/calib/video.py.
Created on Mar 21, 2016.
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
import os.path as osp
import cv2
from calib.data import CameraIntrinsics
import numpy as np
from enum import Enum

#TODO: figure out how to deal with the filters
class Filter(Enum):
    flip_180 = 0
    
def string_list_to_filter_list(string_list):
    filter_list = []
    for item in string_list:
        if not item in Filter._member_map_:
            raise ValueError("'{:s}' does not refer to any existing filter. "+
                             "Please, use one of the following: {:s}"
                             .format(item, str(Filter._member_names_)))
        else:
            filter_list.append(Filter[item])
    return filter_list

class Pose(object):
    def __init__(self, T, T_inv = None, rvec = None, tvec = None):
        self.T = T
        if(type(tvec) == type(None)):
            tvec = T[0:3,3]
        if(type(rvec) == type(None)):
            R = T[0:3,0:3]
            rvec = cv2.Rodrigues(R)[0]
        if(type(T_inv) == type(None)):
            R = cv2.Rodrigues(rvec)[0]
            R_inv = R.T
            tvec_inv = -R_inv.dot(tvec)
            T_inv = np.vstack((np.append(R_inv,tvec_inv,1),[0,0,0,1]))
        
        self.tvec = tvec
        self.rvec = rvec
        self.T_inv = T_inv
        

class Video(object):
    '''
    Represents a video object, a simple convenience wrapper around OpenCV's video_capture
    '''
    def __init__(self, directory, filename, index = 0, calib = None, filters = []):
        '''
        Build a camera from the specified file at the specified directory
        '''
        self.index = index
        self.cap = None
        #self.filters = string_list_to_filter_list(filters)
        if filename[-3:] != "mp4":
            raise ValueError("Specified file does not have .mp4 extension.")
        self.path = osp.join(directory, filename)
        if not osp.isfile(self.path):
            raise ValueError("No video file found at {0:s}".format(self.path))
        self.name = filename[:-4]
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError("Could not open specified .mp4 file ({0:s}) for capture!".format(self.path))
        #TODO: refactor to image_points
        self.imgpoints = []
        self.frame_dims = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if(calib == None):
            self.calib = CameraIntrinsics(self.frame_dims, index = index)
        if(self.cap.get(cv2.CAP_PROP_MONOCHROME) == 0.0):
            self.n_channels = 3
        else:
            self.n_channels = 1
        self.current_image_points = None
        self.frame = np.zeros((self.frame_dims[0],self.frame_dims[1],self.n_channels), np.uint8)
        self.previous_frame = np.zeros((self.frame_dims[0],self.frame_dims[1],self.n_channels), np.uint8)
        self.more_frames_remain = True
        self.poses = []
        self.usable_frames = {}
        
    def clear_results(self):
        self.poses = []
        self.imgpoints = []
        self.usable_frames = {}
        
    
    def read_next_frame(self):
        self.more_frames_remain, self.frame = self.cap.read()
    
    def read_previous_frame(self):
        '''
        For traversing the video backwards.
        '''
        cur_frame_ix = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if(cur_frame_ix == 0):
            self.more_frames_remain = False
            self.frame = None
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,cur_frame_ix - 1)#@UndefinedVariable
        self.more_frames_remain = True
        self.frame = self.cap.read()[1]
        
    def set_previous_to_current(self):
        self.previous_frame = self.frame
        
    def scroll_to_frame(self,i_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,i_frame)#@UndefinedVariable
    
    def scroll_to_beginning(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0.0)#@UndefinedVariable
    
    def scroll_to_end(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.frame_count-1)#@UndefinedVariable
        
    def __del__(self):
        if self.cap != None:
            self.cap.release()
            
    def approximate_corners(self, board_dims):
        found, corners = cv2.findChessboardCorners(self.frame,board_dims)  
        self.current_image_points = corners
        return found
    
    def find_current_pose(self, object_points):
        '''
        Find camera pose relative to object using current image point set, 
        object_points are treated as world coordinates
        '''
        retval, rvec, tvec = cv2.sovlePnPRansac(object_points, video.current_image_points, #@UndefinedVariable
                                                self.calib.intrinsic_mat, self.calib.distortion_coeffs, 
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        if(retval):
            R = cv2.Rodrigues(rvec)[0]
            T = np.vstack((np.append(R,tvec,1),[0,0,0,1]))
            R_inv = R.T
            tvec_inv = -R_inv.dot(tvec)
            T_inv = np.vstack((np.append(R_inv,tvec_inv,1),[0,0,0,1]))
            self.poses.append(Pose(T,T_inv,rvec,tvec))
        else:
            self.poses.append(None)
        return retval
            
    def add_corners(self, i_frame, criteria_subpix, frame_folder_path, save_image):
        grey_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(grey_frame, self.current_image_points, (11,11),(-1,-1),criteria_subpix)
        if(save_image):
            fname = (osp.join(frame_folder_path,
                           "{0:s}{1:04d}{2:s}".format(self.name,i_frame,".png")))
            cv2.imwrite(fname, self.frame)
        self.usable_frames[i_frame] = len(self.imgpoints)
        self.imgpoints.append(self.current_image_points)
        
        
    def filter_frame_manually(self):
        display_image = self.frame
        cv2.imshow("frame of video {0:s}".format(self.name), display_image)
        key = cv2.waitKey(0) & 0xFF
        add_corners = (key == ord('a'))
        cv2.destroyWindow("frame")
        return add_corners, key