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
from calib.data import Video, StereoExtrinsics, CameraIntrinsics
from calib.calib_app import CalibApplication
import calib.io as cio
import os.path
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
        ix_video = 0
        self.videos = []
        #load videos
        for video in args.videos: 
            self.videos.append(Video(args.folder, video, ix_video))
            ix_video +=1
            
        for calib_file in args.input_calibration:
            self.initial_calibration = cio.load_opencv_calibration(os.path.join(args.folder, calib_file))
            
        #sanity checks & mode
        if(type(self.initial_calibration[0]) == StereoExtrinsics):
            self.mode = Mode.multiview_stereo
            for calib_info in self.initial_calibration:
                if (type(calib_info) != StereoExtrinsics):
                    raise TypeError("All calibration files should be of the same type. Expecting: " 
                                    + str(StereoExtrinsics) + ". Got: " + str(type(calib_info)))
        else:
            self.mode = Mode.multiview_separate
            for calib_info in self.initial_calibration:
                if (type(calib_info) != CameraIntrinsics):
                    raise TypeError("All calibration files should be of the same type. Expecting: " 
                                    + str(CameraIntrinsics) + ". Got: " + str(type(calib_info)))
    
    
        
    def calculate_transform_pairs(self, verbose = True):
        '''
        Find camera positions at each filtered frame AND the next usable frame
        '''
        camera_transforms_sets = []
        for video in self.videos:
            camera_transforms = []
            for ix_pos in range(len(video.imgpoints)):
                #find camera rotation and translation for the filtered frame
                source_frame_number = self.frame_numbers[ix_pos]
                imgpoints = video.imgpoints[ix_pos]
                objpoints = self.objpoints[ix_pos]
                retval, rvec, tvec = cv2.sovlePnP(objpoints, imgpoints, video.calib.intrinsic_mat, 
                                                    video.calib.distortion_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                source_frame_transform = (rvec, tvec)
                if(retval):
                    #find camera rotation and translation at the next frame
                    video.scroll_to_frame(source_frame_number+1)
                    video.read_next_frame()
                    if(self.__automatic_filter_basic()):
                        grey_frame = cv2.cvtColor(video.frame,cv2.COLOR_BGR2GRAY)
                        cv2.cornerSubPix(grey_frame, video.current_corners, (11,11),(-1,-1),self.criteria_subpix)
                        retval, rvec, tvec = cv2.sovlePnP(objpoints, video.current_corners, video.calib.intrinsic_mat, 
                                                    video.calib.distortion_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                        target_frame_transform = (rvec,tvec)
                        camera_transforms.append((source_frame_transform,target_frame_transform))
            if(verbose):
                print("Added {0:d} usable transform pairs for video {1:s}".format(len(camera_transforms), video.name))
            camera_transforms_sets.append(camera_transforms)
        return camera_transforms_sets
    
                
    def perform_time_sliding_adjustment(self, verbose = True):
        '''
        This routine will take care of 1-frame time offset between cameras that were not synchronized 
        or not genlocked. It is not currently completed. 
        '''
        #TODO: finish later if need be
        frame_duration = 1.0 / self.video[1].fps
        camera_transorm_sets = self.calculate_transform_pairs()