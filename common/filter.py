'''
/home/algomorph/Factory/calib_video_opencv/common/filter.py.
Created on Feb 9, 2016.
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

import cv2#@UnresolvedImport 

def filter_basic_stereo(videos, board_dims):
    l_frame = videos[0].frame
    r_frame = videos[1].frame
    
    lfound,lcorners = cv2.findChessboardCorners(l_frame,board_dims)
    rfound,rcorners = cv2.findChessboardCorners(r_frame,board_dims)
    if not (lfound and rfound):
        return False
    
    videos[0].current_corners = lcorners
    videos[1].current_corners = rcorners
    
    return True

def filter_basic_mono(video, board_dims):
    frame = video.frame
    found,corners = cv2.findChessboardCorners(frame,board_dims)  
    video.current_corners = corners
    return found