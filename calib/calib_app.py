'''
 calib_app

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

from abc import ABCMeta
from calib.video import Video
import numpy as np
import os
import os.path as osp
import re, datetime
import cv2

class CalibApplication(object):
    __metaclass__ = ABCMeta
    '''
    Base-level abstract Calibration Application class. Contains routines shared
    by all calibration applications.
    '''


    def __init__(self, args):
        '''
        Base constructor
        '''
        self.args = args
        self.video = Video(args.folder,args.videos[0], 0)
        self.frame_dims = self.video.frame_dims
        
        self.frame_numbers = None
        
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
        
        self.pixel_difference_factor = 1.0 / (self.board_dims[0] * self.board_dims[1] * 3 * 256.0)
        
        if(args.output is None):
            args.output = "calib{0:s}.xml".format(re.sub(r"-|:","",
                                                         str(datetime.datetime.now())[:-7])
                                                  .replace(" ","-"))
            
