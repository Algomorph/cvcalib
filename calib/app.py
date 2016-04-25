"""
 calib.app.py

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
"""

from abc import ABCMeta
import os
import os.path as osp

from common.app import VideoProcessingApplication


class CalibrationApplication(VideoProcessingApplication):
    """
        Base-level abstract Calibration Application class. Contains routines shared
        by all calibration applications.
    """
    __metaclass__ = ABCMeta

    def __init__(self, args):
        """
        Base constructor
        """
        super().__init__(args)
        self.full_frame_folder_path = osp.join(args.folder, args.filtered_image_folder)
        # if image folder (for captured frames) doesn't yet exist, create it
        if args.save_images and not os.path.exists(self.full_frame_folder_path):
            os.makedirs(self.full_frame_folder_path)


