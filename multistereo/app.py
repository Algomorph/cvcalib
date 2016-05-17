"""
Created on April 25, 2016
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
"""

from common.app import VideoProcessingApplication
from calib.rig import MultiStereoRig
from calib.utils import compute_stereo_rectification_maps
import cv2


class MultiStereoApplication(VideoProcessingApplication):
    # TODO make this a parameter
    # size factor for undistorted images
    SIZE_FACTOR = 1.8

    def __init__(self, args):
        super().__init__(args)
        # split into two stereo rigs
        if type(self.rig) != MultiStereoRig:
            raise ValueError("The multi-stereo application expects MultiStereoRig as input_calibration")

        if len(self.videos)*2 != len(self.rig.rigs):
            raise ValueError("The number of videos doesn't correspond to the number of cameras in the rig." +
                             "Please check the input_calibration file(s) and the 'videos' argument.")

        # associate the videos with the cameras
        ix_vid = 0
        for rig in self.rig.rigs:
            for cam in rig.cameras:
                cam.video = self.videos[ix_vid]
                ix_vid += 1

        # assume the videos are synced
        self.total_frames = min([video.frame_count for video in self.videos])

        self.init_undistortion()
        self.init_stereo_matcher()

    def init_undistortion(self):
        for rig in self.rig.rigs:
            map1x, map1y, map2x, map2y = compute_stereo_rectification_maps(rig, self.videos[0].frame_dims,
                                                                           MultiStereoApplication.SIZE_FACTOR)
            rig.cameras[0].map_x = map1x
            rig.cameras[0].map_y = map1y
            rig.cameras[1].map_x = map2x
            rig.cameras[1].map_y = map2y

    def init_stereo_matcher(self):
        # TODO
        pass

    def compute_depth(self):
        """
        Compute depth map at each frame and save these as separate video
        @return:
        """
        for video in self.videos:
            video.reopen()

        for ix_frame in range(0, self.total_frames):
            for rig in self.rig.rigs:
                lcam = rig.cameras[0]
                rcam = rig.cameras[1]
                lvid = lcam.video
                rvid = rcam.video
                lvid.read_next_frame()
                rvid.read_next_frame()
                lframe = lcam.rectify_image(lvid.frame)
                rframe = rcam.rectify_image(rvid.frame)









