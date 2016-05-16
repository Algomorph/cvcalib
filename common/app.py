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
import numpy as np
import os
import os.path as osp
import re
import datetime
import cv2
from calib.camera import Camera
from calib.rig import Rig, MultiStereoRig
from calib.video import Video
from calib.io import load_opencv_calibration


class VideoProcessingApplication(object):
    """
        Base-level abstract Calibration Application class. Contains routines shared
        by all calibration applications.
    """
    __metaclass__ = ABCMeta

    def __init__(self, args):
        """
        Base constructor
        """
        self.args = args

        self.aux_data_file = {}
        # load or initialize auxiliary data
        if args.aux_data_file is not None:
            self.aux_data_path = os.path.join(self.args.folder, self.args.aux_data_file)
            if osp.isfile(self.aux_data_path):
                npz_archive = np.load(osp.join(args.folder, args.aux_data_file))
                # convert to dict
                for key in npz_archive:
                    self.aux_data_file[key] = npz_archive[key]

        # set up board (3D object points of checkerboard used for calibration)
        self.object_points = []
        self.board_dims = (args.board_width, args.board_height)
        self.board_object_corner_set = np.zeros((args.board_height * args.board_width, 1, 3), np.float32)
        self.board_object_corner_set[:, :, :2] = np.indices(self.board_dims).T.reshape(-1, 1, 2)
        self.board_object_corner_set *= args.board_square_size

        self.pixel_difference_factor = 1.0 / (self.board_dims[0] * self.board_dims[1] * 3 * 256.0)

        # some vars set to default
        self.criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

        if args.output is None:
            args.output = "intrinsics{0:s}.xml".format(re.sub(r"-|:", "",
                                                              str(datetime.datetime.now())[:-7])
                                                       .replace(" ", "-"))

        self.videos = [Video(os.path.join(args.folder, video_filename)) for video_filename in args.videos]
        if args.input_calibration is not None:
            intrinsic_arr = []
            # load calibration files
            initial_calibration = []
            for calib_file in args.input_calibration:
                initial_calibration.append(load_opencv_calibration(os.path.join(args.folder, calib_file)))

            for calibration_info in initial_calibration:
                if type(calibration_info) == MultiStereoRig:
                    for rig in calibration_info.rigs:
                        for camera in rig.cameras:
                            intrinsic_arr.append(camera.intrinsics)
                elif type(calibration_info) == Rig:
                    for camera in calibration_info.cameras:
                        intrinsic_arr.append(camera.intrinsics)
                elif type(calibration_info) == Camera:
                    intrinsic_arr.append(calibration_info.intrinsics)
                elif type(calibration_info) == Camera.Intrinsics:
                    intrinsic_arr.append(calibration_info)
                else:
                    raise RuntimeError("Unsupported calibration file format.")

            if len(intrinsic_arr) != len(args.videos):
                raise ValueError("The total number of intrinsics in all the provided input calibration files " +
                                 "combined ({:d}) does not equal the total number provided of video file paths ({:d})." +
                                 "These numbers must match."
                                 .format(len(intrinsic_arr), len(args.videos)))

            self.cameras = [Camera(intrinsics=intrinsics) for intrinsics in intrinsic_arr]
            if len(initial_calibration) == 1 and (type(initial_calibration[0]) == Rig or
                                                  type(initial_calibration[0]) == MultiStereoRig):
                self.rig = initial_calibration[0]
            else:
                self.rig = Rig(tuple(self.cameras))
        else:
            self.cameras = [Camera(resolution=video.frame_dims) for video in self.videos]
            self.rig = Rig(tuple(self.cameras))
