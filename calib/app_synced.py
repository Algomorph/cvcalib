"""
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
"""
import os
import os.path as osp
import cv2
import numpy as np
from calib.utils import calibrate, undistort_stereo
from calib import io as cio
from calib.app import CalibrationApplication
from calib.video import Video
import sys
import re


class ApplicationSynced(CalibrationApplication):
    """
    Application class for calibration of single cameras or genlocked stereo cameras
    """
    min_frames_to_calibrate = 4

    def __init__(self, args):
        super().__init__(args)

        self.usable_frame_count = 0

        self.videos = [Video(os.path.join(args.folder, video_path)) for video_path in args.videos]
        self.video = self.videos[0]
        self.total_frames = min([video.frame_count for video in self.videos])

        if args.input_calibration is None and args.test and len(args.videos) == 1:
            raise ValueError("Expecting an input calibration for the test mode, got none.")

        # TODO: redesign for arbitrary number of videos & cameras
        if len(args.videos) != 1 and len(args.videos) != 2:
            raise ValueError("This calibration tool can only work with single " +
                             "video files or video pairs from synchronized stereo. " +
                             "Provided number of videos: {:d}.".format(len(args.videos)))

    def __automatic_filter_complex(self):
        # compare frame from the first video to the previous **filtered** one from the same video
        pixel_difference = (np.sum(abs(self.videos[0].previous_frame - self.videos[0].frame)) *
                            self.pixel_difference_factor)
        if pixel_difference < self.args.difference_threshold:
            return False
        for video in self.videos:
            if not video.try_approximate_corners_blur(self.board_dims, self.args.sharpness_threshold):
                return False
        return True

    def __automatic_filter_basic(self):
        for video in self.videos:
            if not video.try_approximate_corners(self.board_dims):
                return False
        return True

    def load_frame_images(self):
        """
        Load images (or image pairs) from self.full_frame_folder_path
        """
        print("Loading frames from '{0:s}'".format(self.full_frame_folder_path))
        all_files = [f for f in os.listdir(self.full_frame_folder_path)
                     if osp.isfile(osp.join(self.full_frame_folder_path, f)) and f.endswith(".png")]
        all_files.sort()

        usable_frame_ct = sys.maxsize

        frame_number_sets = []

        for video in self.videos:
            # assume matching numbers in corresponding left & right files
            files = [f for f in all_files if f.startswith(video.name)]
            files.sort()  # added to be explicit

            cam_frame_ct = 0
            frame_numbers = []
            for ix_pair in range(len(files)):
                frame = cv2.imread(osp.join(self.full_frame_folder_path, files[ix_pair]))
                frame_number = int(re.search(r'\d\d\d\d', files[ix_pair]).group(0))
                frame_numbers.append(frame_number)
                found, corners = cv2.findChessboardCorners(frame, self.board_dims)
                if not found:
                    raise ValueError("Could not find corners in image '{0:s}'".format(files[ix_pair]))
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), self.criteria_subpix)
                video.image_points.append(corners)
                video.usable_frames[frame_number] = ix_pair
                cam_frame_ct += 1
            usable_frame_ct = min(usable_frame_ct, cam_frame_ct)
            frame_number_sets.append(frame_numbers)

        if len(self.videos) > 1:
            # check that all cameras have the same frame number sets
            if len(frame_number_sets[0]) != len(frame_number_sets[1]):
                raise ValueError(
                    "There are some non-paired frames in folder '{0:s}'".format(self.full_frame_folder_path))
            for i_fn in range(len(frame_number_sets[0])):
                fn0 = frame_number_sets[0][i_fn]
                fn1 = frame_number_sets[1][i_fn]
                if fn0 != fn1:
                    raise ValueError("There are some non-paired frames in folder '{0:s}'." +
                                     " Check frame {1:d} for camera {2:s} and frame {3:d} for camera {4:s}."
                                     .format(self.full_frame_folder_path,
                                             fn0, self.videos[0].name,
                                             fn1, self.videos[1].name))

        for i_frame in range(usable_frame_ct):
            self.object_points.append(self.board_object_corner_set)
        return usable_frame_ct

    def add_corners_for_all(self, usable_frame_ct, report_interval, i_frame):
        if usable_frame_ct % report_interval == 0:
            print("Usable frames: {0:d} ({1:.3%})"
                  .format(usable_frame_ct, float(usable_frame_ct) / (i_frame + 1)))

        for video in self.videos:
            video.add_corners(i_frame, self.criteria_subpix,
                              self.full_frame_folder_path,
                              self.args.save_images,
                              self.args.save_checkerboard_overlays)
        self.object_points.append(self.board_object_corner_set)

    def filter_frame_manually(self):
        display_image = np.hstack([video.frame for video in self.videos])
        cv2.imshow("frame", display_image)
        key = cv2.waitKey(0) & 0xFF
        add_corners = (key == ord('a'))
        cv2.destroyWindow("frame")
        return add_corners, key

    def run_capture_deterministic_count(self):
        skip_interval = int(self.total_frames / self.args.frame_count_target)

        continue_capture = 1
        for video in self.videos:
            # just in case we're running capture again
            video.clear_results()
            video.scroll_to_beginning()
            # init capture
            video.read_next_frame()
            continue_capture &= video.more_frames_remain

        usable_frame_ct = 0
        i_start_frame = 0
        report_interval = 10

        while continue_capture:
            add_corners = False
            i_frame = i_start_frame
            i_end_frame = i_start_frame + skip_interval
            for video in self.videos:
                video.scroll_to_frame(i_frame)
            while not add_corners and i_frame < i_end_frame and continue_capture:
                add_corners = self.__automatic_filter_complex()
                if self.args.manual_filter:
                    add_corners, key = self.filter_frame_manually()

                if add_corners:
                    usable_frame_ct += 1
                    self.add_corners_for_all(usable_frame_ct, report_interval, i_frame)
                    # log last usable **filtered** frame
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
            # just in case we're running capture again
            video.clear_results()
            video.scroll_to_beginning()
            # init capture
            video.read_next_frame()
            continue_capture &= video.more_frames_remain

        report_interval = 10
        i_frame = 0
        usable_frame_ct = 0

        while continue_capture:
            if not self.args.frame_number_filter or i_frame in self.camera.usable_frames:
                add_corners = self.__automatic_filter_complex()

                if self.args.manual_filter:
                    add_corners, key = self.filter_frame_manually()

                if add_corners:
                    usable_frame_ct += 1
                    self.add_corners_for_all(usable_frame_ct, report_interval, i_frame)

                    # log last usable **filtered** frame
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
        self.object_points = []
        print("Gathering frame data...")

        if self.args.load_frame_data:
            self.board_object_corner_set = \
                cio.load_frame_data(self.aux_data_file, self.videos, self.board_dims[0], self.board_dims[1],
                                    self.args.board_square_size)

            usable_frame_ct = len(self.videos[0].image_points)

            for i_frame in range(usable_frame_ct):  # @UnusedVariable
                self.object_points.append(self.board_object_corner_set)

        else:
            if self.args.load_images:
                usable_frame_ct = self.load_frame_images()
            elif self.args.frame_count_target != -1:
                usable_frame_ct = self.run_capture_deterministic_count()
            else:
                usable_frame_ct = self.run_capture()
            if self.args.save_frame_data:
                cio.save_frame_data(self.aux_data_file, os.path.join(self.args.folder, self.args.aux_data_file),
                                    self.videos, self.board_object_corner_set)

        print("Total usable frames: {0:d} ({1:.3%})"
              .format(usable_frame_ct, float(usable_frame_ct) / self.total_frames))
        self.usable_frame_count = usable_frame_ct

    def run_calibration(self):
        min_frames = ApplicationSynced.min_frames_to_calibrate
        if self.usable_frame_count < min_frames:
            print("Not enough usable frames to calibrate." +
                  " Need at least {0:d}, got {1:d}".format(min_frames, self.usable_frame_count))
            return

        if self.args.test:
            print("Testing existing calibration (no output will be saved)...")
        else:
            print("Calibrating for max. {0:d} iterations...".format(self.args.max_iterations))
        calibrate(self.rig, [video.image_points for video in self.videos],
                  self.board_object_corner_set,
                  self.args.use_fisheye_model,
                  self.args.use_rational_model,
                  self.args.use_tangential_coeffs,
                  self.args.use_thin_prism,
                  self.args.fix_radial,
                  self.args.fix_thin_prism,
                  self.args.precalibrate_solo,
                  self.args.stereo_only,
                  self.args.max_iterations,
                  self.args.input_calibration is not None,
                  self.args.test)
        if len(self.videos) > 1:
            if self.args.preview:
                l_im = cv2.imread(osp.join(self.args.folder, self.args.preview_files[0]))
                r_im = cv2.imread(osp.join(self.args.folder, self.args.preview_files[1]))
                l_im, r_im = undistort_stereo(self.rig, l_im, r_im)
                path_l = osp.join(self.args.folder, self.args.preview_files[0][:-4] + "_rect.png")
                path_r = osp.join(self.args.folder, self.args.preview_files[1][:-4] + "_rect.png")
                cv2.imwrite(path_l, l_im)
                cv2.imwrite(path_r, r_im)
        if not self.args.skip_printing_output:
            print(self.rig)
        if not self.args.skip_saving_output and not self.args.test:
            cio.save_opencv_calibration(osp.join(self.args.folder, self.args.output),
                                        self.rig)
