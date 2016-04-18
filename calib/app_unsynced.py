"""
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
"""
import os.path
import time

import cv2
import numpy as np
import math
import calib.io as cio
from calib.app import Application
from calib.camera import Camera, Pose
from calib.data import CameraIntrinsics
from calib.rig import StereoRig
from calib.utils import stereo_calibrate, calibrate_wrapper
import logging, sys


class ApplicationUnsynced(Application):
    """
    ApplicationUnsynced
    """

    def __init__(self, args):
        """
        Constructor
        @type args: object
        """
        Application.__init__(self, args)

        self.frame_numbers = {}

        intrinsic_arr = []

        if args.input_calibration is None or len(args.input_calibration) == 0:
            raise ValueError("Unsynced calibration requires input calibration parameters for all " +
                             "cameras used to take the videos.")

        # load calibration files
        initial_calibration = []
        for calib_file in args.input_calibration:
            initial_calibration.append(cio.load_opencv_calibration(os.path.join(args.folder, calib_file)))

        # sanity checks & calibration arrangement
        if type(initial_calibration[0]) == StereoRig:
            for calibration_info in initial_calibration:
                if type(calibration_info) != StereoRig:
                    # TODO combine with the other thing... remove type restriction, just keep track of the total number
                    raise TypeError("For stereo, all calibration files should contain stereo information. Expecting: " +
                                    str(StereoRig) + ". Got: " + str(type(calibration_info)))
                intrinsic_arr += calibration_info.intrinsics  # aggregate intrinsics into a single array
            if len(args.videos) % 2 != 0:
                raise ValueError("Provided stereo input calibration files: expecting an even " +
                                 "number of videos. Got: {:d}".format(len(args.videos)))
            if len(initial_calibration) != len(args.videos) // 2:
                raise ValueError("Number of stereo calibration files is not half the number of videos.")
        else:
            if len(initial_calibration) != len(args.videos):
                raise ValueError("Number of intrinsics files is does not equal the number of videos.")
            for calibration_info in initial_calibration:
                if type(calibration_info) == Camera:
                    intrinsic_arr.append(calibration_info.intrinsics)
                elif type(calibration_info) == CameraIntrinsics:
                    intrinsic_arr.append(calibration_info)
                else:
                    raise RuntimeError("Unsupported calibration file format.")

        ix_video = 0
        self.cameras = []
        # load videos
        for video_filename in args.videos:
            self.cameras.append(Camera(os.path.join(args.folder, video_filename),
                                       index=ix_video,
                                       intrinsics=intrinsic_arr[ix_video]))
            ix_video += 1
        if len(self.cameras) < 2:
            raise ValueError(
                "Expecting at least two videos & input calibration parameters for the corresponding videos")

    # use this instead of just camera.approximate_corners for very "sparse" videos, with lots of bad frames
    def __browse_around_frame(self, camera, around_frame, frame_range=64, interval=16, verbose=True):
        for i_frame in range(max(around_frame - frame_range, 0),
                             min(around_frame + frame_range, camera.frame_count), interval):
            camera.read_at_pos(i_frame)
            if verbose:
                print('-', end="", flush=True)
            if camera.approximate_corners(self.board_dims):
                return i_frame
        return -1

    def __seek_calib_limit(self, camera, frame_range, max_miss_count=3, verbose=True):
        frame_range_signed_length = frame_range[1] - frame_range[0]
        sample_interval_frames = frame_range_signed_length // 2
        failed_attempts = 0
        while sample_interval_frames != 0:
            miss_count = 0
            try:
                if verbose:
                    if sample_interval_frames < 0:
                        print("\nSampling every {:d} frames within {:s}, backwards."
                              .format(-sample_interval_frames, str((frame_range[1], frame_range[0]))))
                    else:
                        print("\nSampling every {:d} frames within {:s}.".format(sample_interval_frames,
                                                                                 str(frame_range)))
                for i_frame in range(frame_range[0], frame_range[1], sample_interval_frames):
                    camera.read_at_pos(i_frame)
                    if verbose:
                        print('.', end="", flush=True)
                    if camera.approximate_corners(self.board_dims):
                        frame_range[0] = i_frame
                        miss_count = 0
                    else:
                        miss_count += 1
                        if miss_count > max_miss_count:
                            # too many frames w/o calibration board, highly unlikely those are all bad frames,
                            # go to finer scan
                            frame_range[1] = i_frame
                            break
                sample_interval_frames = round(sample_interval_frames / 2)
            except cv2.error as e:
                failed_attempts += 1
                if failed_attempts > 2:
                    raise RuntimeError("Too many failed attempts. Frame index: " + str(i_frame))
                print("FFmpeg hickup, attempting to reopen video.")
                camera.reopen()  # workaround for ffmpeg AVC/H.264 bug
        return frame_range[0]

    def find_calibration_intervals(self, verbose=True):
        start = time.time()
        for camera in self.cameras:

            if self.args.time_range_hint is None:
                rough_seek_range = (0, camera.frame_count - 0)
            else:
                rough_seek_range = (round(max(0, camera.fps * self.args.time_range_hint[0])),
                                    round(min(camera.fps * self.args.time_range_hint[1], camera.frame_count - 0)))

            if verbose:
                print("Performing initial rough scan of {0:s} for calibration board...".format(camera.name))

            found = False
            # find approximate start and end of calibration
            sample_interval_frames = camera.frame_count // 2
            failed_attempts = 0

            while not found and sample_interval_frames > 4:
                try:
                    if verbose:
                        print("\nSampling every {:d} frames.".format(sample_interval_frames))
                    for i_frame in range(rough_seek_range[0],
                                         rough_seek_range[1],
                                         sample_interval_frames):
                        camera.read_at_pos(i_frame)
                        if verbose:
                            print('.', end="", flush=True)
                        if camera.approximate_corners(self.board_dims):
                            calibration_end = calibration_start = i_frame
                            print("Hit at {:d}!".format(i_frame))
                            found = True
                            break

                    sample_interval_frames //= 2
                except cv2.error as e:
                    failed_attempts += 1
                    if failed_attempts > 2:
                        raise RuntimeError("Too many failed attempts. Frame index: " + str(i_frame))
                    print("FFmpeg hickup, attempting to reopen video.")
                    camera.reopen()  # workaround for ffmpeg AVC/H.264 bug

            # ** find exact start & end of streak **

            # traverse backward from inexact start
            if verbose:
                print("Seeking first calibration frame of {0:s}...".format(camera.name))
            calibration_start = self.__seek_calib_limit(camera, [calibration_start, rough_seek_range[0]],
                                                        max_miss_count=self.args.seek_miss_count, verbose=verbose)

            # traverse forward from inexact end
            if verbose:
                print("Seeking last calibration frame of {0:s}...".format(camera.name))
            calibration_end = self.__seek_calib_limit(camera, [calibration_end, rough_seek_range[1]],
                                                      max_miss_count=self.args.seek_miss_count, verbose=verbose)

            camera.calibration_interval = (calibration_start, calibration_end)
            if verbose:
                print("Found calibration frame range for camera {:s} to be within {:s}"
                      .format(camera.name, str(camera.calibration_interval)))

        end = time.time()
        if verbose:
            print("Total calibration interval seek time: {:.3f} seconds.".format(end - start))
        if self.args.save_calibration_intervals:
            cio.save_calibration_intervals(self.aux_data_file, self.aux_data_path, self.cameras, verbose=verbose)

    def __terminate_still_streak(self, streak, longest_streak, still_streaks, verbose=True):
        min_still_streak = self.args.max_frame_offset * 2 + 1
        # min_still_streak = 30

        if len(streak) >= min_still_streak:
            still_streaks.append(streak)
        else:
            return [0, 0], longest_streak
        if len(streak) > len(longest_streak):
            longest_streak = streak
            if verbose:
                print("Longest consecutive streak with calibration board " +
                      "staying still relative to camera: {:d}".format(streak[1] - streak[0]))
        return [0, 0], longest_streak

    def run_capture(self, verbose=True):
        """
        Run capture for each camera separately
        """
        report_interval = 5.0  # seconds

        for camera in self.cameras:
            i_frame = camera.calibration_interval[0]
            if verbose:
                print("Capturing calibration board points for camera {0:s}".format(camera.name))

            camera.reopen()
            # just in case we're running capture again
            camera.clear_results()
            camera.scroll_to_frame(i_frame)
            total_calibration_frames = camera.calibration_interval[1] - camera.calibration_interval[0]
            # init capture
            camera.read_next_frame()
            key = 0
            check_time = time.time()
            longest_still_streak = []
            still_streak = [0, 0]
            still_streaks = []
            frame_counter = 0
            # TODO: fix still streak acquisition
            while i_frame < camera.calibration_interval[1] and not (self.args.manual_filter and key == 27):
                add_corners = False
                if not self.args.frame_number_filter or i_frame in camera.usable_frames:
                    # TODO: add blur filter support to camera class
                    add_corners = camera.approximate_corners(self.board_dims)

                    if self.args.manual_filter and add_corners:
                        add_corners, key = camera.filter_frame_manually()

                    if add_corners:
                        camera.add_corners(i_frame, self.criteria_subpix,
                                           self.full_frame_folder_path, self.args.save_images)
                        #camera.find_current_pose(self.board_object_corner_set)

                        cur_corners = camera.imgpoints[len(camera.imgpoints) - 1]
                        if len(still_streak) > 0:
                            prev_corners = camera.imgpoints[len(camera.imgpoints) - 2]
                            mean_px_dist_to_prev = np.linalg.norm(cur_corners - prev_corners).mean()
                            if mean_px_dist_to_prev < 0.5:
                                still_streak[1] = i_frame
                            else:
                                still_streak, longest_still_streak = \
                                    self.__terminate_still_streak(still_streak, longest_still_streak, still_streaks,
                                                                  verbose)
                        else:
                            still_streak[0] = i_frame

                if not add_corners:
                    still_streak, longest_still_streak = \
                        self.__terminate_still_streak(still_streak, longest_still_streak, still_streaks, verbose)

                camera.read_next_frame()
                frame_counter += 1
                # fixed time interval reporting
                if verbose and time.time() - check_time > report_interval:
                    print("Processed: {:.2%}, frame: {:d}, # usable frames: {:d}, # still streaks: {:d}"
                          .format(frame_counter / total_calibration_frames, i_frame, len(camera.usable_frames),
                                  len(still_streaks)))
                    check_time += report_interval
                i_frame += 1

            # in case the last frame was also added
            still_streak, longest_still_streak = \
                self.__terminate_still_streak(still_streak, longest_still_streak, still_streaks, verbose)
            camera.still_streaks = [tuple(streak) for streak in still_streaks]
            if verbose:
                print(" Done. Found {:d} usable frames. Longest still streak: {:d}".format(len(camera.usable_frames),
                                                                                           len(longest_still_streak)))

        if self.args.manual_filter:
            cv2.destroyAllWindows()

    def find_camera_poses(self, verbose=False):
        for camera in self.cameras:
            camera.poses = []
            object_points = [self.board_object_corner_set for dummy in range(len(camera.imgpoints))]
            rotations, translations = calibrate_wrapper(camera,
                                                        object_points,
                                                        self.args.use_rational_model,
                                                        self.args.use_tangential_coeffs,
                                                        self.args.use_thin_prism,
                                                        fix_radial=True,
                                                        fix_thin_prism=True,
                                                        max_iterations=1,
                                                        use_existing_guess=True,
                                                        test=True)
            if verbose:
                print("Pose reprojection error for camera {:s}: {:.4f}".format(camera.name, camera.intrinsics.error))
            for ix_pose in range(len(rotations)):
                translation = translations[ix_pose]
                rotation = rotations[ix_pose]
                pose = Pose(rotation_vector=rotation, translation_vector=translation)
                camera.poses.append(pose)


    @staticmethod
    def __aux_streak_within(source_streak, target_streak):
        source_middle = source_streak[0] + (source_streak[1] - source_streak[0]) // 2
        return (((target_streak[0] <= source_streak[0] <= target_streak[1]) or
                 (target_streak[0] <= source_streak[1] <= target_streak[1])) and
                (target_streak[0] <= source_middle <= target_streak[1]))

    def match_still_streaks(self, verbose=True):
        """
        look for streaks in target streak set that overlap at or within the bounds of each
        streak within the source streak set
        """
        if verbose:
            print("For each pair of videos, looking for timewise-overlapping"
                  + "streaks of frames where calibration board is not moving.")
        for vid in self.cameras:
            vid.still_streak_overlaps = {}
            # TODO: potentially reduce to processing only one source video
        for i_vid in range(len(self.cameras)):
            source_streaks = self.cameras[i_vid].still_streaks
            for j_vid in range(i_vid + 1, len(self.cameras)):
                target_streaks = self.cameras[j_vid].still_streaks
                overlaps = []
                for source_streak in source_streaks:
                    found = False
                    ix_target_streak = 0
                    while not found and ix_target_streak < len(target_streaks):
                        target_streak = target_streaks[ix_target_streak]
                        if ApplicationUnsynced.__aux_streak_within(source_streak, target_streak):
                            overlaps.append((source_streak, target_streak))
                        ix_target_streak += 1
                self.cameras[i_vid].still_streak_overlaps[j_vid] = overlaps

    def calibrate_using_stills(self, verbose=True):
        source_cam = self.cameras[0]

        # cut at least this number of frames off the range bounds, because
        # some frames in the beginning or end of the ranges will have some minor board motion
        cutoff = 1
        for j_vid in range(1, len(self.cameras)):
            target_cam = self.cameras[j_vid]
            overlaps = source_cam.still_streak_overlaps[j_vid]

            imgpts_src = []
            imgpts_tgt = []
            for overlap in overlaps:
                source_range = overlap[0]
                target_range = overlap[1]

                src_range_len = source_range[1] - source_range[0]
                tgt_range_len = target_range[1] - target_range[0]
                tgt_half_len = tgt_range_len // 2
                src_half_len = src_range_len // 2
                if src_range_len > tgt_range_len:
                    increment = 0 if tgt_range_len % 2 == 0 else 1
                    src_mid = source_range[0] + src_half_len
                    source_range = (src_mid - tgt_half_len, src_mid + tgt_half_len + increment)
                else:  # target range is smaller than or equal to the source range
                    increment = 0 if src_range_len % 2 == 0 else 1
                    tgt_mid = target_range[0] + src_half_len
                    target_range = (tgt_mid - src_half_len, src_mid + src_half_len + increment)

                for ix_frame in range(source_range[0] + cutoff, source_range[1] - cutoff):
                    imgpts_src.append(source_cam.imgpoints[source_cam.usable_frames[ix_frame]])
                for ix_frame in range(target_range[0] + cutoff, target_range[1] - cutoff):
                    imgpts_tgt.append(target_cam.imgpoints[target_cam.usable_frames[ix_frame]])

            rig = StereoRig((source_cam.copy(), target_cam.copy()))
            rig.cameras[0].imgpoints = imgpts_src
            rig.cameras[1].imgpoints = imgpts_tgt

            stereo_calibrate(rig,
                             self.object_points, self.frame_dims,
                             self.args.use_fisheye_model,
                             self.args.use_rational_model,
                             self.args.use_tangential_coeffs,
                             self.args.use_thin_prism,
                             self.args.fix_radial,
                             self.args.fix_thin_prism,
                             precalibrate_solo=False,
                             stereo_only=True,
                             max_iterations=self.args.max_iterations,
                             fix_intrinsics=True)
            target_cam.extrinsics = rig.extrinsics

    def calibrate_time_reprojection(self, sample_count=1000, verbose=2, save_data=False, min_offset_datapoints=10):
        if type(verbose) == bool:
            verbose = int(verbose)
        logging.basicConfig(stream=sys.stderr)#, level=logging.DEBUG)

        max_offset = self.args.max_frame_offset
        source_camera = self.cameras[0]

        source_poses = []
        sample_frame_numbers = []

        source_camera.frame_offset = 0

        if sample_count > 0:
            sampling_interval = len(source_camera.usable_frames) // sample_count
            start = (len(source_camera.usable_frames) % sample_count) // 2

            # source poses array will be parallel to sample_frame_numbers, i.e. source_poses[i] is the pose
            # of the source camera at the frame position sample_frame_numbers[i] in the original source video

            usable_frames = list(source_camera.usable_frames.keys())
            usable_frames.sort()

            for i_usable_frame in range(start, len(usable_frames) - start - 1, sampling_interval):
                usable_frame_num = usable_frames[i_usable_frame]
                sample_frame_numbers.append(usable_frame_num)
                source_pose = source_camera.poses[source_camera.usable_frames[usable_frame_num]]
                source_poses.append(source_pose)
            if verbose:
                print("Sample count: {:d}".format(len(sample_frame_numbers)))
                # DEBUG LINE
                logging.debug("Calib interval: {:s}, first usable frame: {:d}"
                              .format(str(source_camera.calibration_interval), usable_frames[0]))
                logging.debug("Sample frames: {:s}".format(str(sample_frame_numbers)))
        else:
            sample_frame_numbers = list(source_camera.usable_frames.keys())
            sample_frame_numbers.sort()
            source_poses = source_camera.poses
            sample_count = len(sample_frame_numbers)

        offsets = [0]
        for target_camera in self.cameras[1:]:
            if verbose:
                print("========================================================")
                print("Processing time shift between cameras '{:s}' and '{:s}'."
                      .format(source_camera.name, target_camera.name))
                print("========================================================")
            possible_offset_count = max_offset * 2 + 1

            # flag_array to remember unfilled entries
            flag_array = np.zeros((possible_offset_count, sample_count), dtype=np.bool)
            # transforms = np.zeros((possible_offset_count, sample_count, 4, 4), dtype=np.float64)
            pose_differences = np.zeros((possible_offset_count, sample_count), dtype=np.float64)
            projection_rms_mat = np.zeros((possible_offset_count, sample_count), dtype=np.float64)

            offset_sample_counts = np.zeros(possible_offset_count, dtype=np.float64)
            offset_mean_pose_diffs = np.zeros(possible_offset_count, dtype=np.float64)
            offset_pt_rms = np.zeros(possible_offset_count, dtype=np.float64)

            offset_range = range(-max_offset, max_offset + 1)

            best_offset = 0
            best_offset_rms = float(sys.maxsize)
            # traverse all possible offsets
            for offset in offset_range:

                if verbose > 1:
                    print("Processing offset {:d}. ".format(offset), end="")

                ix_offset = offset + max_offset

                # per-offset cumulative things
                offset_comparison_count = 0
                offset_cumulative_pt_counts = 0
                offset_cumulative_pose_counts = 0
                offset_cumulative_pose_error = 0.0
                offset_cumulative_pt_squared_error = 0.0

                # for each offset, traverse all source frame samples
                for j_sample in range(0, len(sample_frame_numbers)):
                    source_frame = sample_frame_numbers[j_sample]
                    j_target_frame = source_frame + offset

                    # check if frame of the target video at this particular offset from the source sample frame has
                    # a usable calibration board
                    if j_target_frame in target_camera.usable_frames:
                        flag_array[ix_offset, j_sample] = True
                        source_pose = source_poses[j_sample]
                        j_target_usable_frame = target_camera.usable_frames[j_target_frame]
                        target_pose = target_camera.poses[j_target_usable_frame]

                        '''
                        Transform between this camera and the other one.
                        Future notation:
                        T(x, y, f, o) denotes estimated transform from camera x at frame f to camera y at frame f + o
                        '''
                        transform = target_pose.T.dot(source_pose.T_inv)

                        rms = target_camera.find_reprojection_error(j_target_usable_frame, self.board_object_corner_set)

                        if rms > 1.0:
                            continue

                        cumulative_pose_error = 0.0
                        cumulative_squared_point_error = 0.0
                        pose_count = 0
                        point_count = 0
                        # for each sample, traverse all other samples
                        for i_sample in range(0, len(sample_frame_numbers)):
                            source_frame = sample_frame_numbers[i_sample]
                            i_target_frame = source_frame + offset
                            if i_sample != j_sample and i_target_frame in target_camera.usable_frames:
                                offset_comparison_count += 1
                                '''
                                use the same estimated transform between source & target cameras for specific offset
                                on other frame samples
                                's' means source camera, 't' means target camera
                                [R|t]_(x,z) means camera x extrinsics at frame z
                                [R|t]'_(x,z) means estimated camera x extrinsics at frame z
                                T(x, y, f, o) denotes estimated transform from camera x at frame f
                                  to camera y at frame f + o
                                X_im denotes image coordinates
                                X_im' denotes estimated image coordinates
                                X_w denotes corresponding world (object) coordinates

                                If the estimate at this offset is a good estimate of the transform between cameras
                                for all frames?

                                Firstly, we can apply transform to the source camera pose and see how far we end up
                                from the target camera transform
                                i != j,
                                [R|t]_(t,i)' = T(s,t,j,k).dot([R|t]_(s,i+k))
                                [R|t]_(t,i) =?= [R|t]_(t,i)'
                                '''
                                target_pose = target_camera.poses[target_camera.usable_frames[i_target_frame]]
                                est_target_pose = Pose(transform.dot(source_poses[i_sample].T))
                                cumulative_pose_error += est_target_pose.diff(target_pose)
                                '''
                                Secondly, we can use the transform applied to source camera pose and the target
                                intrinsics to project the world (object) coordinates to the image plane,
                                and then compare them with the empirical observations

                                X_im(t,i) = K_t.dot(dist_t([R|t]_(t,i).dot(X_w)))
                                X_im'(t,i) = K_t.dot(dist_t([R|t]_(t,i)'.dot(X_w)))
                                X_im(t,i) =?= X_im'(t,i)

                                Note: X_im(t,i) computation above is for reference only, no need to reproject as
                                 we already have empirical observations of the image points
                                '''
                                target_points = target_camera.imgpoints[target_camera.usable_frames[i_target_frame]]
                                est_target_points = \
                                    cv2.projectPoints(objectPoints=self.board_object_corner_set,
                                                      rvec=est_target_pose.rvec,
                                                      tvec=est_target_pose.tvec,
                                                      cameraMatrix=target_camera.intrinsics.intrinsic_mat,
                                                      distCoeffs=target_camera.intrinsics.distortion_coeffs)[0]

                                # np.linalg.norm(target_points - est_target_points, axis=2).flatten()
                                cumulative_squared_point_error += ((target_points - est_target_points) ** 2).sum()
                                pose_count += 1
                                point_count += len(self.board_object_corner_set)

                        if pose_count > 0:
                            mean_pose_error = cumulative_pose_error / pose_count
                            root_mean_square_pt_error = math.sqrt(cumulative_squared_point_error /
                                                                  point_count)
                            pose_differences[ix_offset, j_sample] = mean_pose_error
                            projection_rms_mat[ix_offset, j_sample] = root_mean_square_pt_error

                            offset_cumulative_pose_error += cumulative_pose_error
                            offset_cumulative_pose_counts += pose_count
                            offset_cumulative_pt_counts += point_count
                            offset_cumulative_pt_squared_error += cumulative_squared_point_error
                if verbose > 1:
                    print("Total comparison count: {:d} ".format(offset_comparison_count), end="")
                offset_sample_counts[ix_offset] = offset_comparison_count
                if offset_cumulative_pose_counts > min_offset_datapoints:
                    offset_pose_error = offset_cumulative_pose_error / offset_cumulative_pose_counts
                    offset_mean_pose_diffs[ix_offset] = offset_pose_error
                    rms = math.sqrt(offset_cumulative_pt_squared_error / offset_cumulative_pt_counts)
                    offset_pt_rms[ix_offset] = rms
                    if verbose > 1:
                        print("RMS error: {:.3f}; pose error: {:.3f}".format(rms, offset_pose_error), end="")
                    if rms < best_offset_rms:
                        best_offset = offset
                        best_offset_rms = rms
                if verbose > 1:
                    print("\n", end="")

            if save_data:
                np.savez_compressed(os.path.join(self.args.folder, target_camera.name + "_tc_data.npz"),
                                    sample_counts=offset_sample_counts,
                                    mean_pose_diffs=offset_mean_pose_diffs,
                                    point_rms=offset_pt_rms,
                                    flag_array=flag_array,
                                    pose_diff_mat=pose_differences,
                                    point_rms_mat=projection_rms_mat)

            target_camera.offset = best_offset
            target_camera.offset_error = best_offset_rms
            if verbose:
                print("Offset for {:s}-->{:s}: {d}, RMS error: {:.5f}".format(source_camera.name, target_camera.name,
                                                                              best_offset, best_offset_rms))
            offsets.append(best_offset)
        if save_data:
            np.savetxt('autooffset.txt', offsets)

    def gather_frame_data(self, verbose=True):
        if self.args.load_calibration_intervals:
            cio.load_calibration_intervals(self.aux_data_file, self.cameras, verbose=verbose)
        else:
            if self.args.use_all_frames:
                min_fc = sys.maxsize
                for camera in self.cameras:
                    if camera.frame_count < min_fc:
                        min_fc = camera.frame_count
                for camera in self.cameras:
                    camera.calibration_interval = (0, min_fc)
            else:
                self.find_calibration_intervals(verbose)

        if self.args.load_frame_data:
            self.board_object_corner_set = cio.load_frame_data(self.aux_data_file, self.cameras,
                                                               verbose=verbose)
            if verbose:
                for camera in self.cameras:
                    max_rms = 0.0
                    min_rms = float(sys.maxsize)
                    object_points = self.board_object_corner_set
                    cum_rms = 0.0
                    for i_usable_frame in range(0, len(camera.poses)):
                        rms = camera.find_reprojection_error(i_usable_frame, object_points)
                        # if ... elif ... min/max tracking only works for cases with > 1 usable frame
                        if rms > max_rms:
                            max_rms = rms
                        elif rms < min_rms:
                            min_rms = rms
                        cum_rms += rms
                    print("AVE, MIN, MAX RMS for camera {:s} : {:.4f}, {:.4f}, {:.4f}"
                          .format(camera.name, cum_rms / len(camera.poses), min_rms, max_rms))

        else:
            self.run_capture(verbose)
            self.find_camera_poses(verbose)
            if self.args.save_frame_data:
                cio.save_frame_data(self.aux_data_file, os.path.join(self.args.folder, self.args.aux_data_file),
                                    self.cameras,
                                    self.board_object_corner_set)
