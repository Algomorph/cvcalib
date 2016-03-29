"""
/home/algomorph/Factory/calib_video_opencv/intrinsics/video.py.
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
"""
import os
import cv2
from calib.data import CameraIntrinsics, CameraExtrinsics
import numpy as np
from enum import Enum
from lxml import etree


# TODO: figure out how to deal with the filters
class Filter(Enum):
    flip_180 = 0


def string_list_to_filter_list(string_list):
    filter_list = []
    for item in string_list:
        if not item in Filter._member_map_:
            raise ValueError("'{:s}' does not refer to any existing filter. " +
                             "Please, use one of the following: {:s}"
                             .format(item, str(Filter._member_names_)))
        else:
            filter_list.append(Filter[item])
    return filter_list


class Pose(object):
    def __init__(self, T, T_inv=None, rvec=None, tvec=None):
        self.T = T
        if (type(tvec) == type(None)):
            tvec = T[0:3, 3].reshape(3, 1)
        if (type(rvec) == type(None)):
            R = T[0:3, 0:3]
            rvec = cv2.Rodrigues(R)[0]
        if (type(T_inv) == type(None)):
            R = cv2.Rodrigues(rvec)[0]
            R_inv = R.T
            tvec_inv = -R_inv.dot(tvec)
            T_inv = np.vstack((np.append(R_inv, tvec_inv, 1), [0, 0, 0, 1]))

        self.tvec = tvec
        self.rvec = rvec
        self.T_inv = T_inv

    @staticmethod
    def invert_pose_matrix(T):
        tvec = T[0:3, 3].reshape(3, 1)
        R = T[0:3, 0:3]
        R_inv = R.T
        tvec_inv = -R_inv.dot(tvec)
        return np.vstack((np.append(R_inv, tvec_inv, 1), [0, 0, 0, 1]))


class Camera(object):
    """
    Represents a video object & camera that was used to capture it, a wrapper around OpenCV's video_capture
    """
    # TODO: Video and Camera need to be two separate classes, where a camera may include one or more videos
    _unindexed_instance_counter = 0
    _used_indexes = set()

    def __init__(self, video_path, index=None, intrinsics=None, extrinsics=None, load_video=True, filters=[]):
        """
        Build a camera from the specified file at the specified directory
        """
        if (index is None):
            index = Camera._unindexed_instance_counter
            CameraIntrinsics._unindexed_instance_counter += 1
        if (index in Camera._used_indexes):
            raise RuntimeError("{:s}: index {:d} was already used.".format(self.__class__.__name__, index))
        self.index = index
        self.cap = None
        # self.filters = string_list_to_filter_list(filters)
        if video_path[-3:] != "mp4":
            raise ValueError("Specified file does not have .mp4 extension.")
        self.video_path = video_path
        self.name = os.path.basename(video_path)[:-4]
        if load_video:
            self.reopen()
        else:
            self.cap = None
            self.frame_dims = None
            self.frame = None
            self.previous_frame = None
            self.fps = None
            self.frame_count = 0

        # TODO: refactor to image_points
        self.imgpoints = []
        if intrinsics is None:
            self.intrinsics = CameraIntrinsics(self.frame_dims, index=index)
        else:
            self.intrinsics = intrinsics
        if extrinsics is None:
            self.extrinsics = CameraExtrinsics()
        else:
            self.extrinsics = extrinsics

        self.current_image_points = None

        self.more_frames_remain = True
        self.poses = []
        self.usable_frames = {}
        self.calibration_interval = (0,self.frame_count)

    def copy(self):
        return Camera(self.video_path, index=None, intrinsics=self.intrinsics, extrinsics=self.extrinsics, load_video=False)

    def to_xml(self, root_element, as_sequence=False):
        """
        Build an xml node representation of this object under the provided root xml element
        @type root_element:  lxml.etree.SubElement
        @param root_element: the root element to build under
        """
        if (as_sequence == False):
            elem_name = self.__class__.__name__
        else:
            elem_name = "_"
        camera_elem = etree.SubElement(root_element, elem_name, attrib={"index": str(self.index)})
        name_elem = etree.SubElement(camera_elem, "name")
        name_elem.text = self.name
        video_path_elem = etree.SubElement(camera_elem, "video_path")
        video_path_elem.text = self.video_path
        self.intrinsics.to_xml(camera_elem, False)

    def __str__(self, *args, **kwargs):
        return (("{:s}, index: {:d}\nName (h,w): {:s}\n" +
                 "Video path: {:s}\nIntrinsics:\n{:s}")
                .format(self.__class__.__name__, self.index, str(self.name), str(self.video_path),
                        str(self.intrinsics)))

    @staticmethod
    def from_xml(element):
        """
        @type element: lxml.etree.SubElement
        @param element: the element to construct an CameraIntrinsics object from
        @return a new Camera object constructed from XML node with matrices in OpenCV format
        """
        video_path = element.find("video_path").text
        # name = element.find("name").text
        index = int(element.get("index"))
        intrinsics_elem = element.find(CameraIntrinsics.__name__)  # @UndefinedVariable
        intrinsics = CameraIntrinsics.from_xml(intrinsics_elem)
        return Camera(video_path, index, intrinsics, load_video=False)

    def reopen(self):
        if self.cap is not None:
            self.cap.release()
        if not os.path.isfile(self.video_path):
            raise ValueError("No video file found at {0:s}".format(self.video_path))
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("Could not open specified .mp4 file ({0:s}) for capture!".format(self.video_path))
        self.frame_dims = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if (self.cap.get(cv2.CAP_PROP_MONOCHROME) == 0.0):
            self.n_channels = 3
        else:
            self.n_channels = 1
        self.frame = np.zeros((self.frame_dims[0], self.frame_dims[1], self.n_channels), np.uint8)
        self.previous_frame = np.zeros((self.frame_dims[0], self.frame_dims[1], self.n_channels), np.uint8)

    def clear_results(self):
        self.poses = []
        self.imgpoints = []
        self.usable_frames = {}

    def read_next_frame(self):
        self.more_frames_remain, self.frame = self.cap.read()

    def read_at_pos(self, ix_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, ix_frame)
        self.more_frames_remain, self.frame = self.cap.read()

    def read_previous_frame(self):
        """
        For traversing the video backwards.
        """
        cur_frame_ix = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if (cur_frame_ix == 0):
            self.more_frames_remain = False
            self.frame = None
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_ix - 1)  # @UndefinedVariable
        self.more_frames_remain = True
        self.frame = self.cap.read()[1]

    def set_previous_to_current(self):
        self.previous_frame = self.frame

    def scroll_to_frame(self, i_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)  # @UndefinedVariable

    def scroll_to_beginning(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)  # @UndefinedVariable

    def scroll_to_end(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count - 1)  # @UndefinedVariable

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def approximate_corners(self, board_dims):
        found, corners = cv2.findChessboardCorners(self.frame, board_dims)
        self.current_image_points = corners
        return found

    def find_current_pose(self, object_points):
        """
        Find camera pose relative to object using current image point set, 
        object_points are treated as world coordinates
        """
        retval, rvec, tvec = cv2.solvePnPRansac(object_points, self.current_image_points,
                                                self.intrinsics.intrinsic_mat, self.intrinsics.distortion_coeffs,
                                                flags=cv2.SOLVEPNP_ITERATIVE)[0:3]
        if (retval):
            R = cv2.Rodrigues(rvec)[0]
            T = np.vstack((np.append(R, tvec, 1), [0, 0, 0, 1]))
            R_inv = R.T
            tvec_inv = -R_inv.dot(tvec)
            T_inv = np.vstack((np.append(R_inv, tvec_inv, 1), [0, 0, 0, 1]))
            self.poses.append(Pose(T, T_inv, rvec, tvec))
        else:
            self.poses.append(None)
        return retval

    def add_corners(self, i_frame, criteria_subpix, frame_folder_path, save_image):
        grey_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(grey_frame, self.current_image_points, (11, 11), (-1, -1), criteria_subpix)
        if (save_image):
            fname = (os.path.join(frame_folder_path,
                                  "{0:s}{1:04d}{2:s}".format(self.name, i_frame, ".png")))
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
