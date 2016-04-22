import cv2
import os.path
import numpy as np
import math
from calib.geom import Pose


class Video(object):
    """
    A wrapper around the OpenCV video capture,
    intended only for reading video files and obtaining data relevant for calibration
    """

    def __init__(self, path, load=True):
        self.cap = None
        if path[-3:] != "mp4":
            raise ValueError("Specified file does not have .mp4 extension.")
        self.path = path
        self.name = os.path.basename(path)[:-4]

        if load:
            self.reopen()
            self.__get_video_properties()
            self.more_frames_remain = False
        else:
            self.cap = None
            self.frame_dims = None
            self.frame = None
            self.previous_frame = None
            self.fps = None
            self.frame_count = 0
            self.n_channels = 0
            self.more_frames_remain = False

        # current frame data
        self.current_image_points = None

        # frame data
        self.image_points = []
        self.poses = []
        self.usable_frames = {}

        # interval where the checkerboard is detectable
        self.calibration_interval = (0, self.frame_count)

    def __get_video_properties(self):
        self.frame_dims = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.cap.get(cv2.CAP_PROP_MONOCHROME) == 0.0:
            self.n_channels = 3
        else:
            self.n_channels = 1
        self.frame = np.zeros((self.frame_dims[0], self.frame_dims[1], self.n_channels), np.uint8)
        self.previous_frame = np.zeros((self.frame_dims[0], self.frame_dims[1], self.n_channels), np.uint8)

    def reopen(self):
        if self.cap is not None:
            self.cap.release()
        if not os.path.isfile(self.path):
            raise ValueError("No video file found at {0:s}".format(self.path))
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError("Could not open specified .mp4 file ({0:s}) for capture!".format(self.path))

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
        if cur_frame_ix == 0:
            self.more_frames_remain = False
            self.frame = None
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_ix - 1)  # @UndefinedVariable
        self.more_frames_remain = True
        self.frame = self.cap.read()[1]

    def set_previous_to_current(self):
        self.previous_frame = self.frame

    def scroll_to_frame(self, i_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)

    def scroll_to_beginning(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

    def scroll_to_end(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count - 1)

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def clear_results(self):
        self.poses = []
        self.image_points = []
        self.usable_frames = {}

    def try_approximate_corners_blur(self, board_dims, sharpness_threshold):
        sharpness = cv2.Laplacian(self.frame, cv2.CV_64F).var()
        if sharpness < sharpness_threshold:
            return False
        found, corners = cv2.findChessboardCorners(self.frame, board_dims)
        self.current_image_points = corners
        return found

    def try_approximate_corners(self, board_dims):
        found, corners = cv2.findChessboardCorners(self.frame, board_dims)
        self.current_image_points = corners
        self.current_board_dims = board_dims
        return found

    def find_current_pose(self, object_points, intrinsics):
        """
        Find camera pose relative to object using current image point set,
        object_points are treated as world coordinates
        """
        success, rotation_vector, translation_vector = cv2.solvePnPRansac(object_points, self.current_image_points,
                                                                          intrinsics.intrinsic_mat,
                                                                          intrinsics.distortion_coeffs,
                                                                          flags=cv2.SOLVEPNP_ITERATIVE)[0:3]
        if success:
            self.poses.append(Pose(rotation=rotation_vector, translation_vector=translation_vector))
        else:
            self.poses.append(None)
        return success

    def find_reprojection_error(self, i_usable_frame, object_points, intrinsics):
        rotation_vector = self.poses[i_usable_frame].rvec
        translation_vector = self.poses[i_usable_frame].tvec
        img_pts = self.image_points[i_usable_frame]

        est_pts = cv2.projectPoints(object_points, rotation_vector, translation_vector,
                                    intrinsics.intrinsic_mat, intrinsics.distortion_coeffs)[0]

        rms = math.sqrt(((img_pts - est_pts) ** 2).sum() / len(object_points))
        return rms

    # TODO: passing in both frame_folder_path and save_image doesn't make sense. Make saving dependent on the former.
    def add_corners(self, i_frame, subpixel_criteria, frame_folder_path=None,
                    save_image=False, save_chekerboard_overlay=False):
        grey_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(grey_frame, self.current_image_points, (11, 11), (-1, -1), subpixel_criteria)
        if save_image:
            png_path = (os.path.join(frame_folder_path,
                                     "{0:s}{1:04d}{2:s}".format(self.name, i_frame, ".png")))
            cv2.imwrite(png_path, self.frame)
            if save_chekerboard_overlay:
                png_path = (os.path.join(frame_folder_path,
                                         "checkerboard_{0:s}{1:04d}{2:s}".format(self.name, i_frame, ".png")))
                overlay = self.frame.copy()
                cv2.drawChessboardCorners(overlay, self.current_board_dims, self.current_image_points, True)
                cv2.imwrite(png_path, overlay)
        self.usable_frames[i_frame] = len(self.image_points)
        self.image_points.append(self.current_image_points)

    def filter_frame_manually(self):
        display_image = self.frame
        cv2.imshow("frame of video {0:s}".format(self.name), display_image)
        key = cv2.waitKey(0) & 0xFF
        add_corners = (key == ord('a'))
        cv2.destroyWindow("frame")
        return add_corners, key
