"""
Created on Jan 1, 2016

@author: Gregory Kramida
"""
from lxml import etree
import numpy as np
from calib import camera as camera_module, geom, rig
from calib.geom import Pose
from calib.camera import Camera

IMAGE_POINTS = "image_points"
FRAME_NUMBERS = "frame_numbers"
OBJECT_POINT_SET = "object_point_set"
POSES = "poses"
CALIBRATION_INTERVALS = "calibration_intervals"


def load_frame_data(archive, videos, board_height=None,
                    board_width=None, board_square_size=None,
                    verbose=True):

    if verbose:
        print("Loading object & image positions from archive.")

    if OBJECT_POINT_SET in archive:
        object_point_set = archive[OBJECT_POINT_SET]
    else:
        object_point_set = geom.generate_board_object_points(board_height, board_width, board_square_size)

    video_by_name = {}
    for video in videos:
        video_by_name[video.name] = video

    # legacy frame numbers
    if FRAME_NUMBERS in archive:
        frame_numbers = archive[FRAME_NUMBERS]
        for video in videos:
            video.usable_frames = {}
            i_key = 0
            for key in frame_numbers:
                video.usable_frames[key] = i_key
                i_key += 1
        if verbose:
            print("Loaded {:d} usable frame numbers for all cameras in legacy format.".format(len(frame_numbers)))

    for array_name, value in archive.items():
        if array_name.startswith(IMAGE_POINTS):
            vid_name = array_name[len(IMAGE_POINTS):]
            video_by_name[vid_name].image_points = value
            if verbose:
                print("Loaded {:d} image point sets for camera {:s}".format(len(value), vid_name), flush=True)
        elif array_name.startswith(FRAME_NUMBERS) and not array_name == FRAME_NUMBERS:
            vid_name = array_name[len(FRAME_NUMBERS):]
            video = video_by_name[vid_name]
            video.usable_frames = {}
            i_key = 0
            for key in value:
                video.usable_frames[key] = i_key
                i_key += 1
            if verbose:
                print("Loaded {:d} usable frame numbers for camera {:s}".format(len(value), vid_name), flush=True)
        elif array_name.startswith(POSES):
            vid_name = array_name[len(POSES):]
            # process poses
            video_by_name[vid_name].poses = [Pose(T) for T in value]
            if verbose:
                print("Loaded {:d} poses for camera {:s}".format(len(value), vid_name), flush=True)

    return object_point_set


def save_frame_data(archive, path, videos, object_point_set, verbose=True):
    if verbose:
        print("Saving corners to {0:s}".format(path))
    for video in videos:
        archive[IMAGE_POINTS + str(video.name)] = video.image_points
        archive[FRAME_NUMBERS + str(video.name)] = list(video.usable_frames.keys())
        if len(video.poses) > 0:
            archive[POSES + str(video.name)] = np.array([pose.T for pose in video.poses])

    archive[OBJECT_POINT_SET] = object_point_set
    np.savez_compressed(path, **archive)


def load_calibration_intervals(archive, videos, verbose=True):
    if verbose:
        print("Loading calibration frame intervals from archive.")
    if CALIBRATION_INTERVALS in archive:
        ranges = archive[CALIBRATION_INTERVALS]
        if len(videos) != ranges.shape[0]:
            raise ValueError("Need to have the same number of rows in the frame_ranges array as the number of cameras.")
        ix_cam = 0
        for video in videos:
            video.calibration_interval = tuple(ranges[ix_cam])
            ix_cam += 1
    else:
        raise ValueError("No calibration intervals found in the provided archive.")


def save_calibration_intervals(archive, path, videos, verbose=True):
    if verbose:
        print("Saving calibration intervals to {0:s}".format(path))
    ranges = []
    for video in videos:
        if video.calibration_interval is None:
            raise ValueError("Expecting all cameras to have valid calibration frame ranges. Got: None")
        ranges.append(video.calibration_interval)
    ranges = np.array(ranges)
    archive[CALIBRATION_INTERVALS] = ranges
    np.savez_compressed(path, **archive)


def load_opencv_stereo_calibration(path):
    """
    Load stereo calibration information from xml file
    @type path: str
    @param path: video_path to xml file
    @return stereo calibration: loaded from the given xml file
    @rtype calib.data.StereoRig
    """
    tree = etree.parse(path)
    stereo_calib_elem = tree.find("Rig")
    return rig.Rig.from_xml(stereo_calib_elem)


def load_opencv_single_calibration(path):
    """
    Load single-camera calibration information from xml file
    @type path: str
    @param path: video_path to xml file
    @return calibration info: loaded from the given xml file
    @rtype calib.data.CameraIntrinsics
    """
    tree = etree.parse(path)
    calib_elem = tree.find(Camera.Intrinsics.__name__)
    return Camera.Intrinsics.from_xml(calib_elem)


def load_opencv_calibration(path):
    """
    Load any kind (stereo or single) of calibration result from the file
    @type path: str
    @param path: path to xml file
    @return calibration info: loaded from the given xml file
    @rtype calib.data.CameraIntrinsics | calib.data.StereoRig
    """
    tree = etree.parse(path)
    first_elem = tree.getroot().getchildren()[0]
    class_name = first_elem.tag
    modules = [camera_module, rig]
    object_class = None
    for module in modules:
        if hasattr(module, class_name):
            object_class = getattr(module, class_name)
    if object_class is None:
        # legacy formats
        if class_name == "_StereoRig":
            object_class = rig.Rig
        elif class_name == "CameraIntrinsics":
            object_class = Camera.Intrinsics
        else:
            raise ValueError("Unexpected calibration format in file {:s}, got XML tag {:s}. "
                             "For legacy StereoRig files, be sure to rename the tag to _StereoRig."
                             .format(path, class_name))
    calib_info = object_class.from_xml(first_elem)
    return calib_info


def save_opencv_calibration(path, calibration_info):
    root = etree.Element("opencv_storage")
    calibration_info.to_xml(root)
    et = etree.ElementTree(root)
    with open(path, 'wb') as f:
        et.write(f, encoding="utf-8", xml_declaration=True, pretty_print=True)
    # little hack necessary to replace the single quotes (that OpenCV doesn't like) with double quotes
    s = open(path).read()
    s = s.replace("'", "\"")
    with open(path, 'w') as f:
        f.write(s)
        f.flush()
