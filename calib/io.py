"""
Created on Jan 1, 2016

@author: Gregory Kramida
"""
from lxml import etree
import numpy as np
from calib import data, camera, geom, rig
from calib.camera import Pose

IMAGE_POINTS = "image_points"
FRAME_NUMBERS = "frame_numbers"
OBJECT_POINT_SET = "object_point_set"
POSES = "poses"
CALIBRATION_INTERVALS = "calibration_intervals"

'''
TODO: need a separate set of load/save functions for frame_numbers,
remove these from here OR rename to load_frame_data
'''


def load_corners(archive, cameras, board_height=None,
                 board_width=None, board_square_size=None,
                 verbose=True):

    if verbose:
        print("Loading object & image positions from archive.")

    if OBJECT_POINT_SET in archive:
        object_point_set = archive[OBJECT_POINT_SET]
    else:
        object_point_set = geom.generate_object_points(board_height, board_width, board_square_size)

    camera_by_name = {}
    for camera in cameras:
        camera_by_name[camera.name] = camera

    # legacy frame numbers
    if FRAME_NUMBERS in archive:
        frame_numbers = archive[FRAME_NUMBERS]
        for camera in cameras:
            camera.usable_frames = {}
            i_key = 0
            for key in frame_numbers:
                camera.usable_frames[key] = i_key
                i_key += 1
        if verbose:
            print("Loaded {:d} usable frame numbers for all cameras in legacy format.".format(len(frame_numbers)))

    for array_name, value in archive.items():
        if array_name.startswith(IMAGE_POINTS):
            vid_name = array_name[len(IMAGE_POINTS):]
            camera_by_name[vid_name].imgpoints = value
            if verbose:
                print("Loaded {:d} image point sets for camera {:s}".format(len(value), vid_name), flush=True)
        elif array_name.startswith(FRAME_NUMBERS) and not array_name == FRAME_NUMBERS:
            vid_name = array_name[len(FRAME_NUMBERS):]
            camera = camera_by_name[vid_name]
            camera.usable_frames = {}
            i_key = 0
            for key in value:
                camera.usable_frames[key] = i_key
                i_key += 1
            if verbose:
                print("Loaded {:d} usable frame numbers for camera {:s}".format(len(value), vid_name), flush=True)
        elif array_name.startswith(POSES):
            vid_name = array_name[len(POSES):]
            # process poses
            camera_by_name[vid_name].poses = [Pose(T) for T in value]
            if verbose:
                print("Loaded {:d} poses for camera {:s}".format(len(value),vid_name), flush=True)

    return object_point_set


def save_corners(archive, path, cameras, object_point_set, verbose=True):
    if verbose:
        print("Saving corners to {0:s}".format(path))
    for camera in cameras:
        archive[IMAGE_POINTS + str(camera.name)] = camera.imgpoints
        archive[FRAME_NUMBERS + str(camera.name)] = list(camera.usable_frames.keys())
        if len(camera.poses) > 0:
            archive[POSES + str(camera.name)] = np.array([pose.T for pose in camera.poses])

    archive[OBJECT_POINT_SET] = object_point_set
    np.savez_compressed(path, **archive)


def load_calibration_intervals(archive, cameras, verbose=True):
    if verbose:
        print("Loading calibration frame intervals from archive.")
    if CALIBRATION_INTERVALS in archive:
        ranges = archive[CALIBRATION_INTERVALS]
        if len(cameras) != ranges.shape[0]:
            raise ValueError("Need to have the same number of rows in the frame_ranges array as the number of cameras.")
        ix_cam = 0
        for camera in cameras:
            camera.calibration_interval = tuple(ranges[ix_cam])
            ix_cam +=1
    else:
        raise ValueError("No calibration intervals found in the provided archive.")


def save_calibration_intervals(archive, path, cameras, verbose=True):
    if verbose:
        print("Saving calibration intervals to {0:s}".format(path))
    ranges = []
    for camera in cameras:
        if camera.calibration_interval is None:
            raise ValueError("Expecting all cameras to have valid calibration frame ranges. Got: None")
        ranges.append(camera.calibration_interval)
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
    stereo_calib_elem = tree.find("StereoRig")
    return rig.StereoRig.from_xml(stereo_calib_elem)


def load_opencv_single_calibration(path):
    """
    Load single-camera calibration information from xml file
    @type path: str
    @param path: video_path to xml file
    @return calibration info: loaded from the given xml file
    @rtype calib.data.CameraIntrinsics
    """
    tree = etree.parse(path)
    calib_elem = tree.find("CameraIntrinsics")
    return data.CameraIntrinsics.from_xml(calib_elem)


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
    modules = [data, camera, rig]
    object_class = None
    for module in modules:
        if hasattr(module, class_name):
            object_class = getattr(module, class_name)
    if object_class is None:
        raise ValueError("Unexpected calibration format in file {0:s}".format(path))
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
