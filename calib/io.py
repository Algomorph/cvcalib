'''
Created on Jan 1, 2016

@author: Gregory Kramida
'''
from lxml import etree#@UnresolvedImport
import numpy as np
from calib import data, camera, geom, rig
from calib.camera import Pose


IMAGE_POINTS = "image_points"
FRAME_NUMBERS = "frame_numbers"
OBJECT_POINT_SET = "object_point_set"
POSES = "poses"

#TODO: corner saving/loading for whatever reason is now broken, fix
def load_corners(path, videos, board_height = None, 
                 board_width = None, board_square_size = None, 
                 verbose = True):
    if(verbose):
        print("Loading corners from {0:s}".format(path))    
    archive = np.load(path)
    if OBJECT_POINT_SET in archive:
        objp = archive[OBJECT_POINT_SET]
    else:
        objp = geom.generate_object_points(board_height, board_width, board_square_size)
    archive.files.remove(OBJECT_POINT_SET)
    
    if FRAME_NUMBERS in archive:
        frame_numbers = archive[FRAME_NUMBERS]
        archive.files.remove(FRAME_NUMBERS)
    else:
        frame_numbers = None
    
    archive.files.sort()
    for array_name in archive.files:
        if(array_name.startswith(IMAGE_POINTS)):
            ix_vid = int(array_name[len(IMAGE_POINTS):])
            videos[ix_vid].imgpoints = archive[array_name]
        elif(array_name.startswith(FRAME_NUMBERS)):
            ix_vid = int(array_name[len(FRAME_NUMBERS):])
            videos[ix_vid].usable_frames = {}
            i_key = 0
            for key in archive[array_name]:
                videos[ix_vid].usable_frames[key] = i_key
                i_key += 1 
        elif(array_name.startswith(POSES)):
            ix_vid = int(array_name[len(POSES):])
            #process poses
            videos[ix_vid].poses = [Pose(T) for T in archive[array_name]]
        
    return objp, frame_numbers

def save_corners(path, cameras, object_point_set, verbose = True):
    if(verbose):
        print("Saving corners to {0:s}".format(path))
    file_dict = {}
    for camera in cameras:
        file_dict[IMAGE_POINTS+str(camera.index)] = camera.imgpoints
        file_dict[FRAME_NUMBERS+str(camera.index)] = list(camera.usable_frames.keys())
        if(len(camera.poses)>0):
            file_dict[POSES+str(camera.index)] = np.array([pose.T for pose in camera.poses])
             
    file_dict[OBJECT_POINT_SET]=object_point_set
    np.savez_compressed(path,**file_dict) 

def load_opencv_stereo_calibration(path):
    '''
    Load stereo calibration information from xml file
    @type video_path: str
    @param video_path: video_path to xml file
    @return stereo calibration: loaded from the given xml file
    @rtype calib.data.StereoRig
    '''
    tree = etree.parse(path)
    stereo_calib_elem = tree.find("StereoRig")
    return rig.StereoRig.from_xml(stereo_calib_elem)

def load_opencv_single_calibration(path):
    '''
    Load single-camera calibration information from xml file
    @type video_path: str
    @param video_path: video_path to xml file
    @return calibration info: loaded from the given xml file
    @rtype calib.data.CameraIntrinsics
    '''
    tree = etree.parse(path)
    calib_elem = tree.find("CameraIntrinsics")
    return data.CameraIntrinsics.from_xml(calib_elem)

def load_opencv_calibration(path):
    '''
    Load any kind (stereo or single) of calibration result from the file
    @type video_path: str
    @param video_path: video_path to xml file
    @return calibration info: loaded from the given xml file
    @rtype calib.data.CameraIntrinsics | calib.data.StereoRig
    ''' 
    tree = etree.parse(path)
    first_elem = tree.getroot().getchildren()[0]
    class_name = first_elem.tag
    modules = [data, camera, rig]
    object_class = None
    for module in modules:
        if(hasattr(module, class_name)):
            object_class = getattr(module, class_name)
    if(object_class is None):
        raise ValueError("Unexpected calibration format in file {0:s}".format(path))
    calib_info = object_class.from_xml(first_elem)
    return calib_info

    
def save_opencv_calibration(path, calibration_info):
    root = etree.Element("opencv_storage")
    calibration_info.to_xml(root)
    et = etree.ElementTree(root)
    with open(path,'wb') as f:
        et.write(f,encoding="utf-8",xml_declaration=True, pretty_print=True)
    #little hack necessary to replace the single quotes (that OpenCV doesn't like) with double quotes
    s=open(path).read()
    s = s.replace("'","\"")
    with open(path,'w') as f:
        f.write(s)
        f.flush()