'''
Created on Jan 1, 2016

@author: Gregory Kramida
'''
from lxml import etree#@UnresolvedImport
import numpy as np
import calib.geom as geom
import calib.data as data


#TODO: corner saving/loading for whatever reason is now broken, fix
def load_corners(path, board_height = None, board_width = None, board_square_size = None):
    npzfile = np.load(path)
    if 'object_point_set' in npzfile:
        objp = npzfile['object_point_set']
    else:
        objp = geom.generate_object_points(board_height, board_width, board_square_size)
    imgpoints = []
    npzfile.files.remove('object_point_set')
    npzfile.files.sort()
    for array_name in npzfile.files:
        imgpoints.append(npzfile[array_name])
        
    objpoints = []
    usable_frame_ct = len(imgpoints[0])
    for i_frame in range(usable_frame_ct): # @UnusedVariable
        objpoints.append(objp)
        
    return imgpoints,objpoints,usable_frame_ct

def load_opencv_stereo_calibration(path):
    '''
    Load stereo calibration information from xml file
    @type path: str
    @param path: path to xml file
    @return stereo calibration: loaded from the given xml file
    @rtype calib.data.StereoExtrinsics
    '''
    tree = etree.parse(path)
    stereo_calib_elem = tree.find("StereoExtrinsics")
    return data.StereoExtrinsics.from_xml(stereo_calib_elem)

def load_opencv_single_calibration(path):
    '''
    Load single-camera calibration information from xml file
    @type path: str
    @param path: path to xml file
    @return calibration info: loaded from the given xml file
    @rtype calib.data.CameraIntrinsics
    '''
    tree = etree.parse(path)
    calib_elem = tree.find("CameraIntrinsics")
    return data.CameraIntrinsics.from_xml(calib_elem)

def load_opencv_calibration(path):
    '''
    Load any kind (stereo or single) of calibration result from the file
    @type path: str
    @param path: path to xml file
    @return calibration info: loaded from the given xml file
    @rtype calib.data.CameraIntrinsics | calib.data.StereoExtrinsics
    '''
    tree = etree.parse(path)
    calib_elem = tree.find("CameraIntrinsics")
    if(calib_elem is not None):
        calib_info = data.CameraIntrinsics.from_xml(calib_elem)
    else:
        stereo_calib_elem = tree.find("StereoExtrinsics")
        if(stereo_calib_elem is None):
            raise ValueError("Unexpected calibration format in file {0:s}".format(path))
        calib_info = data.StereoExtrinsics.from_xml(stereo_calib_elem)
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