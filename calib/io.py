'''
Created on Jan 1, 2016

@author: Gregory Kramida
'''
from lxml import etree
import calib.xml as xml
import numpy as np
import calib.geom as geom
import calib.data as data


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
    @rtype calib.data.StereoCalibrationInfo
    '''
    tree = etree.parse(path)
    stereo_calib_elem = tree.find("StereoCalibrationInfo")
    return data.StereoCalibrationInfo(stereo_calib_elem)

def save_opencv_stereo_calibration(path, stereo_calibration_info):
    root = etree.Element("opencv_storage")
    stereo_calibration_info.to_xml(root)
    et = etree.ElementTree(root)
    with open(path,'wb') as f:
        et.write(f,encoding="utf-8",xml_declaration=True, pretty_print=True)
    #little hack necessary to replace the single quotes (that OpenCV doesn't like) with double quotes
    s=open(path).read()
    s = s.replace("'","\"")
    with open(path,'w') as f:
        f.write(s)
        f.flush()

# def load_opencv_stereo_calibration_old(path):
#     tree = etree.parse(path)
#     error = float(tree.find("reprojection_error").text)
#     K1 = xml.parse_xml_matrix(tree.find("K1"))
#     d1 = xml.parse_xml_matrix(tree.find("d1"))
#     K2 = xml.parse_xml_matrix(tree.find("K2"))
#     d2 = xml.parse_xml_matrix(tree.find("d2"))
#     R = xml.parse_xml_matrix(tree.find("R"))
#     T = xml.parse_xml_matrix(tree.find("T"))
#     width = float(tree.find("width").text)
#     height = float(tree.find("height").text)
#     return error, K1, d1, K2, d2, R, T, (height,width)   
    
# def save_opencv_stereo_calibration_old(error, K1, d1, K2, d2, R, T, dims, file_path):
#     root = etree.Element("opencv_storage")
#     width_element = etree.SubElement(root, "width")
#     width_element.text = str(dims[1])
#     width_element = etree.SubElement(root, "height")
#     width_element.text = str(dims[0])
#     K1_element = xml.make_opencv_matrix_xml_element(root, K1, "K1")  # @UnusedVariable
#     d1_element = xml.make_opencv_matrix_xml_element(root, d1, "d1") # @UnusedVariable
#     K2_element = xml.make_opencv_matrix_xml_element(root, K2, "K2") # @UnusedVariable
#     d2_element = xml.make_opencv_matrix_xml_element(root, d2, "d2") # @UnusedVariable
#     R_element = xml.make_opencv_matrix_xml_element(root, R, "R") # @UnusedVariable
#     T_element = xml.make_opencv_matrix_xml_element(root, T, "T") # @UnusedVariable
#     error_element = etree.SubElement(root, "reprojection_error")
#     error_element.text = str(error)
#     et = etree.ElementTree(root)
#     with open(file_path,'wb') as f:
#         et.write(f,encoding="utf-8",xml_declaration=True, pretty_print=True)
#     s=open(file_path).read()
#     s = s.replace("'","\"")
#     with open(file_path,'w') as f:
#         f.write(s)
#         f.flush()