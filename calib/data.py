'''
 data.py

 @author: Gregory Kramida
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
'''
import calib.xml as xml
from lxml import etree#@UnresolvedImport
import numpy as np
import os.path as osp
import cv2#@UnresolvedImport


def _resolution_from_xml(element):
    resolution_elem = element.find("resolution")
    width = int(resolution_elem.find("width").text)
    height = int(resolution_elem.find("height").text)
    return (height, width)

def _resolution_to_xml(element, resolution):
    resolution_elem = etree.SubElement(element, "resolution")
    width_elem = etree.SubElement(resolution_elem, "width")
    width_elem.text = str(resolution[1])
    height_elem = etree.SubElement(resolution_elem, "height")
    height_elem.text = str(resolution[0])

def _error_and_time_from_xml(element):
    error = float(element.find("error").text)
    time = float(element.find("time").text)
    return error, time

def _error_and_time_to_xml(element, error, time):
    error_element = etree.SubElement(element,"error")
    error_element.text = str(error)
    time_element = etree.SubElement(element,"time")
    time_element.text = str(time)



class CameraIntrinsics(object):
    '''
    Represents videos of a camera, i.e. intrinsic matrix & distortion coefficients
    '''
    def __init__(self, resolution, intrinsic_mat = None, 
                 distortion_coeffs = np.zeros(8,np.float64), 
                 error = -1.0, time = 0.0, index = 0):
        '''
        Constructor
        @type intrinsic_mat: numpy.ndarray
        @param intrinsic_mat: intrinsic matrix (3x3)
        @type distortion_coeffs: numpy.ndarray
        @param distortion_coeffs: distortion coefficients (1x8)
        @type resolution: tuple[int]
        @param resolution: pixel resolution (height,width) of the camera
        @type error: float
        @param error: mean square distance error to object points after reprojection
        @type  time: float
        @param time: calibration time in seconds
        '''
        if(type(intrinsic_mat) == type(None)):
            intrinsic_mat = np.eye(3,dtype=np.float64)
        self.intrinsic_mat = intrinsic_mat
        self.distortion_coeffs = distortion_coeffs
        self.resolution = resolution
        self.error = error
        self.time = time
        self.index = index
    
    def to_xml(self, root_element, as_sequence = False):
        '''
        Build an xml node representation of this object under the provided root xml element
        @type root_element:  lxml.etree.SubElement
        @param root_element: the root element to build under
        '''
        if(as_sequence == False):
            elem_name = "CameraIntrinsics"
        else:
            elem_name = "_"
        intrinsics_elem = etree.SubElement(root_element,elem_name,attrib={"index":str(self.index)})
        _resolution_to_xml(intrinsics_elem, self.resolution)
        xml.make_opencv_matrix_xml_element(intrinsics_elem, self.intrinsic_mat, "intrinsic_mat")
        xml.make_opencv_matrix_xml_element(intrinsics_elem, self.distortion_coeffs, "distortion_coeffs")
        _error_and_time_to_xml(intrinsics_elem, self.error, self.time)
        
    def __str__(self):
        return (("Camera Calibration Info, index: {0:d}\nResolution (h,w): {1:s}\n"+
                 "Intrinsic matrix:\n{2:s}\nDistortion coefficients:\n{3:s}\n"+
                 "Error: {4:f}\nTime: {5:f}")
                .format(self.index,str(self.resolution),str(self.intrinsic_mat),
                        str(self.distortion_coeffs),self.error,self.time))
    
    @staticmethod
    def from_xml(element):
        '''
        @type element: lxml.etree.SubElement
        @param element: the element to construct an CameraIntrinsics object from
        @return a new CameraIntrinsics object constructed from XML node with matrices in OpenCV format
        '''
        resolution = _resolution_from_xml(element)
        intrinsic_mat = xml.parse_xml_matrix(element.find("intrinsic_mat"))
        distortion_coeffs = xml.parse_xml_matrix(element.find("distortion_coeffs"))
        error, time = _error_and_time_from_xml(element)
        index = int(element.get("index"))
        return CameraIntrinsics(resolution, intrinsic_mat, distortion_coeffs, error, time, index)
    
#TODO: StereoRig should contain video objects instead of intrinsics and be in its own separate module
class StereoRig(object):
    _unnamed_instance_counter = 0
    '''
    Represents the results of a stereo calibration procedure, including all the information
    necessary to stereo-rectify images from the corresponding videos.
    Camera intrinsics <left,right> are always represented by indices <0,1> respectively
    '''
    def __init__(self, intrinsics, rotation = np.eye(3,dtype=np.float64), 
                 translation = np.zeros(3,np.float64), essential_mat = np.eye(3,dtype=np.float64), 
                 fundamental_mat = np.eye(3,dtype=np.float64), error = -1.0, time = 0.0, _id = None):
        '''
        Constructor
        @type intrinsics: tuple[calib.data.CameraIntrinsics]
        @param intrinsics: tuple composed of two videos of the stereo camera pair.
        @type rotation: numpy.ndarray
        @param rotation: 3x3 rotation matrix from camera 0 to camera 1
        @type translation: numpy.ndarray
        @param translation: a 3x1 translation vector from camera 0 to camera 1
        @type error: float
        @param error: mean square distance error to object points after reprojection
        @type  time: float
        @param time: calibration time in seconds
        '''
        self.intrinsics = intrinsics
        self.rotation = rotation
        self.translation = translation
        self.essential_mat = essential_mat
        self.fundamental_mat = fundamental_mat
        self.error = error
        self.time = time
        if(_id is None):
            self.id = StereoRig._unnamed_instance_counter
            StereoRig._unnamed_instance_counter+=1
        else:
            self.id = _id
        
    def to_xml(self, root_element, as_sequence = False):
        '''
        Build an xml node representation of this object under the provided root xml element
        @type root_element:  lxml.etree.SubElement
        @param root_element: the root element to build under
        '''
        if(as_sequence == False):
            elem_name = "StereoRig"
        else:
            elem_name = "_"
        stereo_calib_elem = etree.SubElement(root_element, elem_name,
                                             attrib={"id":str(self.id)})
        cameras_elem = etree.SubElement(stereo_calib_elem, "Cameras")
        self.intrinsics[0].to_xml(cameras_elem, as_sequence = True)
        self.intrinsics[1].to_xml(cameras_elem, as_sequence = True)
        xml.make_opencv_matrix_xml_element(stereo_calib_elem, self.rotation, "rotation")
        xml.make_opencv_matrix_xml_element(stereo_calib_elem, self.translation, "translation")
        xml.make_opencv_matrix_xml_element(stereo_calib_elem, self.essential_mat, "essential_mat")
        xml.make_opencv_matrix_xml_element(stereo_calib_elem, self.fundamental_mat,
                                           "fundamental_mat")
        _error_and_time_to_xml(stereo_calib_elem, self.error, self.time)
        
    def __str__(self):
        return (("Stereo Calibration Info, id: {0:s}\n-----CAM0-----\n{1:s}\n-----CAM1-----\n{2:s}"+
                 "\n--------------\nRotation:\n{3:s}\nTranslation:\n{4:s}\nEssential Matrix:\n{5:s}"
                 +"\nFundamental Matrix:\n{6:s}\nError: {7:f}\nTime: {8:f}")
                .format(str(self.id),str(self.intrinsics[0]),str(self.intrinsics[1]),str(self.rotation),
                        str(self.translation),str(self.essential_mat),str(self.fundamental_mat),
                        self.error,self.time))
        
    @staticmethod
    def from_xml(element):
        '''
        Build a StereoRig object out of the given xml node
        @type element: lxml.etree.SubElement
        @param element: the element to construct an StereoRig object from
        @return a new StereoRig object constructed from XML node with matrices in 
        OpenCV format
        '''
        cameras_elem = element.find("Cameras")
        intrinsics = []
        intrinsics.append(CameraIntrinsics.from_xml(cameras_elem[0]))
        intrinsics.append(CameraIntrinsics.from_xml(cameras_elem[1]))
        rotation = xml.parse_xml_matrix(element.find("rotation"))
        translation = xml.parse_xml_matrix(element.find("translation"))
        essential_mat = xml.parse_xml_matrix(element.find("essential_mat"))
        fundamental_mat = xml.parse_xml_matrix(element.find("fundamental_mat"))
        error, time = _error_and_time_from_xml(element)
        _id = element.get("id")
        return StereoRig(intrinsics, rotation, translation, essential_mat,
                                     fundamental_mat, error, time)
        
        
