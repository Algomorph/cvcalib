'''
/home/algomorph/Factory/calib_video_opencv/calib/rig.py.
Created on Mar 24, 2016.
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
'''
from calib.data import CameraIntrinsics, CameraExtrinsics
from lxml import etree

class StereoRig(object):
    
    '''
    Represents the results of a stereo calibration procedure, including all the information
    necessary to stereo-rectify images from the corresponding videos.
    Camera intrinsics <left,right> are always represented by indices <0,1> respectively
    '''
    def __init__(self, intrinsics, extrinsics=CameraExtrinsics(), _id = None):
        '''
        Constructor
        @type intrinsics: tuple[intrinsics.data.CameraIntrinsics]
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
        self.extrinsics = extrinsics
        if(_id is None):
            self.id = StereoRig._unindexed_instance_counter
            StereoRig._unindexed_instance_counter+=1
        else:
            self.id = _id
        
    def to_xml(self, root_element, as_sequence = False):
        '''
        Build an xml node representation of this object under the provided root xml element
        @type root_element:  lxml.etree.SubElement
        @param root_element: the root element to build under
        '''
        if(as_sequence == False):
            elem_name = self.__class__.__name__
        else:
            elem_name = "_"
        stereo_rig_elem = etree.SubElement(root_element, elem_name,
                                             attrib={"id":str(self.id)})
        cameras_elem = etree.SubElement(stereo_rig_elem, "Cameras")
        self.intrinsics[0].to_xml(cameras_elem, as_sequence = True)
        self.intrinsics[1].to_xml(cameras_elem, as_sequence = True)
        self.extrinsics.to_xml(stereo_rig_elem)
        
    def __str__(self):
        return (("{:s}, id: {:s}\n-----CAM0-----\n{:s}\n-----CAM1-----\n{:s}"+
                 "\n--------------\nExtrinsics:\n{:s}\n--------------")
                .format(self.__class__.__name__, str(self.id),str(self.intrinsics[0]),
                        str(self.intrinsics[1]),str(self.extrinsics)))
        
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
        extrinsics = CameraExtrinsics.from_xml(element.find("CameraExtrinsics"))
       
        _id = element.get("id")
        return StereoRig(intrinsics, extrinsics, _id)