"""
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
"""

from lxml import etree
from calib.camera import Camera
from calib.data import CameraExtrinsics
import cv2


# TODO: StereoRig & Camera should have a single, abstract ancestor to enforce interface compliance,
# such as filtering functions


class StereoRig(object):
    # TODO: get rid of instance counter, initialize with current date instead (always!).

    __unindexed_instance_counter = 0

    '''
    Represents the results of a stereo calibration procedure, including all the information
    necessary to stereo-rectify images from the corresponding videos.
    Camera cameras <left,right> are always represented by indices <0,1> respectively
    '''
    def __init__(self, cameras, extrinsics=CameraExtrinsics(), _id = None):
        """
        Constructor
        @type cameras: tuple[calib.camera.Camera]
        @param cameras: tuple composed of two videos of the stereo camera pair.
        @type extrinsics: calib.data.CameraExtrinsics
        @param extrinsics: extrinsic parameters, representing transformation of camera 1 from
        camera 0, as well as the essential & fundamental matrices of this relationship
        """
        self.cameras = cameras
        self.extrinsics = extrinsics
        if _id is None:
            self.id = StereoRig.__unindexed_instance_counter
            StereoRig.__unindexed_instance_counter += 1
        else:
            self.id = _id
        
    def to_xml(self, root_element, as_sequence = False):
        """
        Build an xml node representation of this object under the provided root xml element
        @type as_sequence: bool
        @param as_sequence: whether to write to OpenCV sequence XML format, i.e. with "_" as element name
        @type root_element:  lxml.etree.SubElement
        @param root_element: the root element to build under
        """
        if not as_sequence:
            elem_name = self.__class__.__name__
        else:
            elem_name = "_"
        stereo_rig_elem = etree.SubElement(root_element, elem_name,
                                             attrib={"id":str(self.id)})
        cameras_elem = etree.SubElement(stereo_rig_elem, "Cameras")
        # TODO: change serialization scheme to serialize cameras directly when that becomes possible
        self.cameras[0].to_xml(cameras_elem, as_sequence = True)
        self.cameras[1].to_xml(cameras_elem, as_sequence = True)
        self.extrinsics.to_xml(stereo_rig_elem)
        
    def __str__(self):
        return (("{:s}, id: {:s}\n-----CAM0-----\n{:s}\n-----CAM1-----\n{:s}"+
                 "\n--------------\nExtrinsics:\n{:s}\n--------------")
                .format(self.__class__.__name__, str(self.id),str(self.cameras[0]),
                        str(self.cameras[1]),str(self.extrinsics)))
        
    @staticmethod
    def from_xml(element):
        """
        Build a StereoRig object out of the given xml node
        @type element: lxml.etree.SubElement
        @param element: the element to construct an StereoRig object from
        @return a new StereoRig object constructed from XML node with matrices in
        OpenCV format
        """
        cameras_elem = element.find("Cameras")
        cameras = [Camera.from_xml(cameras_elem[0]), Camera.from_xml(cameras_elem[1])]
        extrinsics = CameraExtrinsics.from_xml(element.find("CameraExtrinsics"))
       
        _id = element.get("id")
        return StereoRig(cameras, extrinsics, _id)

    def filter_basic_stereo(self, board_dims):
        l_frame = self.cameras[0].frame
        r_frame = self.cameras[1].frame

        lfound, lcorners = cv2.findChessboardCorners(l_frame, board_dims)
        rfound, rcorners = cv2.findChessboardCorners(r_frame, board_dims)
        if not (lfound and rfound):
            return False

        self.cameras[0].current_image_points = lcorners
        self.cameras[1].current_image_points = rcorners

        return True