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

class Rig(object):
    """
    Represents the results of a stereo calibration procedure, including all the information
    necessary to stereo-rectify images from the corresponding videos.
    Camera cameras <left,right> are always represented by indices <0,1> respectively
    """
    def __init__(self, cameras=()):
        """
        Constructor
        @type cameras: tuple[calib.camera.Camera]
        @param cameras: tuple composed of two videos of the stereo camera pair.
        @type extrinsics: calib.data.CameraExtrinsics
        @param extrinsics: extrinsic parameters, representing transformation of camera 1 from
        camera 0, as well as the essential & fundamental matrices of this relationship
        """
        self.cameras = cameras
        
    def to_xml(self, root_element, as_sequence=False):
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
        stereo_rig_elem = etree.SubElement(root_element, elem_name)
        cameras_elem = etree.SubElement(stereo_rig_elem, "Cameras")
        # TODO: change serialization scheme to serialize cameras directly when that becomes possible
        for camera in self.cameras:
            camera.to_xml(cameras_elem, as_sequence=True)
        
    def __str__(self):
        representation = "=====" + self.__class__.__name__ + "====="
        ix_camera = 0
        for camera in self.cameras:
            representation += "\n-----CAM{:d}-----\n{:s}".format(ix_camera, str(camera))
            ix_camera += 1
        representation += "\n--------------"
        return representation
        
    @staticmethod
    def from_xml(element):
        """
        Build a StereoRig object out of the given xml node
        @type element: lxml.etree.SubElement
        @param element: the element to construct an StereoRig object from
        @return a new StereoRig object constructed from XML node with matrices in
        OpenCV format
        """
        use_old_format = element.find("CameraExtrinsics") is not None
        if use_old_format:
            extrinsics0 = Camera.Extrinsics()
            extrinsics1 = Camera.Extrinsics.from_xml(element.find("CameraExtrinsics"))
            cameras_elem = element.find("Cameras")
            cameras = (Camera.from_xml(cameras_elem[0]), Camera.from_xml(cameras_elem[1]))
            cameras[0].extrinsics = extrinsics0
            cameras[1].extrinsics = extrinsics1
        else:
            cameras_element = element.find("Cameras")
            cameras = tuple([Camera.from_xml(camera_element) for camera_element in cameras_element])
        return Rig(cameras)

