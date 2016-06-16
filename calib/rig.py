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
from calib.geom import Pose
import numpy as np


class Rig(object):
    """
    Represents the results of a stereo calibration procedure, including all the information
    necessary to stereo-rectify images from the corresponding videos.
    Camera cameras <left,right> are always represented by indices <0,1> respectively
    """

    CAMERAS_ELEMENT_TAG = Camera.__name__ + "s"

    def __init__(self, cameras=(), extrinsics=None):
        """
        Constructor
        @type cameras: tuple[calib.camera.Camera]
        @param cameras: tuple composed of several cameras. The cameras extrinsics are exptected to be relative to the
        @param extrinsics: extrinsic parameters that may serve as 'global positioning' within a larger rig
        first camera in the tuple.
        """
        self.cameras = cameras
        self.extrinsics = extrinsics

    def __str__(self):
        representation = "=====" + self.__class__.__name__ + "====="
        ix_camera = 0
        for camera in self.cameras:
            representation += "\n-----CAM{:d}-----\n{:s}".format(ix_camera, str(camera))
            ix_camera += 1
        if self.extrinsics is not None:
            representation += "\n----RIG EXTRINSICS----\n{:s}".format(str(self.extrinsics))
        representation += "\n--------------"

        return representation

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
        rig_elem = etree.SubElement(root_element, elem_name)
        cameras_elem = etree.SubElement(rig_elem, Rig.CAMERAS_ELEMENT_TAG)
        if self.extrinsics is not None:
            self.extrinsics.to_xml(rig_elem, as_sequence=False)

        for camera in self.cameras:
            camera.to_xml(cameras_elem, as_sequence=True)

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
            cameras_elem = element.find(Rig.CAMERAS_ELEMENT_TAG)
            cameras = (Camera.from_xml(cameras_elem[0]), Camera.from_xml(cameras_elem[1]))
            cameras[0].extrinsics = extrinsics0
            cameras[1].extrinsics = extrinsics1
            extrinsics = None
        else:
            cameras_element = element.find(Rig.CAMERAS_ELEMENT_TAG)
            cameras = tuple([Camera.from_xml(camera_element) for camera_element in cameras_element])
            extrinsics_element = element.find("Extrinsics")
            if extrinsics_element is not None:
                extrinsics = Camera.Extrinsics.from_xml(extrinsics_element)
            else:
                extrinsics = None
        return Rig(cameras, extrinsics)


class MultiStereoRig(object):
    """
    Represents a combination of stereo rigs
    """
    RIGS_ELEMENT_TAG = Rig.__name__ + "s"

    def __init__(self, stereo_rigs=None, rig=None):
        """
        Construct a multi-stereo rig out of either a collection of stereo rigs or a single rig
        @type stereo_rigs: tuple[calib.rig.Rig]
        @param stereo_rigs: rigs with two cameras each and extrinsics with reference to the global coordinate frame,
         i.e. the first camera of the first rig
        @type rig: calib.rig.Rig
        @param rig: a regular rig to convert to a multistereo rig. Assumes stereo camera pairs are in order in the input
        rig.
        """
        if stereo_rigs is not None and rig is None:
            ix_rig = 0
            for rig in stereo_rigs:
                if len(rig.cameras) != 2:
                    raise ValueError("Each of the stereo rigs should only have two cameras")
                if rig.extrinsics is None and ix_rig != 0:
                    raise ValueError("Expecting valid extrinsics for rig at position {:d} relative to the first rig."
                                     .format(ix_rig))
                ix_rig += 1
            self.rigs = stereo_rigs
        elif rig is not None and stereo_rigs is None:
            # convert an arbitrary rig to a multistereo rig.
            if len(rig.cameras) % 2 != 0 and len(rig.cameras) != 0:
                raise ValueError("Expecting a non-zero even number of cameras in the input rig.")

            rigs = [Rig(cameras=(rig.cameras[0], rig.cameras[1], Camera.Extrinsics()))]

            # find local extrinsics
            for ix_cam in range(2, len(rig.cameras), 2):
                left_cam = rig.cameras[ix_cam]
                right_cam = rig.cameras[ix_cam + 1]
                left_extrinsics = left_cam.extrinsics
                right_extrinsics = right_cam.extrinsics
                left_pose = Pose(rotation=left_extrinsics.rotation, translation_vector=left_extrinsics.translation)
                right_pose = Pose(rotation=right_extrinsics.rotation, translation_vector=right_extrinsics.translation)
                local_pose = Pose(transform=np.linalg.inv(left_pose.T).dot(right_pose.T))
                local_right_extrinsics = Camera.Extrinsics(rotation=local_pose.rmat, translation=local_pose.rvec)

                stereo_rig_extrinsics = left_cam.extrinsics
                left_cam.extrinsics = Camera.Extrinsics()
                right_cam.extrinsics = local_right_extrinsics
                rigs.append(Rig(cameras=(left_cam, right_cam), extrinsics=stereo_rig_extrinsics))

            self.rigs = tuple(rigs)
        else:
            raise ValueError("Either the ['stereo_rigs' and 'stereo_rig_extrinsics'] OR the 'rig' argument should be "
                             "provided and not None.")

    def __str__(self):
        representation = "=====" + self.__class__.__name__ + "====="
        ix_rig = 0
        for rig in self.rigs:
            representation += "\n-----RIG{:d}-----\n{:s}".format(ix_rig, str(rig))
            ix_rig += 1
        representation += "\n--------------"
        return representation

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
        rig_elem = etree.SubElement(root_element, elem_name)
        cameras_elem = etree.SubElement(rig_elem, MultiStereoRig.RIGS_ELEMENT_TAG)

        for rig in self.rigs:
            rig.to_xml(cameras_elem, as_sequence=True)

    @staticmethod
    def from_xml(element):
        """
        Build a StereoRig object out of the given xml node
        @type element: lxml.etree.SubElement
        @param element: the element to construct an StereoRig object from
        @return a new StereoRig object constructed from XML node with matrices in
        OpenCV format
        """
        rigs_element = element.find(MultiStereoRig.RIGS_ELEMENT_TAG)
        rigs = tuple([Rig.from_xml(rig_element) for rig_element in rigs_element])
        return MultiStereoRig(rigs)
