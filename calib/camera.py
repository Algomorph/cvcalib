"""
/home/algomorph/Factory/calib_video_opencv/intrinsics/video.py.
Created on Mar 21, 2016.
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
import calib.xml as xml
import numpy as np

DEFAULT_RESOLUTION = (1080, 1920)


def _resolution_from_xml(element):
    resolution_elem = element.find("resolution")
    width = int(resolution_elem.find("width").text)
    height = int(resolution_elem.find("height").text)
    return height, width


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
    error_element = etree.SubElement(element, "error")
    error_element.text = str(error)
    time_element = etree.SubElement(element, "time")
    time_element.text = str(time)


class Camera(object):
    """
    Represents a video object & camera that was used to capture it, a wrapper around OpenCV's video_capture
    """

    class Intrinsics(object):
        """
        Represents videos of a camera, i.e. intrinsic matrix & distortion coefficients
        """

        def __init__(self, resolution, intrinsic_mat=None,
                     distortion_coeffs=np.zeros(8, np.float64),
                     error=-1.0, time=0.0):
            """
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
            """
            if intrinsic_mat is None:
                intrinsic_mat = np.eye(3, dtype=np.float64)
                intrinsic_mat[0, 2] = resolution[1] / 2
                intrinsic_mat[1, 2] = resolution[0] / 2
            self.intrinsic_mat = intrinsic_mat
            self.distortion_coeffs = distortion_coeffs
            self.resolution = resolution
            self.error = error
            self.time = time
            self.timestamp = None

        def to_xml(self, root_element, as_sequence=False):
            """
            Build an xml node representation of this object under the provided root xml element
            @type root_element:  lxml.etree.SubElement
            @param root_element: the root element to build under
            @type as_sequence: bool
            @param as_sequence: whether to generate XML for sequences (see OpenCV's documentation on XML/YAML persistence)
            """
            if not as_sequence:
                elem_name = self.__class__.__name__
            else:
                elem_name = "_"
            intrinsics_elem = etree.SubElement(root_element, elem_name)
            _resolution_to_xml(intrinsics_elem, self.resolution)
            xml.make_opencv_matrix_xml_element(intrinsics_elem, self.intrinsic_mat, "intrinsic_mat")
            xml.make_opencv_matrix_xml_element(intrinsics_elem, self.distortion_coeffs, "distortion_coeffs")
            _error_and_time_to_xml(intrinsics_elem, self.error, self.time)

        def __str__(self):
            return (("{:s}\nResolution (h,w): {:s}\n" +
                     "Intrinsic matrix:\n{:s}\nDistortion coefficients:\n{:s}\n" +
                     "Error: {:f}\nTime: {:f}")
                    .format(self.__class__.__name__, str(self.resolution), str(self.intrinsic_mat),
                            str(self.distortion_coeffs), self.error, self.time))

        @staticmethod
        def from_xml(element):
            """
            @type element: lxml.etree.SubElement
            @param element: the element to construct an CameraIntrinsics object from
            @return a new CameraIntrinsics object constructed from XML node with matrices in OpenCV format
            """
            if element is None:
                return Camera.Intrinsics(DEFAULT_RESOLUTION)
            resolution = _resolution_from_xml(element)
            intrinsic_mat = xml.parse_xml_matrix(element.find("intrinsic_mat"))
            distortion_coeffs = xml.parse_xml_matrix(element.find("distortion_coeffs"))
            error, time = _error_and_time_from_xml(element)
            return Camera.Intrinsics(resolution, intrinsic_mat, distortion_coeffs, error, time)

    class Extrinsics(object):
        def __init__(self, rotation=None, translation=None, essential_mat=None,
                     fundamental_mat=None, error=-1.0, time=0.0):
            """
            Constructor
            @type rotation: numpy.ndarray
            @param rotation: 3x3 rotation matrix from camera 0 to camera 1
            @type translation: numpy.ndarray
            @param translation: a 3x1 translation vector from camera 0 to camera 1
            @type error: float
            @param error: mean square distance error to object points after reprojection
            @type  time: float
            @param time: calibration time in seconds
            """
            if rotation is None:
                rotation = np.eye(3, dtype=np.float64)
            if translation is None:
                translation = np.zeros((1,3), np.float64)
            if essential_mat is None:
                essential_mat = np.eye(3, dtype=np.float64)
            if fundamental_mat is None:
                fundamental_mat = np.eye(3, dtype=np.float64)
            self.rotation = rotation
            self.translation = translation
            self.essential_mat = essential_mat
            self.fundamental_mat = fundamental_mat
            self.error = error
            self.time = time
            self.timestamp = None

        def __str__(self):
            return (("{:s}\nRotation:\n{:s}\nTranslation:\n{:s}\nEssential Matrix:\n{:s}" +
                     "\nFundamental Matrix:\n{:s}\nError: {:f}\nTime: {:f}")
                    .format(self.__class__.__name__, str(self.rotation),
                            str(self.translation), str(self.essential_mat),
                            str(self.fundamental_mat), self.error, self.time))

        def to_xml(self, root_element, as_sequence=False):
            """
            Build an xml node representation of this object under the provided root xml element
            @type root_element:  lxml.etree.SubElement
            @param root_element: the root element to build under
            @type as_sequence: bool
            @param as_sequence: whether to generate XML for sequences (see OpenCV's documentation on XML/YAML persistence)

            """
            if not as_sequence:
                elem_name = self.__class__.__name__
            else:
                elem_name = "_"

            extrinsics_elem = etree.SubElement(root_element, elem_name)

            xml.make_opencv_matrix_xml_element(extrinsics_elem, self.rotation, "rotation")
            xml.make_opencv_matrix_xml_element(extrinsics_elem, self.translation, "translation")
            xml.make_opencv_matrix_xml_element(extrinsics_elem, self.essential_mat, "essential_mat")
            xml.make_opencv_matrix_xml_element(extrinsics_elem, self.fundamental_mat,
                                               "fundamental_mat")
            _error_and_time_to_xml(extrinsics_elem, self.error, self.time)

        @staticmethod
        def from_xml(element):
            """
            Build a CameraExtrinsics object out of the provided XML node with matrices in
            OpenCV format
            @type element: lxml.etree.SubElement
            @param element: the element to construct an StereoRig object from
            @rtype: calib.CameraExtrinsics|None
            @return a new StereoRig object constructed from given XML node, None if element is None
            """
            if element is None:
                return Camera.Extrinsics()
            rotation = xml.parse_xml_matrix(element.find("rotation"))
            translation = xml.parse_xml_matrix(element.find("translation"))
            essential_mat = xml.parse_xml_matrix(element.find("essential_mat"))
            fundamental_mat = xml.parse_xml_matrix(element.find("fundamental_mat"))
            error, time = _error_and_time_from_xml(element)
            return Camera.Extrinsics(rotation, translation, essential_mat,
                                     fundamental_mat, error, time)

    def __init__(self, resolution=None, intrinsics=None, extrinsics=None):
        """
        Build a camera with the specified parameters
        """
        if resolution is None:
            resolution = DEFAULT_RESOLUTION
        if intrinsics is None:
            self.intrinsics = Camera.Intrinsics(resolution)
        else:
            self.intrinsics = intrinsics
        if extrinsics is None:
            self.extrinsics = Camera.Extrinsics()
        else:
            self.extrinsics = extrinsics

    def copy(self):
        return Camera(intrinsics=self.intrinsics, extrinsics=self.extrinsics)

    def to_xml(self, root_element, as_sequence=False):
        """
        Build an xml node representation of this object under the provided root xml element
        @type as_sequence: bool
        @param as_sequence: use sequence opencv XML notation, i.e. XML element name set to "_"
        @type root_element:  lxml.etree.SubElement
        @param root_element: the root element to build under
        """
        if not as_sequence:
            elem_name = self.__class__.__name__
        else:
            elem_name = "_"
        camera_elem = etree.SubElement(root_element, elem_name)
        if self.intrinsics:
            self.intrinsics.to_xml(camera_elem, False)
        if self.extrinsics and self.extrinsics.error > 0.0:
            self.extrinsics.to_xml(camera_elem, False)

    def __str__(self, *args, **kwargs):
        if self.extrinsics.error > 0.0:
            extrinsics_string = "\n" + str(self.extrinsics)
        else:
            extrinsics_string = ""
        return Camera.__name__ + "\n" + str(self.intrinsics) + extrinsics_string

    @staticmethod
    def from_xml(element):
        """
        @type element: lxml.etree.SubElement
        @param element: the element to construct an CameraIntrinsics object from
        @return a new Camera object constructed from XML node with matrices in OpenCV format
        """
        intrinsics_elem = element.find(Camera.Intrinsics.__name__)
        if intrinsics_elem:
            intrinsics = Camera.Intrinsics.from_xml(intrinsics_elem)
        else:
            intrinsics = Camera.Intrinsics(DEFAULT_RESOLUTION)
        extrinsics_elem = element.find(Camera.Extrinsics.__name__)
        if extrinsics_elem:
            extrinsics = Camera.Extrinsics.from_xml(extrinsics_elem)
        else:
            extrinsics = Camera.Extrinsics()
        return Camera(intrinsics, extrinsics)
