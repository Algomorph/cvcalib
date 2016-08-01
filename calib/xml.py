"""
 file_name

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
"""
import re
from lxml import etree  # @UnresolvedImport
import numpy as np


def make_opencv_matrix_xml_element(root, mat, name):
    """
    Construct an xml element out of a numpy matrix formatted for OpenCV XML input
    @type root: lxml.etree.SubElement
    @param root: root xml element to build under
    @type mat: numpy.ndarray
    @param mat: the numpy matrix to convert
    @type name: str
    @param name: name of the matrix XML element
    """
    mat_element = etree.SubElement(root, name, attrib={"type_id": "opencv-matrix"})
    rows_elem = etree.SubElement(mat_element, "rows")
    rows_elem.text = str(mat.shape[0])
    cols_elem = etree.SubElement(mat_element, "cols")
    cols_elem.text = str(mat.shape[1])
    dt_elem = etree.SubElement(mat_element, "dt")
    if mat.dtype == np.dtype('float64'):
        dt_elem.text = "d"
    elif mat.dtype == np.dtype("float32"):
        dt_elem.text = "f"
    else:
        raise ValueError("dtype " + str(mat.dtype) + "not supported. Expecting float64 or float32.")

    data_elem = etree.SubElement(mat_element, "data")
    data_string = str(mat.flatten()).replace("\n", "").replace("[", "").replace("]", "")
    data_string = re.sub("\s+", " ", data_string)
    data_elem.text = data_string
    return mat_element


def make_opencv_size_xml_element(root, sizelike, name):
    if len(sizelike) != 2:
        raise ValueError("Expecting a tuple of length 2. Got length {:d}".format(len(tuple)))
    size_element = etree.SubElement(root, name)
    size_element.text = str(sizelike[0]) + " " + str(sizelike[1])
    return size_element


def parse_xml_matrix(mat_element):
    """
    Generate numpy matrix from opencv-formatted xml of a 2d matrix
    """
    rows = int(mat_element.find("rows").text)
    cols = int(mat_element.find("cols").text)
    type_flag = mat_element.find("dt").text
    if type_flag == "f":
        dtype = np.float32
    elif type_flag == "d":
        dtype = np.float64
    else:
        raise ValueError("dtype flag " + type_flag + " not supported.")
    data_string = mat_element.find("data").text
    data = np.array([float(part) for part in data_string.strip().split(" ") if len(part) > 0])
    return data.reshape((rows, cols)).astype(dtype)
