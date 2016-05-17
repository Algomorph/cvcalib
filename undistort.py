#!/usr/bin/python3
import calib.io as cio
import argparse
import sys
import os.path as osp
import cv2
from common.args import required_length
from calib.rig import Rig
from calib.camera import Camera
from calib.utils import undistort_stereo


def main():
    parser = argparse.ArgumentParser(description='Simply print out a calibration result in a more' +
                                                 'humanly-readable form.')
    parser.add_argument("-f", "--folder", help="Path to root folder to work in",
                        required=False, default="./")

    parser.add_argument("-c", "--calibration_result_file",
                        help="A calibration result file procded by 'calibrate video opencv.py'",
                        required=False, default="./cvcalib.xml")

    parser.add_argument("-v", "--verbose", help="Print additional information along the way.",
                        required=False, default=False, action='store_true')

    parser.add_argument("-i", "--images", help="Input image(s), lens-distorted", required=False,
                        action=required_length(1, 2), nargs="+", metavar="PATH", default=["left.png, right.png"])
    parser.add_argument("-o", "--output", help="Output file names, undistored image(s)"
                        , metavar="PATH", action=required_length(1, 2), nargs="+", required=False,
                        default=["left_rect.png", "right_rect.png"])
    parser.add_argument("-cf", "--canvas_size_factor", help="Output canvas size, in pixels, will be " +
                                                            "(input_width * canvas_size_factor, input_height * canvas_size_factor)",
                        type=float, default=1.8)

    args = parser.parse_args()

    calibration_info = cio.load_opencv_calibration(osp.join(args.folder, args.calibration_result_file))
    if args.verbose:
        print(calibration_info)

    if type(calibration_info) == Rig and len(calibration_info.cameras) == 2:
        if len(args.images) < 2:
            raise ValueError("Got a stereo rig but less than two input images. Aborting.")
        if len(args.output) < 2:
            raise ValueError("Got a stereo rig but less than two output filenames. Aborting.")

        left = cv2.imread(osp.join(args.folder, args.images[0]))
        right = cv2.imread(osp.join(args.folder, args.images[1]))

        rig = calibration_info
        c0 = rig.cameras[0].intrinsics
        c1 = rig.cameras[1].intrinsics

        if (left.shape[0], left.shape[1]) != c0.resolution:
            raise ValueError("Left image size does not correspond to resolution provided in the calibration file.")

        if (right.shape[0], right.shape[1]) != c1.resolution:
            raise ValueError("Right image size does not correspond to resolution provided in the calibration file.")

        cf = args.canvas_size_factor
        rect_left, rect_right = undistort_stereo(rig, left, right, cf)
        cv2.imwrite(osp.join(args.folder, args.output[0]), rect_left)
        cv2.imwrite(osp.join(args.folder, args.output[1]), rect_right)
    else:
        if len(args.images) > 1:
            print("Warning: provided a single-camera calibration but more than one input image." +
                  " Using only the first one.")

        img = cv2.imread(osp.join(args.folder, args.images[0]))
        if type(calibration_info) == Camera:
            calibration_info = calibration_info.intrinsics
        elif type(calibration_info) == Rig:
            calibration_info = calibration_info.cameras[0].intrinsics
        if (img.shape[0], img.shape[1]) != calibration_info.resolution:
            raise ValueError("Image size does not correspond to resolution provided in the calibration file.")

        cf = args.canvas_size_factor

        map_x, map_y = cv2.initUndistortRectifyMap(calibration_info.intrinsic_mat,
                                                   calibration_info.distortion_coeffs,
                                                   None, None,
                                                   (int(img.shape[1] * cf), int(img.shape[0] * cf)),
                                                   cv2.CV_32FC1)
        out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        cv2.imwrite(osp.join(args.folder, args.output[0]), out)


if __name__ == "__main__":
    sys.exit(main())
