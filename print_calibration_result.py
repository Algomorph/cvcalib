from calib.data import StereoCalibrationInfo, CameraCalibrationInfo
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='Delete all right frame files in folder based on which left frame files remain.')

    parser.add_argument("-f", "--calibration_result_file", 
                        help="A calibration result file procded by 'calibrate video opencv.py'", 
                        required=False, default= "./cvcalib.xml")
    
    parser.add_argument("-s", "--single_camera",
                        help="Whether to expect a single-camera (non-stereo) result only.",
                        required=False, action="store_true",default=False)
    
    args = parser.parse_args()
    args.
    

if __name__ == "__main__":
    sys.exit(main())
    