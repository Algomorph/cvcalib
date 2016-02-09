import calib.io as cio
import argparse
import sys
import os.path as osp


def main(argv=None):
    parser = argparse.ArgumentParser(description='Simply print out a calibration result in a more'+
                                     'humanly-readable form.')

    parser.add_argument("-f", "--calibration_result_file", 
                        help="A calibration result file procded by 'calibrate video opencv.py'", 
                        required=False, default= "./cvcalib.xml")
    
    parser.add_argument("-s", "--single_camera",
                        help="Whether to expect a single-camera (non-stereo) result only.",
                        required=False, action="store_true",default=False)
    
    args = parser.parse_args()
    if(not osp.isfile(args.calibration_result_file)):
        raise ValueError("The file at " + args.calibration_result_file + " could not be openened.")
    if(args.single_camera):
        calibration_info = cio.load_opencv_signle_calibration(args.calibration_result_file)
    else:
        calibration_info = cio.load_opencv_stereo_calibration(args.calibration_result_file)
    
    print(calibration_info)
    

if __name__ == "__main__":
    sys.exit(main())
    