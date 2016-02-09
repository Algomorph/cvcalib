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

    
    args = parser.parse_args()
    if(not osp.isfile(args.calibration_result_file)):
        raise ValueError("The file at " + args.calibration_result_file + " could not be openened.")

    calibration_info = cio.load_opencv_calibration(args.calibration_result_file)
    print(calibration_info)
    

if __name__ == "__main__":
    sys.exit(main())
    