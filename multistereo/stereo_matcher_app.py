"""
Copyright 2016 Gregory Kramida
"""
import cv2
import os.path
import calib.io as cio
from calib.utils import undistort_stereo


class StereoMatcherApp:
    def __init__(self, args):
        self.args = args
        for path in args.images:
            print(os.path.join(args.folder, path))

        self.images = [cv2.imread(os.path.join(args.folder, path)) for path in args.images]
        print(args.input_calibration)
        if args.input_calibration is not None:
            self.rig = cio.load_opencv_calibration(os.path.join(args.folder, args.input_calibration))
            self.images = undistort_stereo(self.rig, self.images[0], self.images[1], 1.8)

    def disparity(self):
        matcher = cv2.StereoBM_create(1024, 7)
        disparity = matcher.compute(cv2.cvtColor(self.images[0], cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(self.images[1], cv2.COLOR_BGR2GRAY))
        self.process_output(disparity)

    def process_output(self, disparity):
        cv8uc = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        if self.args.preview:
            cv2.imshow("disparity", cv8uc)
            cv2.waitKey(0)
        cv2.imwrite(os.path.join(self.args.folder, self.args.output), cv8uc)

    def disparity2(self):
        #matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=384, blockSize=7, P1=1176, P2=4704,
        #                                preFilterCap=16, uniquenessRatio=10)
        # matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=512, blockSize=7, P1=2352, P2=9408,
        #                                 preFilterCap=16)
        matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=384, blockSize=7, P1=2352, P2=18816, #4704
                                         preFilterCap=16)
        disparity = matcher.compute(self.images[0], self.images[1])
        self.process_output(disparity)
