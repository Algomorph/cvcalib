#!/usr/bin/python3

'''
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
'''

#import calib.io as cio
import argparse
import sys
from sync.app import SyncVideoApp
from enum import Enum

class Setting(Enum):
    #settings_file = "settings_file"
    #save_settings = "save_settings"
    folder = "folder"
    videos = "videos"
    output = "output"
    board_width = "board_width"
    board_height = "board_height"
    calibration_clip = "calibration_clip"
    calibration_seek_interval = "calibration_seek_interval"
    trim_end = "trim_end"

def main(argv=None):
    defaults = {
       #Setting.settings_file.name:None,
       #Setting.save_settings.name:False,
       Setting.folder.name:"./",
       Setting.videos.name: ["left.mp4","right.mp4"],
       Setting.output.name: ["out_left.mp4","out_right.mp4"],
       Setting.board_width.name: 9,
       Setting.board_height.name: 6,
       Setting.calibration_clip.name: False,
       Setting.calibration_seek_interval.name: 1.0,
       Setting.trim_end.name:20.0
       }
    
    parser = argparse.ArgumentParser(description='Synchronize two videos based on their sound.')

    #============== INPUT / OUTPUT PATHS ==========================================================#
    parser.add_argument("-f", "--" + Setting.folder.name, help="Path to root folder to work in", 
                        required=False, default=defaults[Setting.folder.name])
    parser.add_argument("-v", "--" + Setting.videos.name,metavar="VIDEO", nargs=2,
                        help="input stereo video tuple (left, right)"+
                        " relative to the 'folder' argument", 
                        required=False, default=defaults[Setting.videos.name])
    parser.add_argument("-o", "--" + Setting.output.name,metavar="OUTPUT", nargs=2,
                        help="where to output synced stereo video tuple (left, right)"+
                        " relative to the 'folder' argument", 
                        required=False, default=defaults[Setting.output.name])
    
    #============== RANGE CLIPPING ================================================================#
    parser.add_argument("-cc", "--" + Setting.calibration_clip.name,
                        help="Seek out where a calibration board first and last appears, and clip "+
                        "using this range in addition to the offset.", action="store_true",
                        default=defaults[Setting.calibration_clip.name])
    
    parser.add_argument("-csi", "--" + Setting.calibration_seek_interval.name, type = float,
                        help="Time intervals (in seconds) to step over during the calibration clip"+
                        " procedure. Higher values will reduce seek time, but result in coarser "+
                        " intervals with more potentially usable frames omitted.",
                        default=defaults[Setting.calibration_seek_interval.name])
    
    parser.add_argument("-te", "--" + Setting.trim_end.name, type = float,
                        help="Time to force-trim away from the end. Typically, relevant for calibration,"+
                        " where cameras are set down and turned off after capture. Use sparingly otherwise.",
                        default=defaults[Setting.trim_end.name])
    
    #============== BOARD DIMENSIONS ==============================================================#
    parser.add_argument("-bw", "--" + Setting.board_width.name, 
                        help="(Calibration only) checkerboard inner corner count across (width)",
                        required = False, default=defaults[Setting.board_width.name], type=int)
    parser.add_argument("-bh", "--" + Setting.board_height.name, 
                        help="(Calibration only) checkerboard inner corner count up (height)",
                        required = False,default=defaults[Setting.board_height.name], type=int)
    
    

    args = parser.parse_args()
    if(len(args.videos) != len(args.output)):
        raise ValueError("The number of input videos specified does not match the number of specified output files.")
    
    app = SyncVideoApp(args)
    app.run_sync()

if __name__ == "__main__":
    sys.exit(main())