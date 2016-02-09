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

def main(argv=None):
    defaults = {
       #Setting.settings_file.name:None,
       #Setting.save_settings.name:False,
       Setting.folder.name:"./",
       Setting.videos.name: ["left.mp4","right.mp4"],
       Setting.output.name: ["out_left.mp4","out_right.mp4"],
       }
    
    parser = argparse.ArgumentParser(description='Synchronize two videos based on their sound.')

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

    args = parser.parse_args()
    app = SyncVideoApp(args)
    app.run_sync()

if __name__ == "__main__":
    sys.exit(main())