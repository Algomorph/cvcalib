'''
Created on Feb 9, 2016
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

from sync.offset import find_time_offset
from sync.ranges import find_calibration_conversion_range, find_offset_range
from sync.recode import recode_ffmpeg

class SyncVideoApp(object):
    '''
    Application for syncing two videos 
    '''
    def __init__(self, args):
        '''
        Constructor 
        '''
        #(preparatory work goes here)
        self.args = args
        if(not args.calculate_offset_only and args.output[0] == None):
            for ix in range(0,len(args.videos)):
                if("raw_" in args.videos[ix] and len(args.videos[ix]) > 4):
                    args.output[ix] = args.videos[ix].replace("raw_", "")
                else:
                    args.output[ix] = args.videos[ix][:-4] + "_out.mp4"  
            print("Output filenames not set.\n  Setting output filenames to:"+
                  " {0:s}. ATTENTION: will overwrite.".format(str(args.output)))
        self.offset = None
        self.board_dims = (args.board_width,args.board_height)
        
    def calc_offset(self, verbose = True):
        args = self.args
        offset, frame_rate = find_time_offset(args.videos, args.folder, args.audio_delay)
        if(verbose):
            print("Offset: {0:s}".format(str(offset)))
        self.offset = offset; self.frame_rate = frame_rate
        
    def run_sync(self):
        args = self.args
        
        if(not self.offset):
            self.calc_offset()
        
        if(args.calibration_clip):
            frame_ranges, time_ranges = find_calibration_conversion_range(args.videos, args.folder, self.offset, self.board_dims, 
                                                       args.calibration_seek_interval, args.trim_end)
        else:
            frame_ranges, time_ranges = find_offset_range(args.videos, args.folder, self.offset, args.trim_end)
        print("Final clip ranges (in frames): {0:s}".format(str(frame_ranges)))
        recode_ffmpeg(args.videos, args.folder, time_ranges, args.flip, args.output, args.preserve_audio)
        
        
        
        