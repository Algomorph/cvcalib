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
from sync.convert import find_calibration_conversion_range, find_offset_range

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
        self.board_dims = (args.board_width,args.board_height)
        
    def run_sync(self):
        args = self.args
        
        offset, frame_rate = find_time_offset(args.videos[0], args.videos[1], args.folder)#@UnusedVariable
        print("Offset: {0:s}".format(str(offset)))
        
        if(args.calibration_clip):
            frame_ranges, time_ranges = find_calibration_conversion_range(args.videos, args.folder, offset, self.board_dims, 
                                                       args.calibration_seek_interval, args.trim_end)
        else:
            frame_ranges, time_ranges = find_offset_range(args.videos, args.folder, offset, args.trim_end)
        print("Final clip ranges (in frames): {0:s}".format(str(frame_ranges)))
        
        time_ranges
        
        