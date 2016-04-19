'''
/home/algomorph/Factory/calib_video_opencv/audiosync/recode.py.
Created on Feb 11, 2016.
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
import os.path as osp
from subprocess import call

def recode_ffmpeg(video_filenames, folder, time_ranges, flip_list, output_filenames, 
                  preserve_sound = False, verbose = False):
    ix_vid = 0
    for fn in video_filenames:
        print("\n=============Recoding {0:s} to {1:s}=================\n".format(fn, output_filenames[ix_vid]))
        video_path = osp.join(folder, fn)
        output_path = osp.join(folder, output_filenames[ix_vid])
        time_range = time_ranges[ix_vid]
        flip = flip_list[ix_vid]
        args = ["ffmpeg", "-y", "-i", video_path]
        #clip start
        args += ["-ss",  str(time_range[0])]
        #clip end
        args += ["-to", str(time_range[1])]
        #flip/no flip
        if(flip):
            args +=["-vf", "transpose=cclock_flip,transpose=clock_flip"]
        #output video codec & preset
        args += ["-c:v","libx264", "-preset", "slow"]
        if(preserve_sound):
            args += ["-c:a", "copy"]
        else:
            args += ["-an"]
        args += ["-f", "mp4",  output_path]
        ix_vid+=1
        if(verbose):
            print("FFMPEG command: {0:s}".format(" ".join(args)))
        call(args)
        