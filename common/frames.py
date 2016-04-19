"""
/home/algomorph/Factory/calib_video_opencv/common/frames.py.
Created on Apr 4, 2016.
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
"""


def frames_to_mm_ss_ff(frames):
    mm = frames // 3600
    ss = frames // 60 - mm*60
    ff = int(frames % 60)
    return mm, ss, ff


def mm_ss_ff_to_frames(mm, ss, ff):
    return mm*3600+ss*60+ff

def seconds_to_frames(seconds):
    return round(seconds*60)
