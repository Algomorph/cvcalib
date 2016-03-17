'''
/home/algomorph/Factory/calib_video_opencv/args.py.
Created on Feb 9, 2016.
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

import argparse
from ast import literal_eval

def required_length(nmin,nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, conf_parser, args, values, option_string=None):
            if(type(values) == str):
                values = [values]
            if not nmin<=len(values)<=nmax:
                print(values)
                msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest,nmin=nmin,nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength

def boolarg(s):
    error_msg = "Invalid 'boolarg' format. Accepted values: True, False, 0, 1"
    try:
        data = literal_eval(s)
        if data != True and data != False:
            raise argparse.ArgumentTypeError(error_msg)
    except:  # TODO: avoid bare except and handle more specific errors
        raise argparse.ArgumentTypeError(error_msg)
    return data     