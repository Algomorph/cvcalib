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

class SyncVideoApp(object):
    '''
    Application for syncing two videos 
    '''


    def __init__(self, args):
        '''
        Constructor
        '''
        self.args = args
        
    def run_sync(self):
        args = self.args
        offset = find_time_offset(args.videos[0], args.videos[1], args.folder)
        print(offset)
        
        