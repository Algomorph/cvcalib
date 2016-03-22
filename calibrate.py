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

import sys
import os.path as osp
import argparse as ap
from enum import Enum
from common.args import required_length
from yaml import load, dump
from calib.synced_calib_app import SyncedCalibApplication
from calib.unsynced_calib_app import UnsyncedCalibApplication

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

class Setting(Enum):
    settings_file = "settings_file"
    save_settings = "save_settings"
    
    folder = "folder"
    videos = "videos"
    preview_files = "preview_files"
    preview = "preview"
    
    board_width = "board_width"
    board_height = "board_height"
    board_square_size = "board_square_size"
    
    sharpness_threshold = "sharpness_threshold"
    difference_threshold = "difference_threshold"
    manual_filter = "manual_filter"
    frame_count_target = "frame_count_target"
    frame_numbers = "frame_numbers"
    
    corner_positions_file = "corners_file"
    save_corner_positions = "save_corner_positions"
    load_corner_positions = "load_corner_positions"
    
    max_iterations = "max_iterations"
    precalibrate_solo = "precalibrate_solo"
    stereo_only = "stereo_only"
    use_rational_model = "use_rational_model"
    use_tangential_coeffs = "use_tangential_coeffs"
    use_fisheye_model = "use_fisheye_model"
    
    output = "output"
    input_calibration = "input_calibration"
    
    #TODO: make a max_time_offset as well for convenience
    max_frame_offset = "max_frame_offset"
    
    skip_printing_output = "skip_printing_output"
    skip_saving_output = "skip_saving_output"

    filtered_image_folder = "filtered_image_folder"
    save_images = "save_images"
    load_images = "load_images"
    
    unsynced = "unsynced"

def main(argv=None):
    defaults = {
        Setting.settings_file.name:None,
        Setting.save_settings.name:False,
        
        Setting.folder.name:"./",
        Setting.videos.name: ["left.mp4","right.mp4"],
        Setting.preview_files.name:["left.png","right.png"],
        Setting.preview.name:False,
        
        Setting.board_width.name:9,
        Setting.board_height.name:6,
        Setting.board_square_size.name:0.0198888,
        
        Setting.sharpness_threshold.name:55,
        Setting.difference_threshold.name:0.4,
        Setting.manual_filter.name:False,
        Setting.frame_count_target.name:-1,
        Setting.frame_numbers.name:None,
        
        Setting.corner_positions_file.name:"corners.npz",
        Setting.save_corner_positions.name:False,
        Setting.load_corner_positions.name:False,
        
        Setting.max_iterations.name:30,
        Setting.precalibrate_solo.name:False,
        Setting.stereo_only.name:False,
        Setting.use_rational_model.name:False,
        Setting.use_tangential_coeffs.name:False,
        Setting.use_fisheye_model.name:False,
        
        Setting.output.name:None, 
        Setting.input_calibration.name:None,
        
        Setting.max_frame_offset.name:100,
        
        Setting.skip_printing_output.name:False,
        Setting.skip_saving_output.name:False,
        
        Setting.filtered_image_folder.name:"frames",
        Setting.save_images.name:False,
        Setting.load_images.name:False,
        
        Setting.unsynced.name: False
        }
    
    #============== STORAGE/RETRIEVAL OF SETTINGS =================================================#
    conf_parser = ap.ArgumentParser(description='Traverse two .mp4 stereo video files and '+
                               ' stereo_calibrate the cameras based on specially selected frames within.',
                               formatter_class=ap.RawDescriptionHelpFormatter, add_help=False)
    conf_parser.add_argument("-sf", "--" + Setting.settings_file.name, required = False, 
                        default=defaults[Setting.settings_file.name], 
                        help="File (absolute or relative path) where to save and/or load"+
                        " settings for the program in YAML format.")
    conf_parser.add_argument("-ss", "--" + Setting.save_settings.name, 
                        help="save (or update) setting file.", 
                        action="store_true", required = False, default=defaults[Setting.save_settings.name])
    
    args, remaining_argv = conf_parser.parse_known_args()
    defaults[Setting.save_settings.name] = args.save_settings
    if(args.settings_file):
        defaults[Setting.settings_file.name] = args.settings_file
        if(osp.isfile(args.settings_file)):
            file_stream = open(args.settings_file, "r", encoding="utf-8")
            config_defaults = load(file_stream, Loader=Loader)
            file_stream.close()
            for key,value in config_defaults.items():
                defaults[key] = value
        else:
            raise ValueError("Settings file not found at: {0:s}".format(args.settings_file))
    
    parser = ap.ArgumentParser(parents=[conf_parser])
    
    parser.add_argument("-f", "--" + Setting.folder.name, help="Path to root folder to work in", 
                        required=False, default=defaults[Setting.folder.name])
    parser.add_argument("-v", "--" + Setting.videos.name,metavar="VIDEO", nargs='+', action=required_length(1, 2),
                        help="input stereo video tuple (left, right) or a single video file,"+
                        " relative to the 'folder' argument", 
                        required=False, default=defaults[Setting.videos.name])
    
    #============== CALIBRATION PREVIEW ===========================================================#
    #TODO: test
    parser.add_argument("-cpf", "--" + Setting.preview_files.name, nargs='+', help="input frames to test"+
                        " calibration result (currently only for stereo)", 
                        required=False, default= ["left.png","right.png"], action=required_length(1, 2))
    parser.add_argument("-cp", "--" + Setting.preview.name, help="Test calibration result on left/right"+
                        " frame pair (currently only for stereo)", 
                        action = "store_true", required=False, default=defaults[Setting.preview.name])
    
    #============== BOARD DIMENSIONS ==============================================================#
    parser.add_argument("-bw", "--" + Setting.board_width.name, 
                        help="checkerboard inner corner count across (width)",
                        required = False, default=defaults[Setting.board_width.name], type=int)
    parser.add_argument("-bh", "--" + Setting.board_height.name, 
                        help="checkerboard inner corner count up (height)",
                        required = False,default=defaults[Setting.board_height.name], type=int)
    parser.add_argument("-bs", "--" + Setting.board_square_size.name, 
                        help="checkerboard square size, in meters", 
                        required = False, type=float, default=defaults[Setting.board_square_size.name])
    
    #============== FRAME FILTERING CONTROLS ======================================================#
    parser.add_argument("-fs", "--" + Setting.sharpness_threshold.name, 
                        help="sharpness threshold based on variance of "+
                        "Laplacian; used to filter out frames that are too blurry (default 55.0).", 
                        type=float, required = False, default=defaults[Setting.sharpness_threshold.name])
    parser.add_argument("-fd", "--" + Setting.difference_threshold.name, 
                        help="difference threshold: minimum average "
                        +" per-pixel difference (in range [0,1.0]) between current and previous frames to "
                        +"filter out frames that are too much alike (default: 0.4)", type=float, 
                        required = False, default=defaults[Setting.difference_threshold.name])
    parser.add_argument("-fm", "--" + Setting.manual_filter.name, 
                        help="pick which (pre-filtered)frames to use manually"+
                        " one-by-one (use 'a' key to approve)", required = False, action='store_true', 
                        default=defaults[Setting.manual_filter.name])
    parser.add_argument("-ft", "--" + Setting.frame_count_target.name, required=False, 
                        default=defaults[Setting.frame_count_target.name], type=int,
                        help="total number of frames (from either camera) to target for calibration.")
    parser.add_argument("-fn", "--" + Setting.frame_numbers.name, help="frame numbers .npz file with"+
                        " frame_numbers array."+
                        " If specified, program filters frame pairs based on these numbers instead of other"+
                        " criteria.",required=False, default=None)
    
    #============== STORAGE OF BOARD INNER CORNER POSITIONS =======================================#
    parser.add_argument("-pf", "--" + Setting.corner_positions_file.name,
                        help="file (relative to 'folder') where to load from / save to inner corner positions",
                         required = False, 
                        default=defaults[Setting.corner_positions_file.name])
    parser.add_argument("-ps", "--" + Setting.save_corner_positions.name, action='store_true',
                        help = "save the gathered locations of inner board corners.",
                        required = False, 
                        default=defaults[Setting.save_corner_positions.name])
    parser.add_argument("-pl", "--" + Setting.load_corner_positions.name, action='store_true',
                        help = "load the previously-gathered locations of inner board corners"+
                        " (skips gathering frame data)", 
                        required = False, 
                        default=defaults[Setting.load_corner_positions.name])
    
    #============== CALIBRATION & DISTORTION MODEL CONTROLS =======================================#
    parser.add_argument("-ci", "--" + Setting.max_iterations.name, 
                        help="maximum number of iterations for the stereo"+
                        " calibration (optimization) loop", type=int, required = False, 
                        default=defaults[Setting.max_iterations.name])
    parser.add_argument("-cs", "--" + Setting.precalibrate_solo.name, help="calibrate each camera "+
                        "individually (in case of stereo calibration) "+
                        "first, then perform stereo calibration",action='store_true', required = False, 
                        default=defaults[Setting.precalibrate_solo.name])
    parser.add_argument("-cso", "--" + Setting.stereo_only.name, 
                        help="Fix intrinsics and perform stereo calibration only."
                        +" Useful in conjunction with the " + Setting.input_calibration.name +
                        " option. Does nothing for single-camera calibration.", action='store_true', 
                        required=False, default=defaults[Setting.stereo_only.name])
    parser.add_argument("-cr", "--" + Setting.use_rational_model.name,
                        help="Use the newer OpenCV rational model (8 distorition coefficients"+
                        " w/ tangential ones, 6 without)", 
                        action='store_true', required = False, 
                        default=defaults[Setting.use_rational_model.name])
    parser.add_argument("-ct", "--" + Setting.use_tangential_coeffs.name, action='store_true',
                        help="Use tangential distortion coefficients (usually unnecessary)", 
                        required = False, default=defaults[Setting.use_tangential_coeffs.name])
    parser.add_argument("-cf", "--" + Setting.use_fisheye_model.name,
                        help="Use the fisheye distortion model (WARNING: OpenCV3 python bindings may still be broken!)", 
                        action='store_true', 
                        required = False, default=defaults[Setting.use_fisheye_model.name])
    
    #============== INPUT/OUTPUT CALIBRATION FILES ================================================#
    parser.add_argument("-cl", "--" + Setting.input_calibration.name, nargs='+', action=required_length(1, 2),
                        help="an existing calibration file to initialize calibration parameters (optional).",
                        required = False, default=defaults[Setting.input_calibration.name])
    parser.add_argument("-co", "--" + Setting.output.name, help="output file to store calibration results (relative to 'folder')", 
                        required = False, default=defaults[Setting.output.name])
    
    #============== MAXIMUM FRAME OFFSET ==========================================================#
    parser.add_argument("-mfo", "--" + Setting.max_frame_offset.name, 
                    help="Used for unsynced calibration only: maximum delay, in frames, between videos",
                    required = False, default=defaults[Setting.max_frame_offset.name], type=int)
    
    #============== SKIP CERTAIN OPERATIONS =======================================================#
    parser.add_argument("-skp", "--" + Setting.skip_printing_output.name, action='store_true', 
                        required = False, default= defaults[Setting.skip_printing_output.name])
    parser.add_argument("-sko", "--" + Setting.skip_saving_output.name, action='store_true', 
                        required = False, default= defaults[Setting.skip_saving_output.name])
    
    #============== FILTERED IMAGE/FRAME BACKUP & LOADING =========================================#
    parser.add_argument("-if", "--" + Setting.filtered_image_folder.name, help="filtered frames"+
                        " will be saved into this folder (relative to work folder specified by --folder)", 
                        required = False, default=defaults[Setting.filtered_image_folder.name])
    parser.add_argument("-is", "--" + Setting.save_images.name,
                        help="save images picked out for calibration", 
                        action='store_true', required = False, 
                        default= defaults[Setting.save_images.name])
    parser.add_argument("-il", "--" + Setting.load_images.name,
                        help="load images previously picked out for calibration (skips frame gathering)", 
                        action='store_true', required = False, 
                        default= defaults[Setting.load_images.name])
    
    #============== UNSYNCED =====================================================================#
    parser.add_argument("-u", "--" + Setting.unsynced.name, help="Use unsynced calibration mode. "+
                        "In unsynced calibration mode, multiple videos don't have to be "+
                        "synchronized at all. They just need to contain a long sequence of frames"+
                        "with the calibration board taken during the same session with all the "+
                        "cameras in static positions relative to each-other."+
                        "However, you must supply reliable intrinsics for each camera (see " +
                        Setting.input_calibration.name + " parameter). ",
                        action='store_true',required = False, default=defaults[Setting.unsynced.name])
    
    
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)

    if(args.save_settings and args.settings_file):
        setting_dict = vars(args)
        file_stream = open(args.settings_file, "w", encoding="utf-8")
        del setting_dict[Setting.save_settings.name]
        del setting_dict[Setting.settings_file.name]
        dump(setting_dict, file_stream, Dumper=Dumper)
        file_stream.close()
    
    if args.unsynced:
        app = UnsyncedCalibApplication(args)
        app.gather_frame_data()
        print(app.calibrate_time_variance())
    else:
        app = SyncedCalibApplication(args)
        app.gather_frame_data()
        app.run_calibration()
    
if __name__ == "__main__":
    sys.exit(main())
        
    
