#!/usr/bin/python3
"""
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

import sys
import os.path as osp
import argparse as ap
from enum import Enum
from common.args import required_length, string_arr
from yaml import load, dump
from calib.app_synced import ApplicationSynced
from calib.app_unsynced import ApplicationUnsynced
import re

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class Argument(object):
    def __init__(self, default,
                 nargs=1,
                 arg_type=str,
                 action='store',
                 arg_help="Documentation N/A",
                 console_only=False,
                 required=False,
                 shorthand=None):
        """
        @rtype: Argument
        @type name: str
        @param name: argument name -- to be used in both console and config file
        @type default: object
        @param default: the default value
        @type nargs: int | str
        @param nargs: number of arguments. See python documentation for ArgumentParser.add_argument.
        @type arg_type: type | str
        @param arg_type: type of value to expect during parsing
        @type action: str | function
        @param action: action to perform with the argument value during parsing
        @type arg_help: str
        @param arg_help: documentation for this argument
        @type console_only: bool
        @param console_only: whether the argument is for console use only or for both config file & console
        @type required: bool
        @param required: whether the argument is required
        @type shorthand: str
        @param shorthand: shorthand to use for argument in console
        """
        self.default = default
        self.required = required
        self.console_only = console_only
        self.nargs = nargs
        self.type = arg_type
        self.action = action
        self.help = arg_help
        if shorthand is None:
            self.shorthand = None
        else:
            self.shorthand = "-" + shorthand


class Setting(Enum):
    settings_file = Argument(None, '?', str, 'store',
                             "File (absolute or relative-to-execution path) where to save and/or " +
                             "load settings for the program in YAML format.",
                             console_only=True, required=False)
    save_settings = Argument(False, '?', 'bool_flag', 'store_true',
                             "Save (or update) setting file.",
                             console_only=True, required=False)

    folder = Argument("./", '?', str, 'store',
                      "Path to root folder to work in",
                      console_only=False, required=False)
    videos = Argument(["left.mp4", "right.mp4"], '+', string_arr, required_length(1, 10),
                      "Input videos. May be multiple videos for unsynced mode, a stereo video tuple (left, right), " +
                      "or a single video file, specified relative to the work folder (see 'folder' argument).",
                      console_only=False, required=False)

    preview = Argument(False, '?', 'bool_flag', 'store_true',
                       "Save (or update) setting file.",
                       console_only=False, required=False)
    preview_files = Argument(["left.png", "right.png"], '+', string_arr, required_length(1, 10),
                             "Test calibration result on left/right frame pair (currently only for stereo in synced " +
                             "mode).", console_only=False, required=False)

    board_width = Argument(9, '?', int, 'store',
                           "Checkerboard horizontal inner corner count (width in squares - 1).",
                           console_only=False, required=False)
    board_height = Argument(6, '?', int, 'store',
                            "Checkerboard vertial inner corner count (height in squares - 1).",
                            console_only=False, required=False)
    board_square_size = Argument(0.0198888, '?', float, 'store',
                                 "Checkerboard square size, in meters.",
                                 console_only=False, required=False)

    sharpness_threshold = Argument(55.0, '?', float, 'store',
                                   "Sharpness threshold based on variance of " +
                                   "Laplacian; used to filter out frames that are too blurry. Synced mode only.",
                                   console_only=False, required=False, shorthand="fs")
    difference_threshold = Argument(.4, '?', float, 'store',
                                    "Per-pixel difference (in range [0,1.0]) between current and previous frames to "
                                    + "filter out frames that are too much alike. Synced mode only.",
                                    console_only=False, required=False, shorthand="fd")
    manual_filter = Argument(False, '?', 'bool_flag', 'store_true',
                             "Pick which (pre-filtered)frames to use manually" +
                             "one-by-one (use 'a' key to approve). Synced mode only.",
                             console_only=False, required=False, shorthand="fm")
    frame_count_target = Argument(-1, '?', int, 'store',
                                  "Total number of frames (from either camera) to target for calibration usage." +
                                  "Synced mode only.",
                                  console_only=False, required=False, shorthand="ft")
    frame_number_filter = Argument(False, '?', 'bool_flag', 'store_true',
                                   "Use only frame numbers specified in the auxiliary data file. Synced mode only.",
                                   console_only=False, required=False, shorthand="fn")
    time_range_hint = Argument(None, 2, int, 'store',
                               "Look at frames only within this time range (in seconds) when seeking exact periods of" +
                               "calibration in all videos. A good hint will decrease the search time, but any frames " +
                               "outside the range hint will not be used. Unsynced mode only.",
                               console_only=False, required=False)

    aux_data_file = Argument("aux.npz", '?', str, 'store',
                             "File (relative to 'folder') where to load from and/or save to inner corner positions, " +
                             "calibration time ranges, frame numbers, and other auxiliary data.",
                             console_only=False, required=False, shorthand="df")
    save_calibration_intervals = Argument(False, '?', 'bool_flag', 'store_true',
                                          "Save the calculated time bounds of calibration period within the video for" +
                                          " future re-use.",
                                          console_only=False, required=False)
    load_calibration_intervals = Argument(False, '?', 'bool_flag', 'store_true',
                                          "Load the previously-determined time bounds of calibration period within " +
                                          "video (avoids potentially-long computation that seeks out the calibration " +
                                          "in the video)",
                                          console_only=False, required=False)
    # TODO rename to 'save/load_frame_data', b/c were also loading poses and other such things
    save_corner_positions = Argument(False, '?', 'bool_flag', 'store_true',
                                     "Save (or update) the gathered locations of inner board corners.",
                                     console_only=False, required=False)
    load_corner_positions = Argument(False, '?', 'bool_flag', 'store_true',
                                     "Load the previously-gathered locations of inner board corners " +
                                     "(skips gathering frame data)",
                                     console_only=False, required=False)

    max_iterations = Argument(100, '?', int, 'store',
                              "Maximum number of iterations for the stereo  for calibration (optimization) loop.",
                              console_only=False, required=False, shorthand="ci")
    precalibrate_solo = Argument(False, '?', 'bool_flag', 'store_true',
                                 "calibrate each camera individually (in case of stereo calibration) first, then " +
                                 "perform stereo calibration.",
                                 console_only=False, required=False, shorthand="cs")
    stereo_only = Argument(False, '?', 'bool_flag', 'store_true',
                           "Use in conjunction with the input_calibration option. " +
                           "Does nothing for single-camera calibration. Synced mode only.",
                           console_only=False, required=False, shorthand="cso")
    use_rational_model = Argument(False, '?', 'bool_flag', 'store_true',
                                  "Use the newer OpenCV rational model (8 distorition coefficients w/ tangential " +
                                  "ones, 6 without) as opposed to the old 3+2 polynomeal coefficient model.",
                                  console_only=False, required=False, shorthand="cr")
    use_tangential_coeffs = Argument(False, '?', 'bool_flag', 'store_true',
                                     "Use tangential distortion coefficients (usually unnecessary).",
                                     console_only=False, required=False, shorthand="ct")
    # TODO: test
    use_fisheye_model = Argument(False, '?', 'bool_flag', 'store_true',
                                 "Use the fisheye distortion model.",
                                 console_only=False, required=False, shorthand="cf")

    output = Argument(None, '?', str, 'store',
                      "Output file to store calibration results (relative to work folder, see 'folder' setting)",
                      console_only=False, required=False)
    input_calibration = Argument(None, '+', string_arr, required_length(1, 10),
                                 "Existing calibration file[s] to initialize calibration parameters. " +
                                 "Optional for synced mode, mandatory for unsynced mode.",
                                 console_only=False, required=False)

    skip_printing_output = Argument(False, '?', 'bool_flag', 'store_true',
                                    "Skip printing output.",
                                    console_only=False, required=False)
    skip_saving_output = Argument(False, '?', 'bool_flag', 'store_true',
                                  "Skip saving the output file. Usually, you don't want to skip that.",
                                  console_only=False, required=False)

    filtered_image_folder = Argument("frames", '?', str, 'store',
                                     "Filtered frames will be saved into this folder (relative to work folder " +
                                     "specified in 'folder'). Synced mode only.",
                                     console_only=False, required=False, shorthand="if")
    save_images = Argument(False, '?', 'bool_flag', 'store_true',
                           "Save images picked out for calibration. Synced mode only.",
                           console_only=False, required=False)
    load_images = Argument(False, '?', 'bool_flag', 'store_true',
                           "Load images previously picked out for calibration (skips frame gathering). Synced only.",
                           console_only=False, required=False)

    unsynced = Argument(False, '?', 'bool_flag', 'store_true',
                        "Used to find extrinsics between multiple unsynchronized cameras."
                        "The multiple videos need to contain a long sequence of frames" +
                        "with the calibration board taken during the same session with all " +
                        "cameras in static positions relative to each-other. However, you must supply reliable " +
                        "intrinsics for each camera (see input_calibration) and an appropriate max_frame_offset. ",
                        console_only=False, required=False)
    max_frame_offset = Argument(100, '?', int, 'store',
                                "Used for unsynced calibration only: maximum delay, in frames, between videos.",
                                console_only=False, required=False)
    @staticmethod
    def generate_missing_shorthands():
        for item in Setting:
            if item.value.shorthand is None:
                item.value.shorthand = "-" + "".join([item[1] for item in re.findall(r"(:?^|_)(\w)", item.name)])

    @staticmethod
    def generate_defaults_dict():
        dict = {}
        for item in Setting:
            dict[item.name] = item.value.default
        return dict

    @staticmethod
    def generate_parser(defaults, console_only=False, description="Description N/A", parents=None):
        """
        @type defaults: dict
        @param defaults: dictionary of default settings and their values.
        For a conf-file+console parser, these come from the config file. For a console-only parser, these are generated.
        @type console_only: bool
        @type description: str
        @param description: description of the program that uses the parser, to be used in the help file
        @type parents: list[argparse.ArgumentParser] | None

        """
        if console_only:
            parser = ap.ArgumentParser(description=description, formatter_class=ap.RawDescriptionHelpFormatter,
                                       add_help=False)
        else:
            if parents is None:
                raise ValueError("A conf-file+console parser requires at least a console-only parser as a parent.")
            parser = ap.ArgumentParser(parents=parents)

        for item in Setting:
            if (item.value.console_only and console_only) or (not item.value.console_only and not console_only):
                if item.value.type == 'bool_flag':
                    parser.add_argument(item.value.shorthand, '--' + item.name, action=item.value.action,
                                        default=defaults[item.name],required=item.value.required,
                                        help=item.value.help)
                else:
                    parser.add_argument(item.value.shorthand, '--' + item.name, action=item.value.action,
                                        type=item.value.type, nargs=item.value.nargs,required=item.value.required,
                                        default=defaults[item.name], help=item.value.help)
        if not console_only:
            parser.set_defaults(**defaults)
        return parser


class SettingOld(Enum):
    """
    Application settings
    """
    '''
    TODO: revise such that values contain everything: help, type, default, etc., to have the defaults dictionary
    & the argument parser itself be constructed via for loops, and everything is specified only once
    (see prelim work on Argument above)
    '''

    settings_file = "settings_file"
    save_settings = "save_settings"

    folder = "folder"
    videos = "cameras"
    preview_files = "preview_files"
    preview = "preview"

    board_width = "board_width"
    board_height = "board_height"
    board_square_size = "board_square_size"

    sharpness_threshold = "sharpness_threshold"
    difference_threshold = "difference_threshold"
    manual_filter = "manual_filter"
    frame_count_target = "frame_count_target"
    frame_number_filter = "frame_number_filter"
    time_range = "time_range"

    aux_data_file = "aux_data_file"
    save_calibration_intervals = "save_calibration_intervals"
    load_calibration_intervals = "load_calibration_intervals"
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

    # TODO: make a max_time_offset as well for convenience
    max_frame_offset = "max_frame_offset"

    skip_printing_output = "skip_printing_output"
    skip_saving_output = "skip_saving_output"

    filtered_image_folder = "filtered_image_folder"
    save_images = "save_images"
    load_images = "load_images"

    unsynced = "unsynced"


def generate_defaults():
    defaults = {
        SettingOld.settings_file.name:              None,
        SettingOld.save_settings.name:              False,

        SettingOld.folder.name:                     "./",
        SettingOld.videos.name:                     ["left.mp4", "right.mp4"],
        SettingOld.preview_files.name:              ["left.png", "right.png"],
        SettingOld.preview.name:                    False,

        SettingOld.board_width.name:                9,
        SettingOld.board_height.name:               6,
        SettingOld.board_square_size.name:          0.0198888,

        SettingOld.sharpness_threshold.name:        55,
        SettingOld.difference_threshold.name:       0.4,
        SettingOld.manual_filter.name:              False,
        SettingOld.frame_count_target.name:         -1,
        SettingOld.frame_number_filter.name:         False,
        SettingOld.time_range.name:                 None,

        SettingOld.aux_data_file.name:              "aux.npz",
        SettingOld.save_calibration_intervals.name: False,
        SettingOld.load_calibration_intervals.name: False,
        SettingOld.save_corner_positions.name:      False,
        SettingOld.load_corner_positions.name:      False,

        SettingOld.max_iterations.name:             100,
        SettingOld.precalibrate_solo.name:          False,
        SettingOld.stereo_only.name:                False,
        SettingOld.use_rational_model.name:         False,
        SettingOld.use_tangential_coeffs.name:      False,
        SettingOld.use_fisheye_model.name:          False,

        SettingOld.output.name:                     None,
        SettingOld.input_calibration.name:          None,

        SettingOld.max_frame_offset.name:           100,

        SettingOld.skip_printing_output.name:       False,
        SettingOld.skip_saving_output.name:         False,

        SettingOld.filtered_image_folder.name:      "frames",
        SettingOld.save_images.name:                False,
        SettingOld.load_images.name:                False,

        SettingOld.unsynced.name:                   False
    }
    return defaults


def generate_conf_parser(defaults):
    """
    Storage/retrieval of console-only settings
    """
    conf_parser = ap.ArgumentParser(description='Traverse two .mp4 stereo video files and  stereo_calibrate the ' +
                                                'cameras based on specially selected frames within.',
                                    formatter_class=ap.RawDescriptionHelpFormatter, add_help=False)

    conf_parser.add_argument("-sf", "--" + SettingOld.settings_file.name, required=False,
                             default=defaults[SettingOld.settings_file.name],
                             help="File (absolute or relative path) where to save and/or load" +
                                  " settings for the program in YAML format.")
    conf_parser.add_argument("-ss", "--" + SettingOld.save_settings.name,
                             help="save (or update) setting file.",
                             action="store_true", required=False, default=defaults[SettingOld.save_settings.name])
    return conf_parser


def generate_main_parser(defaults, conf_parser):
    """
    Storage/retrieval of regular settings
    """
    parser = ap.ArgumentParser(parents=[conf_parser])

    parser.add_argument("-f", "--" + SettingOld.folder.name, help="Path to root folder to work in",
                        required=False, default=defaults[SettingOld.folder.name])
    parser.add_argument("-v", "--" + SettingOld.videos.name, metavar="VIDEO", nargs='+',
                        action=required_length(1, 10), type=string_arr,
                        help="input stereo video tuple (left, right) or a single video file," +
                             " relative to the 'folder' argument",
                        required=False, default=defaults[SettingOld.videos.name])

    # ============== CALIBRATION PREVIEW ===========================================================#
    # TODO: test
    # TODO: remove preview setting, just use preview_files --> preview_images instead (None means off)
    parser.add_argument("-cpf", "--" + SettingOld.preview_files.name, nargs='+', help="input frames to test" +
                                                                                      " calibration result (currently only for stereo)",
                        required=False, default=["left.png", "right.png"],
                        action=required_length(1, 10), type=string_arr)
    parser.add_argument("-cp", "--" + SettingOld.preview.name, help="Test calibration result on left/right" +
                                                                    " frame pair (currently only for stereo)",
                        action="store_true", required=False, default=defaults[SettingOld.preview.name])

    # ============== BOARD DIMENSIONS ==============================================================#
    parser.add_argument("-bw", "--" + SettingOld.board_width.name,
                        help="checkerboard inner corner count across (width)",
                        required=False, default=defaults[SettingOld.board_width.name], type=int)
    parser.add_argument("-bh", "--" + SettingOld.board_height.name,
                        help="checkerboard inner corner count up (height)",
                        required=False, default=defaults[SettingOld.board_height.name], type=int)
    parser.add_argument("-bs", "--" + SettingOld.board_square_size.name,
                        help="checkerboard square size, in meters",
                        required=False, type=float, default=defaults[SettingOld.board_square_size.name])

    # ============== FRAME FILTERING CONTROLS ======================================================#
    parser.add_argument("-fs", "--" + SettingOld.sharpness_threshold.name,
                        help="sharpness threshold based on variance of " +
                             "Laplacian; used to filter out frames that are too blurry (default 55.0).",
                        type=float, required=False, default=defaults[SettingOld.sharpness_threshold.name])
    parser.add_argument("-fd", "--" + SettingOld.difference_threshold.name,
                        help="difference threshold: minimum average "
                             + " per-pixel difference (in range [0,1.0]) between current and previous frames to "
                             + "filter out frames that are too much alike (default: 0.4)", type=float,
                        required=False, default=defaults[SettingOld.difference_threshold.name])
    parser.add_argument("-fm", "--" + SettingOld.manual_filter.name,
                        help="pick which (pre-filtered)frames to use manually" +
                             " one-by-one (use 'a' key to approve)", required=False, action='store_true',
                        default=defaults[SettingOld.manual_filter.name])
    parser.add_argument("-ft", "--" + SettingOld.frame_count_target.name, required=False,
                        default=defaults[SettingOld.frame_count_target.name], type=int,
                        help="total number of frames (from either camera) to target for calibration.")
    parser.add_argument("-fn", "--" + SettingOld.frame_number_filter.name, help="frame numbers .npz file with" +
                                                                               " frame_numbers array." +
                                                                               " If specified, program filters frame pairs " +
                                                                               " based on these numbers instead of other" +
                                                                               " criteria.",
                        required=False, default=defaults[SettingOld.frame_number_filter.name])
    parser.add_argument("-tr", "--" + SettingOld.time_range.name,
                        help="(Approximate) time range (seconds) for the calibration seeking algorithm in 'unsynced' mode.",
                        nargs=2, type=int)

    # ============== STORAGE OF AUXILIARY DATA ===================================================#
    parser.add_argument("-pf", "--" + SettingOld.aux_data_file.name,
                        help="file (relative to 'folder') where to load from / save to inner corner positions",
                        required=False,
                        default=defaults[SettingOld.aux_data_file.name])
    parser.add_argument("-sci", "--" + SettingOld.save_calibration_intervals.name, action='store_true',
                        help="save the calculated time bounds of calibration period within the video for future re-use.",
                        required=False,
                        default=defaults[SettingOld.save_calibration_intervals.name])
    parser.add_argument("-lci", "--" + SettingOld.load_corner_positions.name, action='store_true',
                        help="load the previously-gathered locations of inner board corners" +
                             " (skips gathering frame data)",
                        required=False,
                        default=defaults[SettingOld.load_corner_positions.name])
    parser.add_argument("-sp", "--" + SettingOld.save_corner_positions.name, action='store_true',
                        help="save the gathered locations of inner board corners.",
                        required=False,
                        default=defaults[SettingOld.save_corner_positions.name])
    parser.add_argument("-lp", "--" + SettingOld.load_calibration_intervals.name, action='store_true',
                        help="load the previously-determined time bounds of calibration period within video" +
                             " (avoids potentially-long computation that seeks out the calibration in the video)",
                        required=False,
                        default=defaults[SettingOld.load_calibration_intervals.name])

    # ============== CALIBRATION & DISTORTION MODEL CONTROLS =======================================#
    parser.add_argument("-ci", "--" + SettingOld.max_iterations.name,
                        help="maximum number of iterations for the stereo" +
                             " calibration (optimization) loop", type=int, required=False,
                        default=defaults[SettingOld.max_iterations.name])
    parser.add_argument("-cs", "--" + SettingOld.precalibrate_solo.name, help="calibrate each camera " +
                                                                              "individually (in case of stereo calibration) " +
                                                                              "first, then perform stereo calibration",
                        action='store_true', required=False,
                        default=defaults[SettingOld.precalibrate_solo.name])
    parser.add_argument("-cso", "--" + SettingOld.stereo_only.name,
                        help="Fix intrinsics and perform stereo calibration only."
                             + " Useful in conjunction with the " + SettingOld.input_calibration.name +
                             " option. Does nothing for single-camera calibration.", action='store_true',
                        required=False, default=defaults[SettingOld.stereo_only.name])
    parser.add_argument("-cr", "--" + SettingOld.use_rational_model.name,
                        help="Use the newer OpenCV rational model (8 distorition coefficients" +
                             " w/ tangential ones, 6 without)",
                        action='store_true', required=False,
                        default=defaults[SettingOld.use_rational_model.name])
    parser.add_argument("-ct", "--" + SettingOld.use_tangential_coeffs.name, action='store_true',
                        help="Use tangential distortion coefficients (usually unnecessary)",
                        required=False, default=defaults[SettingOld.use_tangential_coeffs.name])
    parser.add_argument("-cf", "--" + SettingOld.use_fisheye_model.name,
                        help="Use the fisheye distortion model (WARNING: OpenCV3 python bindings may still be broken!)",
                        action='store_true',
                        required=False, default=defaults[SettingOld.use_fisheye_model.name])

    # ============== INPUT/OUTPUT CALIBRATION FILES ================================================#
    parser.add_argument("-cl", "--" + SettingOld.input_calibration.name, nargs='+', action=required_length(1, 10),
                        type=string_arr,
                        help="an existing calibration file to initialize calibration parameters (optional).",
                        required=False, default=defaults[SettingOld.input_calibration.name])
    parser.add_argument("-co", "--" + SettingOld.output.name,
                        help="output file to store calibration results (relative to 'folder')",
                        required=False, default=defaults[SettingOld.output.name])

    # ============== MAXIMUM FRAME OFFSET ==========================================================#
    parser.add_argument("-mfo", "--" + SettingOld.max_frame_offset.name,
                        help="Used for unsynced calibration only: maximum delay, in frames, between cameras",
                        required=False, default=defaults[SettingOld.max_frame_offset.name], type=int)

    # ============== SKIP CERTAIN OPERATIONS =======================================================#
    parser.add_argument("-skp", "--" + SettingOld.skip_printing_output.name, action='store_true',
                        required=False, default=defaults[SettingOld.skip_printing_output.name])
    parser.add_argument("-sko", "--" + SettingOld.skip_saving_output.name, action='store_true',
                        required=False, default=defaults[SettingOld.skip_saving_output.name])

    # ============== FILTERED IMAGE/FRAME BACKUP & LOADING =========================================#
    parser.add_argument("-if", "--" + SettingOld.filtered_image_folder.name,
                        help="filtered frames will be saved into this folder (relative to work folder specified in " +
                             " '--folder')",
                        required=False, default=defaults[SettingOld.filtered_image_folder.name])
    parser.add_argument("-is", "--" + SettingOld.save_images.name,
                        help="save images picked out for calibration",
                        action='store_true', required=False,
                        default=defaults[SettingOld.save_images.name])
    parser.add_argument("-il", "--" + SettingOld.load_images.name,
                        help="load images previously picked out for calibration (skips frame gathering)",
                        action='store_true', required=False,
                        default=defaults[SettingOld.load_images.name])

    # ============== UNSYNCED =====================================================================#
    parser.add_argument("-u", "--" + SettingOld.unsynced.name,
                        help="Use unsynced calibration mode. " +
                             "In unsynced calibration mode, multiple cameras don't have to be " +
                             "synchronized at all. They just need to contain a long sequence of frames" +
                             "with the calibration board taken during the same session with all the " +
                             "cameras in static positions relative to each-other." +
                             "However, you must supply reliable intrinsics for each camera (see " +
                             SettingOld.input_calibration.name + " parameter). ",
                        action='store_true', required=False, default=defaults[SettingOld.unsynced.name])

    parser.set_defaults(**defaults)
    return parser


def main():
    Setting.generate_missing_shorthands()
    defaults = Setting.generate_defaults_dict()
    conf_parser = \
        Setting.generate_parser(defaults, console_only=True, description=
                                "Use one or more .mp4 video files to perform calibration: " +
                                "find the cameras' intrinsics and/or extrinsics.")

    # ============== STORAGE/RETRIEVAL OF CONSOLE SETTINGS ===========================================#
    args, remaining_argv = conf_parser.parse_known_args()
    defaults[Setting.save_settings.name] = args.save_settings
    if args.settings_file:
        defaults[Setting.settings_file.name] = args.settings_file
        if osp.isfile(args.settings_file):
            file_stream = open(args.settings_file, "r", encoding="utf-8")
            config_defaults = load(file_stream, Loader=Loader)
            file_stream.close()
            for key, value in config_defaults.items():
                defaults[key] = value
        else:
            raise ValueError("Settings file not found at: {0:s}".format(args.settings_file))

    parser = Setting.generate_parser(defaults, parents=[conf_parser])
    args = parser.parse_args(remaining_argv)

    if args.save_settings and args.settings_file:
        setting_dict = vars(args)
        file_stream = open(args.settings_file, "w", encoding="utf-8")
        del setting_dict[SettingOld.save_settings.name]
        del setting_dict[SettingOld.settings_file.name]
        dump(setting_dict, file_stream, Dumper=Dumper)
        file_stream.close()

    if args.unsynced:
        app = ApplicationUnsynced(args)
        app.find_calibration_intervals()
        # app.gather_frame_data()
        # print(app.calibrate_time_variance())
    else:
        app = ApplicationSynced(args)
        app.gather_frame_data()
        app.run_calibration()


if __name__ == "__main__":
    sys.exit(main())
