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
from yaml import load, dump
from multistereo.stereo_matcher_app import StereoMatcherApp
import re

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class Argument(object):
    def __init__(self, default,
                 nargs='?',
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


# TODO: investigate enum inheritance. There is too much duplicate code between this script file and others, like
# sync_based_on_audio.py and multistereo.py

class Setting(Enum):
    # ================= SETTING FILE STORAGE ==========================================================================#
    settings_file = Argument(None, '?', str, 'store',
                             "File (absolute or relative-to-execution path) where to save and/or " +
                             "load settings for the program in YAML format.",
                             console_only=True, required=False)
    save_settings = Argument(False, '?', 'bool_flag', 'store_true',
                             "Save (or update) setting file.",
                             console_only=True, required=False)
    # ================= WORK FOLDER, INPUT & OUTPUT FILES =============================================================#
    folder = Argument("./", '?', str, 'store',
                      "Path to root folder to work in. If set to '!settings_file_location' and a " +
                      " settings file is provided, will be set to the location of the settings file.",
                      console_only=False, required=False)
    images = Argument(["left.png", "right.png"], nargs=2,
                      arg_help="Paths from work folder to left & right stereo images.")

    input_calibration = Argument(None,
                      arg_help="Path from work folder to left & right calibration files.")

    output = Argument("disparity.png", arg_help="Name of the output disparity image.")

    preview = Argument(False, arg_type='bool_flag', arg_help="Preview the generated disparity map before saving.")

    @staticmethod
    def generate_missing_shorthands():
        for item in Setting:
            if item.value.shorthand is None:
                item.value.shorthand = "-" + "".join([item[1] for item in re.findall(r"(:?^|_)(\w)", item.name)])

    @staticmethod
    def generate_defaults_dict():
        """
        @rtype: dict
        @return: dictionary of Setting defaults
        """
        dict = {}
        for item in Setting:
            dict[item.name] = item.value.default
        return dict

    @staticmethod
    def generate_parser(defaults, console_only=False, description="Description N/A", parents=None):
        """
        @rtype: argparse.ArgumentParser
        @return: either a console-only or a config_file+console parser using the specified defaults and, optionally,
        parents.
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
                                        default=defaults[item.name], required=item.value.required,
                                        help=item.value.help)
                else:
                    parser.add_argument(item.value.shorthand, '--' + item.name, action=item.value.action,
                                        type=item.value.type, nargs=item.value.nargs, required=item.value.required,
                                        default=defaults[item.name], help=item.value.help)
        if not console_only:
            parser.set_defaults(**defaults)
        return parser


def load_app_from_config(path):
    """
    Generate app directly from config file, bypassing command line settings (useful for testing in ipython)
    """
    Setting.generate_missing_shorthands()
    defaults = Setting.generate_defaults_dict()
    if osp.isfile(path):
        file_stream = open(path, "r", encoding="utf-8")
        config_defaults = load(file_stream, Loader=Loader)
        file_stream.close()
        for key, value in config_defaults.items():
            defaults[key] = value
    else:
        raise ValueError("Settings file not found at: {0:s}".format(path))
    args = ap.Namespace()
    for key, value in defaults.items():
        args.__dict__[key] = value

    app = StereoMatcherApp(args)

    return app


def main():
    Setting.generate_missing_shorthands()
    defaults = Setting.generate_defaults_dict()
    conf_parser = \
        Setting.generate_parser(defaults, console_only=True, description=
        "Test stereo algorithms on two image files.")

    # ============== STORAGE/RETRIEVAL OF CONSOLE SETTINGS ===========================================#
    args, remaining_argv = conf_parser.parse_known_args()
    defaults[Setting.save_settings.name] = args.save_settings
    if args.settings_file:
        defaults[Setting.settings_file.name] = args.settings_file
        if osp.isfile(args.settings_file):
            file_stream = open(args.settings_file, "r", encoding="utf-8")
            config_defaults = load(file_stream, Loader=Loader)
            file_stream.close()
            if config_defaults:
                for key, value in config_defaults.items():
                    defaults[key] = value
        else:
            raise ValueError("Settings file not found at: {0:s}".format(args.settings_file))

    parser = Setting.generate_parser(defaults, parents=[conf_parser])
    args = parser.parse_args(remaining_argv)

    # process "special" setting values
    if args.folder == "!settings_file_location":
        if args.settings_file and osp.isfile(args.settings_file):
            args.folder = osp.dirname(args.settings_file)

    # save settings if prompted to do so
    if args.save_settings and args.settings_file:
        setting_dict = vars(args)
        file_stream = open(args.settings_file, "w", encoding="utf-8")
        file_name = setting_dict[Setting.save_settings.name]
        del setting_dict[Setting.save_settings.name]
        del setting_dict[Setting.settings_file.name]
        dump(setting_dict, file_stream, Dumper=Dumper)
        file_stream.close()
        setting_dict[Setting.save_settings.name] = file_name
        setting_dict[Setting.settings_file.name] = True

    app = StereoMatcherApp(args)
    app.disparity2()


if __name__ == "__main__":
    sys.exit(main())
